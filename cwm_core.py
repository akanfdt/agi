from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import re

import torch

from cwm_context import ContextField
from cwm_gravity import GravityField
from cwm_imitation import ImitationTrainer
from cwm_learning import LearningMixin
from cwm_memory import MemoryMixin
from cwm_orbit import OrbitMemory
from cwm_points import PointStore
from cwm_predictor import Predictor
from cwm_shockwave import ShockWaveMixin
from cwm_types import Anchor, ContextState, CWMSpec, QAMemoryState, StatsState, l2_normalize


class CWMCore(MemoryMixin, ShockWaveMixin, LearningMixin):
    def __init__(self, vocab: List[str], spec: Optional[CWMSpec] = None):
        self.spec = spec or CWMSpec()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0

        self.points = PointStore(spec=self.spec, device=self.device)
        self.points.init_vocab(vocab)
        self.gravity = GravityField(spec=self.spec)
        self.context = ContextField(spec=self.spec, state=ContextState())
        self.orbits = OrbitMemory(spec=self.spec)
        self.imitation = ImitationTrainer(spec=self.spec)
        self.predictor = Predictor(spec=self.spec, device=self.device)
        self.qa = QAMemoryState()
        self.stats = StatsState()
        self.raw_line_memory: List[Dict[str, Any]] = []
        self.compressed_line_memory: List[Dict[str, Any]] = []
        self.shock_waves: List[Dict[str, Any]] = []

        self.similarity_threshold = self.spec.similarity_merge
        self.imitation_noise = self.spec.imitation_noise_init
        self.stabilization_steps = max(1, int(len(vocab) * self.spec.stabilization_ratio)) if vocab else 0
        self.dim_importance = torch.zeros(self.spec.dim, device=self.device)

        self._cache_tokens: List[str] = []
        self._cache_index: Dict[str, int] = {}
        self._cache_matrix: Optional[torch.Tensor] = None
        self._cache_dirty = True

    @property
    def anchors(self) -> Dict[str, Anchor]:
        return self.points.anchors

    @property
    def char_anchors(self) -> Dict[str, Anchor]:
        return self.points.char_anchors

    @property
    def error_history(self) -> List[float]:
        return self.stats.error_history

    @property
    def error_count(self) -> int:
        return self.stats.error_count

    @property
    def error_mean(self) -> float:
        return self.stats.error_mean

    @property
    def error_m2(self) -> float:
        return self.stats.error_m2

    @property
    def ema_error(self) -> float:
        return self.stats.ema_error

    @property
    def last_error(self) -> float:
        return self.stats.last_error

    @property
    def last_near_dist(self) -> float:
        return self.stats.last_near_dist

    @property
    def last_context_error(self) -> float:
        return self.stats.last_context_error

    @property
    def context_path(self) -> List[Tuple[str, int]]:
        return self.context.state.path

    @property
    def context_signatures(self) -> List[Tuple[Tuple[str, ...], int]]:
        return self.context.state.signatures

    def _mark_cache_dirty(self) -> None:
        self._cache_dirty = True

    def _ensure_cache(self) -> None:
        if self._cache_matrix is not None and not self._cache_dirty:
            return
        self._cache_tokens = list(self.points.anchors.keys())
        self._cache_index = {tok: i for i, tok in enumerate(self._cache_tokens)}
        if not self._cache_tokens:
            self._cache_matrix = None
            self._cache_dirty = False
            return
        self._cache_matrix = torch.stack([self.points.anchors[tok].vec for tok in self._cache_tokens], dim=0).to(self.device)
        self._cache_dirty = False

    def alpha(self) -> float:
        return self._scheduled_value(self.spec.alpha_init, self.spec.alpha_final)

    def sigma(self) -> float:
        return max(1e-4, self._scheduled_value(self.spec.sigma_init, self.spec.sigma_final))

    def _scheduled_value(self, start: float, end: float) -> float:
        start = max(1e-6, start)
        end = max(1e-6, end)
        if self.spec.total_steps <= 0:
            return start
        ratio = min(1.0, self.step / self.spec.total_steps)
        return start * ((end / start) ** ratio)

    def cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b).clamp(-1.0, 1.0).item())

    def dist_cos(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return 1.0 - self.cosine_sim(a, b)

    def dist_euclid(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.norm(a - b).item())

    def error_threshold(self) -> float:
        if self.stats.error_count <= 1:
            return max(0.5, self.stats.ema_error or 1.0)
        variance = self.stats.error_m2 / max(1, self.stats.error_count - 1)
        std = math.sqrt(max(0.0, variance))
        return max(1e-4, self.stats.ema_error + std)

    def stop_threshold(self, sigma: float, strengths: Optional[torch.Tensor] = None) -> float:
        if strengths is None or strengths.numel() == 0:
            return math.exp(-0.5)
        mean = float(strengths.mean().item())
        std = float(strengths.std(unbiased=False).item())
        return max(self.spec.min_activation_for_update, min(1.0, mean + std))

    def add_anchor_near(self, token: str, base_vec: torch.Tensor) -> None:
        if self.points.add_anchor_near(token, base_vec):
            self._mark_cache_dirty()

    def _ensure_anchor_exists(self, token: str) -> None:
        existed = token in self.points.anchors
        self.points.ensure_anchor(token)
        if not existed and token in self.points.anchors:
            self._mark_cache_dirty()

    def token_vector(self, tokens: List[str]) -> Optional[torch.Tensor]:
        return self.points.token_vector(tokens)

    def context_vector(self) -> Optional[torch.Tensor]:
        anchor_map = {token: anchor.vec for token, anchor in self.points.anchors.items()}
        return self.context.build_context_vector(anchor_map, self.step, self.device)

    def advance_context(self, tokens: List[str]) -> None:
        for tok in tokens:
            self._ensure_anchor_exists(tok)
            self.context.advance_token(tok, self.step)

    def _qa_key(self, tokens: List[str]) -> Optional[Tuple[str, ...]]:
        if not tokens:
            return None
        key = tuple(tokens[-self.spec.qa_key_len :])
        return key if key else None

    def update_qa_memory(self, question_tokens: List[str], answer_tokens: List[str]) -> None:
        key = self._qa_key(question_tokens)
        ans_vec = self.token_vector(answer_tokens)
        if key is None or ans_vec is None:
            return
        if key in self.qa.memory:
            prev = self.qa.memory[key]
            decay = self.spec.qa_memory_decay
            self.qa.memory[key] = l2_normalize(prev * decay + ans_vec * (1.0 - decay))
            self.qa.counts[key] = self.qa.counts.get(key, 1) + 1
        else:
            self.qa.memory[key] = ans_vec
            self.qa.counts[key] = 1

    # sentence memory 메서드는 MemoryMixin에서 제공
    # (reset_line_memory, sentence_summary_vector, observe_sentence,
    #  summary_memory_vector, summary_prior, ...)

    def reinforce_dialogue(self, question_tokens: List[str], answer_tokens: List[str]) -> None:
        """
        질문 문장의 느낌 F_q → 답변 문장의 느낌 F_a 흐름을 학습.

        핵심 변화:
          - 이전: 질문 토큰 평균 벡터로 답변 토큰을 당김 (토큰 단위)
          - 지금: 질문 느낌 F_q → 답변 느낌 F_a 방향으로
                  답변 토큰들을 당기고 gravity도 느낌 단위로 기록
          - 효과: "나 슬퍼(F_q) → 왜?(F_a)" 패턴이
                  토큰이 아니라 느낌 공간에 새겨짐
        """
        # 질문과 답변의 느낌 벡터 계산
        F_q = self.sentence_summary_vector(question_tokens)
        F_a = self.sentence_summary_vector(answer_tokens)
        if F_q is None or F_a is None:
            return

        alpha = self.alpha() * self.spec.dialogue_strength

        # 답변 토큰들을 F_q 방향이 아니라 F_a 방향으로 당김
        # (답변의 느낌을 강화 — 답변이 자기 느낌을 더 선명하게 가지도록)
        for tok in answer_tokens:
            self._ensure_anchor_exists(tok)
            self.points.update_vector(tok, F_a, alpha, self.step)
            self._add_importance(tok, self.spec.dialogue_strength)

        # 질문 토큰들도 F_q 방향으로 당김
        # (질문이 자기 느낌을 더 선명하게 가지도록)
        for tok in question_tokens:
            self._ensure_anchor_exists(tok)
            self.points.update_vector(tok, F_q, alpha * 0.5, self.step)

        # 느낌 흐름 gravity: F_q 공간 근처 토큰 → F_a 공간 근처 토큰
        # "이 느낌 다음엔 저 느낌이 온다"를 gravity에 기록
        # F_q와 가장 가까운 질문 핵심 토큰 → F_a와 가장 가까운 답변 핵심 토큰
        q_core_tok = self._closest_token_in_candidates(F_q, question_tokens)
        a_core_tok = self._closest_token_in_candidates(F_a, answer_tokens)
        if q_core_tok is not None and a_core_tok is not None:
            self.gravity.reinforce_pair(
                q_core_tok, a_core_tok,
                self.spec.gravity_reinforce_amount * self.spec.dialogue_strength * 2.0,
            )

        self._mark_cache_dirty()

    def _apply_qa_hint(self, vec: torch.Tensor) -> torch.Tensor:
        if not self.qa.memory or self.spec.qa_weight <= 0.0:
            return vec
        matrix = torch.stack(list(self.qa.memory.values()), dim=0).to(self.device)
        sims = torch.mv(matrix, vec)
        weights = torch.softmax(sims, dim=0)
        qa_vec = l2_normalize(torch.sum(matrix * weights.unsqueeze(-1), dim=0))
        weight = min(1.0, max(0.0, self.spec.qa_weight))
        return l2_normalize(vec * (1.0 - weight) + qa_vec * weight)

    def _input_summary_vector(self, tokens: List[str]) -> Optional[torch.Tensor]:
        return self.sentence_summary_vector(tokens)

    def _context_summary_vector(self, current_step: Optional[int] = None, state: Optional[ContextState] = None) -> Optional[torch.Tensor]:
        step = self.step if current_step is None else current_step
        if state is None:
            return self.context_vector()
        temp_context = ContextField(spec=self.spec, state=state)
        anchor_map = {token: anchor.vec for token, anchor in self.points.anchors.items()}
        return temp_context.build_context_vector(anchor_map, step, self.device)

    def _summary_core_vector(
        self,
        fallback_token: Optional[str],
        recent_tokens: List[str],
        current_step: Optional[int] = None,
        state: Optional[ContextState] = None,
    ) -> Optional[torch.Tensor]:
        parts: List[torch.Tensor] = []
        weights: List[float] = []

        input_summary = self._input_summary_vector(recent_tokens)
        memory_summary = self.summary_memory_vector()
        context_summary = self._context_summary_vector(current_step=current_step, state=state)

        if input_summary is not None:
            parts.append(input_summary.to(self.device))
            weights.append(self.spec.query_weight_input)
        if memory_summary is not None:
            parts.append(memory_summary.to(self.device))
            weights.append(self.spec.query_weight_memory)
        if context_summary is not None:
            parts.append(context_summary.to(self.device))
            weights.append(self.spec.query_weight_context)

        if fallback_token is not None:
            self._ensure_anchor_exists(fallback_token)
            fb_vec = self.points.anchors[fallback_token].vec
            parts.append(fb_vec.to(self.device))
            weights.append(self.spec.query_weight_fallback)

        if not parts:
            return None

        core = torch.stack([vec * weight for vec, weight in zip(parts, weights)], dim=0).sum(dim=0)
        core = l2_normalize(core / max(sum(weights), 1e-8))
        return self._apply_qa_hint(core)

    def _context_query_vector(self, fallback_token: Optional[str]) -> Optional[torch.Tensor]:
        recent_tokens = self.context.recent_tokens(self.spec.context_window)
        return self._summary_core_vector(fallback_token=fallback_token, recent_tokens=recent_tokens)

    def _query_vector_for_state(self, state: ContextState, current_step: int, fallback_token: Optional[str]) -> Optional[torch.Tensor]:
        recent_tokens = [token for token, _ in state.path[-self.spec.context_window :]]
        return self._summary_core_vector(
            fallback_token=fallback_token,
            recent_tokens=recent_tokens,
            current_step=current_step,
            state=state,
        )

    @staticmethod
    def is_emittable_token(tok: str) -> bool:
        """
        출력으로 내보낼 수 있는 토큰인지 판단.
        내부 메타 토큰은 학습에는 쓰이지만 생성 결과로 나오면 안 됨.

        제외 대상:
        - '▁'  단독 word-start 마커 (decode시 공백/빈 문자열로 렌더링됨)
        - '__loop_' 포함 토큰 (CWM 내부 루프 앵커)
        - '<unk>' 등 특수 토큰
        구두점(! ? . ,)은 문장 끝 패턴 학습을 위해 허용.
        과출력 방지는 importance cap과 gravity 가중치로 제어.
        """
        if tok in ("▁", "<unk>", "", " "):
            return False
        if "__loop_" in tok:
            return False
        if tok.startswith("<") and tok.endswith(">"):
            return False
        punct = set("!?.,;:-~/@#$%^&*()[]{}|'\"+=<>")
        if len(tok) == 1 and (tok in punct or tok.isascii() and tok.isalnum()):
            return False
        if tok.startswith("▁") and len(tok) == 2:
            tail = tok[1]
            if tail in punct or tail.isascii() and tail.isalnum():
                return False
        return True

    def _add_importance(self, token: str, amount: float) -> None:
        anchor = self.points.anchors.get(token)
        if anchor is None or amount == 0.0:
            return
        anchor.importance += amount

    def _closest_token_in_candidates(self, vec: torch.Tensor, candidates: List[str]) -> Optional[str]:
        best_token: Optional[str] = None
        best_sim = -float("inf")
        seen = set()
        for tok in candidates:
            if tok in seen:
                continue
            seen.add(tok)
            anchor = self.points.anchors.get(tok)
            if anchor is None:
                continue
            sim = self.cosine_sim(anchor.vec.to(self.device), vec.to(self.device))
            if sim > best_sim:
                best_sim = sim
                best_token = tok
        return best_token

    def _filter_output_tokens(self, tokens: List[str], scores: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        keep = [i for i, tok in enumerate(tokens) if self.is_emittable_token(tok)]
        if len(keep) == len(tokens):
            return tokens, scores
        if not keep:
            return [], scores[:0]
        idx = torch.tensor(keep, device=scores.device, dtype=torch.long)
        return [tokens[i] for i in keep], scores.index_select(0, idx)

    def _repeat_pattern_similar(self) -> bool:
        sig = self.context.repeat_signature()
        if not sig:
            return False
        count = sum(1 for cand, _ in self.context.state.signatures if cand == sig)
        repeated = count >= max(1, self.spec.repeat_theta)
        beta = self.spec.ema_beta
        self.stats.repeat_ema = beta * self.stats.repeat_ema + (1.0 - beta) * (1.0 if repeated else 0.0)
        return repeated

    def scores_from_vector(self, vec: torch.Tensor) -> Optional[Tuple[List[str], torch.Tensor]]:
        self._ensure_cache()
        if self._cache_matrix is None:
            return None
        scores = self.predictor.score_candidates(
            l2_normalize(vec.to(self.device)),
            self._cache_matrix,
            gravity_prior=torch.zeros(len(self._cache_tokens), dtype=torch.float32),
            summary_prior=torch.zeros(len(self._cache_tokens), dtype=torch.float32),
            orbit_prior=torch.zeros(len(self._cache_tokens), dtype=torch.float32),
            imitation_prior=torch.zeros(len(self._cache_tokens), dtype=torch.float32),
            repeat_penalty=self.stats.repeat_ema,
        )
        scores, self.stats.score_mean, self.stats.score_m2, self.stats.score_count = self.predictor.normalize_scores(
            scores,
            self.stats.score_mean,
            self.stats.score_m2,
            self.stats.score_count,
            self.stats.repeat_ema,
        )
        tokens, scores = self._filter_output_tokens(self._cache_tokens, scores)
        if not tokens:
            return None
        return tokens, scores

    def _transition_prior_from_path(self, path: List[Tuple[str, int]]) -> torch.Tensor:
        if not self._cache_tokens:
            return torch.empty(0, dtype=torch.float32)
        recent = path[-self.spec.orbit_max_length :]
        if not recent:
            return torch.zeros(len(self._cache_tokens), dtype=torch.float32)

        scores = torch.zeros(len(self._cache_tokens), dtype=torch.float32)
        denom = float(max(1, len(recent)))
        for pos, (src, _) in enumerate(recent):
            # Prefer actual movement along recently active forward links.
            recency = (pos + 1) / denom
            hop_weight = self.spec.transition_hop_base + self.spec.transition_hop_range * recency
            for cand_idx, cand in enumerate(self._cache_tokens):
                forward = self.gravity.get_forward_gravity(src, cand)
                if forward > 0.0:
                    scores[cand_idx] += forward * hop_weight
        scores = scores / denom
        return torch.tanh(scores)

    def _compute_scores(
        self,
        query: torch.Tensor,
        context_field: ContextField,
        path: List[Tuple[str, int]],
        current_step: int,
        update_stats: bool,
    ) -> Optional[Tuple[List[str], torch.Tensor]]:
        """
        scores_from_context / _scores_from_state 공통 코어.
        prior 계산 → score_candidates → normalize → filter.
        update_stats=True이면 score_mean/m2/count를 갱신 (학습/추론 경로),
        False이면 현재 통계 기준으로만 정규화 (시뮬레이션 경로).
        """
        self._ensure_cache()
        if self._cache_matrix is None:
            return None
        active_tokens = context_field.active_tokens(current_step)
        gravity_prior = self.gravity.context_prior(active_tokens, self._cache_tokens)
        transition_prior = self._transition_prior_from_path(path)
        orbit_map = self.orbits.query(context_field.recent_tokens(self.spec.orbit_max_length))
        orbit_prior = torch.tensor([orbit_map.get(tok, 0.0) for tok in self._cache_tokens], dtype=torch.float32)
        imitation_prior = self.imitation.token_prior(self._cache_tokens, context_field.recent_tokens(self.spec.context_window), self.step)
        summary_prior = self.summary_prior()
        scores = self.predictor.score_candidates(
            query,
            self._cache_matrix,
            gravity_prior=gravity_prior + transition_prior,
            summary_prior=summary_prior,
            orbit_prior=orbit_prior,
            imitation_prior=imitation_prior,
            repeat_penalty=self.stats.repeat_ema,
        )
        if update_stats:
            scores, self.stats.score_mean, self.stats.score_m2, self.stats.score_count = self.predictor.normalize_scores(
                scores, self.stats.score_mean, self.stats.score_m2, self.stats.score_count, self.stats.repeat_ema,
            )
        else:
            scores, _, _, _ = self.predictor.normalize_scores(
                scores, self.stats.score_mean, self.stats.score_m2, self.stats.score_count, self.stats.repeat_ema,
            )
        tokens, scores = self._filter_output_tokens(self._cache_tokens, scores)
        if not tokens:
            return None
        return tokens, scores

    def scores_from_context(self, fallback_token: Optional[str] = None) -> Optional[Tuple[List[str], torch.Tensor]]:
        query = self._context_query_vector(fallback_token)
        if query is None:
            return None
        return self._compute_scores(query, self.context, self.context.state.path, self.step, update_stats=True)

    def _scores_from_state(self, state: ContextState, current_step: int, fallback_token: Optional[str] = None) -> Optional[Tuple[List[str], torch.Tensor]]:
        query = self._query_vector_for_state(state, current_step, fallback_token)
        if query is None:
            return None
        temp_context = ContextField(spec=self.spec, state=state)
        return self._compute_scores(query, temp_context, state.path, current_step, update_stats=False)

    def _advance_state_token(self, state: ContextState, current_step: int, token: str) -> int:
        temp_context = ContextField(spec=self.spec, state=state)
        next_step = current_step + 1
        temp_context.advance_token(token, next_step)
        return next_step

    def predict_next_from_vector(self, vec: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        scored = self.scores_from_vector(vec)
        if scored is None:
            return []
        tokens, scores = scored
        top_k = min(top_k, scores.numel())
        vals, idx = torch.topk(scores, k=top_k)
        return [(tokens[int(i.item())], float(v.item())) for i, v in zip(idx, vals)]

    def predict_next_context(self, fallback_token: Optional[str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        scored = self.scores_from_context(fallback_token=fallback_token)
        if scored is None:
            return []
        tokens, scores = scored
        top_k = min(top_k, scores.numel())
        vals, idx = torch.topk(scores, k=top_k)
        return [(tokens[int(i.item())], float(v.item())) for i, v in zip(idx, vals)]

    def predict_next(self, input_token: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if input_token not in self.points.anchors:
            return []
        pairs = self.predict_next_from_vector(self.points.anchors[input_token].vec, top_k=top_k + 1)
        return [(tok, score) for tok, score in pairs if tok != input_token][:top_k]

    def most_similar(self, vec: torch.Tensor, top_k: int = 2) -> List[Tuple[str, float]]:
        return self.predict_next_from_vector(vec, top_k=top_k)

    # activation / learning 메서드는 LearningMixin에서 제공
    # (_compute_activation, _compute_activation_batch, step_update,
    #  step_update_batch, train_imitation_pair, _apply_learning_step*, ...)

    def get_metrics(self) -> Dict[str, float]:
        gravity_edges = sum(len(mapping) for mapping in self.gravity.forward_gravity.values())
        orbit_count = sum(1 for _, count in self.orbits.path_counts.items() if count >= self.spec.orbit_min_count)
        avg_importance = 0.0
        if self.points.anchors:
            avg_importance = sum(anchor.importance for anchor in self.points.anchors.values()) / len(self.points.anchors)
        return {
            "step": float(self.step),
            "anchors": float(len(self.points.anchors)),
            "prediction_loss": float(self.stats.last_prediction_loss),
            "prediction_loss_ema": float(self.stats.prediction_loss_ema),
            "imitation_loss_ema": float(self.stats.imitation_loss_ema),
            "structure_loss": float(self.stats.last_structure_loss),
            "structure_loss_ema": float(self.stats.structure_loss_ema),
            "alignment_score": float(self.stats.last_alignment_score),
            "ema_error": float(self.stats.ema_error),
            "last_error": float(self.stats.last_error),
            "last_context_error": float(self.stats.last_context_error),
            "repeat_ema": float(self.stats.repeat_ema),
            "gravity_edges": float(gravity_edges),
            "orbits": float(orbit_count),
            "avg_importance": float(avg_importance),
            "imitation_ratio": float(self.imitation.imitation_ratio(self.step)),
        }

    # shock wave 메서드는 ShockWaveMixin에서 제공
    # (_maybe_emit_shock_wave, _propagate_shock_waves, _compute_repulsion)

    # learning 메서드는 LearningMixin에서 제공
    # (train_imitation_pair, _update_context_error, _next_loop_token, _maybe_create_loop,
    #  _apply_learning_step_precomputed, _apply_learning_step,
    #  step_update, step_update_batch)

    def update_imitation_noise(self, diff_ratio: float) -> None:
        if diff_ratio > self.spec.imitation_target_ratio:
            self.imitation_noise *= (1.0 - self.spec.imitation_adjust_rate)
        else:
            self.imitation_noise *= (1.0 + self.spec.imitation_adjust_rate)
        self.imitation_noise = max(self.spec.imitation_noise_min, min(self.spec.imitation_noise_max, self.imitation_noise))

    def _build_state(self) -> Dict[str, Any]:
        return {
            "format_version": 3,
            "spec": asdict(self.spec),
            "step": self.step,
            "anchors": {tok: anchor.vec.detach().cpu() for tok, anchor in self.points.anchors.items()},
            "importance": {tok: anchor.importance for tok, anchor in self.points.anchors.items()},
            "last_active": {tok: anchor.last_active_step for tok, anchor in self.points.anchors.items()},
            "char_anchors": {tok: anchor.vec.detach().cpu() for tok, anchor in self.points.char_anchors.items()},
            "gravity_base": self.gravity.base_gravity,
            "gravity_forward": self.gravity.forward_gravity,
            "orbit_counts": dict(self.orbits.path_counts),
            "orbit_strength": dict(self.orbits.path_strength),
            "orbit_last_seen": dict(self.orbits.path_last_seen_step),
            "context_path": list(self.context.state.path),
            "context_signatures": list(self.context.state.signatures),
            "raw_line_memory": [
                {
                    "vec": item["vec"].detach().cpu(),
                    "span": int(item.get("span", 1)),
                    "depth": int(item.get("depth", 0)),
                    "last_step": int(item.get("last_step", 0)),
                }
                for item in self.raw_line_memory
            ],
            "compressed_line_memory": [
                {
                    "vec": item["vec"].detach().cpu(),
                    "span": int(item.get("span", 1)),
                    "depth": int(item.get("depth", 0)),
                    "last_step": int(item.get("last_step", 0)),
                }
                for item in self.compressed_line_memory
            ],
            "shock_waves": [
                {
                    "center_token": str(item.get("center_token", "")),
                    "center_vec": item["center_vec"].detach().cpu(),
                    "amplitude": float(item.get("amplitude", 0.0)),
                    "radius": float(item.get("radius", 0.0)),
                    "speed": float(item.get("speed", 0.0)),
                    "decay": float(item.get("decay", self.spec.shock_decay)),
                    "preserve_tokens": list(item.get("preserve_tokens", [])),
                }
                for item in self.shock_waves
            ],
            "qa_memory": {key: value.detach().cpu() for key, value in self.qa.memory.items()},
            "qa_counts": dict(self.qa.counts),
            "stats": {
                "error_history": list(self.stats.error_history),
                "error_count": self.stats.error_count,
                "error_mean": self.stats.error_mean,
                "error_m2": self.stats.error_m2,
                "ema_error": self.stats.ema_error,
                "last_error": self.stats.last_error,
                "last_near_dist": self.stats.last_near_dist,
                "last_context_error": self.stats.last_context_error,
                "repeat_ema": self.stats.repeat_ema,
                "score_count": self.stats.score_count,
                "score_mean": self.stats.score_mean,
                "score_m2": self.stats.score_m2,
                "last_prediction_loss": self.stats.last_prediction_loss,
                "prediction_loss_ema": self.stats.prediction_loss_ema,
                "imitation_loss_ema": self.stats.imitation_loss_ema,
                "last_structure_loss": self.stats.last_structure_loss,
                "structure_loss_ema": self.stats.structure_loss_ema,
                "last_alignment_score": self.stats.last_alignment_score,
            },
            "similarity_threshold": self.similarity_threshold,
            "stabilization_steps": self.stabilization_steps,
            "dim_importance": self.dim_importance.detach().cpu(),
            "imitation_noise": self.imitation_noise,
        }

    def save(self, path: str) -> None:
        torch.save(self._build_state(), path)

    @staticmethod
    def load(path: str) -> "CWMCore":
        data = torch.load(path, map_location="cpu", weights_only=False)
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(CWMSpec)}
        spec = CWMSpec(**{k: v for k, v in data["spec"].items() if k in valid_keys})
        core = CWMCore(vocab=[], spec=spec)
        core.step = int(data.get("step", 0))

        core.points.anchors = {}
        for tok, vec in data.get("anchors", {}).items():
            anchor = Anchor(token=tok, vec=l2_normalize(vec.to(core.device)))
            anchor.importance = float(data.get("importance", {}).get(tok, 0.0))
            anchor.last_active_step = int(data.get("last_active", {}).get(tok, 0))
            core.points.anchors[tok] = anchor

        core.points.char_anchors = {}
        for ch, vec in data.get("char_anchors", {}).items():
            core.points.char_anchors[ch] = Anchor(token=ch, vec=l2_normalize(vec.to(core.device)))

        core.gravity.base_gravity = {str(src): {str(dst): float(val) for dst, val in mapping.items()} for src, mapping in data.get("gravity_base", {}).items()}
        core.gravity.forward_gravity = {str(src): {str(dst): float(val) for dst, val in mapping.items()} for src, mapping in data.get("gravity_forward", {}).items()}
        core.orbits.path_counts = {tuple(path): int(count) for path, count in data.get("orbit_counts", {}).items()}
        core.orbits.path_strength = {tuple(path): float(score) for path, score in data.get("orbit_strength", {}).items()}
        core.orbits.path_last_seen_step = {tuple(path): int(step) for path, step in data.get("orbit_last_seen", {}).items()}
        # _prefix_index 재구축 (저장 파일에 없으므로 path_counts에서 복원)
        core.orbits._prefix_index = {}
        for full_path in list(core.orbits.path_counts.keys()):
            if len(full_path) < 2:
                continue
            prefix: Tuple[str, ...] = tuple(full_path[:-1])
            next_tok: str = str(full_path[-1])
            if prefix not in core.orbits._prefix_index:
                core.orbits._prefix_index[prefix] = {}
            core.orbits._prefix_index[prefix][next_tok] = full_path
        core.context.state.path = [tuple(item) for item in data.get("context_path", [])]
        core.context.state.signatures = [(tuple(sig), int(step)) for sig, step in data.get("context_signatures", [])]
        core.raw_line_memory = [
            {
                "vec": l2_normalize(item["vec"].to(core.device)),
                "span": int(item.get("span", 1)),
                "depth": int(item.get("depth", 0)),
                "last_step": int(item.get("last_step", 0)),
            }
            for item in data.get("raw_line_memory", [])
        ]
        core.compressed_line_memory = [
            {
                "vec": l2_normalize(item["vec"].to(core.device)),
                "span": int(item.get("span", 1)),
                "depth": int(item.get("depth", 0)),
                "last_step": int(item.get("last_step", 0)),
            }
            for item in data.get("compressed_line_memory", [])
        ]
        core.shock_waves = [
            {
                "center_token": str(item.get("center_token", "")),
                "center_vec": l2_normalize(item["center_vec"].to(core.device)),
                "amplitude": float(item.get("amplitude", 0.0)),
                "radius": float(item.get("radius", 0.0)),
                "speed": float(item.get("speed", 0.0)),
                "decay": float(item.get("decay", core.spec.shock_decay)),
                "preserve_tokens": list(item.get("preserve_tokens", [])),
            }
            for item in data.get("shock_waves", [])
            if item.get("center_vec") is not None
        ]
        core.qa.memory = {tuple(key): l2_normalize(value.to(core.device)) for key, value in data.get("qa_memory", {}).items()}
        core.qa.counts = {tuple(key): int(value) for key, value in data.get("qa_counts", {}).items()}

        stats = data.get("stats", {})
        core.stats.error_history = list(stats.get("error_history", []))
        core.stats.error_count = int(stats.get("error_count", 0))
        core.stats.error_mean = float(stats.get("error_mean", 0.0))
        core.stats.error_m2 = float(stats.get("error_m2", 0.0))
        core.stats.ema_error = float(stats.get("ema_error", 0.0))
        core.stats.last_error = float(stats.get("last_error", 0.0))
        core.stats.last_near_dist = float(stats.get("last_near_dist", 1.0))
        core.stats.last_context_error = float(stats.get("last_context_error", 0.0))
        core.stats.repeat_ema = float(stats.get("repeat_ema", 0.0))
        core.stats.score_count = int(stats.get("score_count", 0))
        core.stats.score_mean = float(stats.get("score_mean", 0.0))
        core.stats.score_m2 = float(stats.get("score_m2", 0.0))
        core.stats.last_prediction_loss = float(stats.get("last_prediction_loss", 0.0))
        core.stats.prediction_loss_ema = float(stats.get("prediction_loss_ema", 0.0))
        core.stats.imitation_loss_ema = float(stats.get("imitation_loss_ema", 0.0))
        core.stats.last_structure_loss = float(stats.get("last_structure_loss", 0.0))
        core.stats.structure_loss_ema = float(stats.get("structure_loss_ema", 0.0))
        core.stats.last_alignment_score = float(stats.get("last_alignment_score", 0.0))

        core.similarity_threshold = float(data.get("similarity_threshold", core.spec.similarity_merge))
        core.stabilization_steps = int(data.get("stabilization_steps", core.stabilization_steps))
        dim_importance = data.get("dim_importance")
        if dim_importance is not None:
            core.dim_importance = dim_importance.to(core.device)
        core.imitation_noise = float(data.get("imitation_noise", core.spec.imitation_noise_init))
        core._mark_cache_dirty()
        return core
