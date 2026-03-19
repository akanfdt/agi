from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import re

import torch

from cwm_context import ContextField
from cwm_gravity import GravityField
from cwm_imitation import ImitationTrainer
from cwm_orbit import OrbitMemory
from cwm_points import PointStore
from cwm_predictor import Predictor
from cwm_types import Anchor, ContextState, CWMSpec, QAMemoryState, StatsState, l2_normalize


class CWMCore:
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
        self.emotion_pos = self.points._random_unit_vector()
        self.emotion_neg = self.points._random_unit_vector()
        self.emotion_bias = 0.0

        self._cache_tokens: List[str] = []
        self._cache_index: Dict[str, int] = {}
        self._cache_matrix: Optional[torch.Tensor] = None
        self._cache_importance: Optional[torch.Tensor] = None
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
            self._cache_importance = None
            self._cache_dirty = False
            return
        self._cache_matrix = torch.stack([self.points.anchors[tok].vec for tok in self._cache_tokens], dim=0).to(self.device)
        self._cache_importance = torch.tensor(
            [float(self.points.anchors[tok].importance) for tok in self._cache_tokens],
            device=self.device,
            dtype=self._cache_matrix.dtype,
        )
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
            self.step += 1
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

    def reset_line_memory(self) -> None:
        self.raw_line_memory = []
        self.compressed_line_memory = []

    def sentence_summary_vector(self, tokens: List[str]) -> Optional[torch.Tensor]:
        if not tokens:
            return None
        weighted: List[torch.Tensor] = []
        weights: List[float] = []
        total = len(tokens)
        for idx, tok in enumerate(tokens):
            self._ensure_anchor_exists(tok)
            anchor = self.points.anchors.get(tok)
            if anchor is None:
                continue
            # Sentence understanding should lean slightly toward later tokens
            # while still preserving the whole utterance.
            recency = self.spec.sentence_recency_base + self.spec.sentence_recency_range * ((idx + 1) / max(1, total))
            importance = 1.0 + min(1.0, float(anchor.importance) * 0.05)
            weight = recency * importance
            weighted.append(anchor.vec.to(self.device) * weight)
            weights.append(weight)
        if not weighted:
            return None
        summary = torch.stack(weighted, dim=0).sum(dim=0) / max(sum(weights), 1e-8)
        return l2_normalize(summary)

    def _line_vector(self, tokens: List[str]) -> Optional[torch.Tensor]:
        return self.sentence_summary_vector(tokens)

    def _all_summary_segments(self) -> List[Dict[str, Any]]:
        return list(self.compressed_line_memory) + list(self.raw_line_memory)

    def _all_line_segments(self) -> List[Dict[str, Any]]:
        return self._all_summary_segments()

    def _merge_summary_segments(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        left_vec = left["vec"].to(self.device)
        right_vec = right["vec"].to(self.device)
        left_span = int(left.get("span", 1))
        right_span = int(right.get("span", 1))
        total_span = max(1, left_span + right_span)
        merged = l2_normalize((left_vec * left_span + right_vec * right_span) / total_span)
        return {
            "vec": merged,
            "span": total_span,
            "depth": max(int(left.get("depth", 0)), int(right.get("depth", 0))) + 1,
            "last_step": max(int(left.get("last_step", 0)), int(right.get("last_step", 0))),
        }

    def _merge_line_segments(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        return self._merge_summary_segments(left, right)

    def _compress_summary_memory_if_needed(self) -> None:
        while len(self.raw_line_memory) > self.spec.line_recent_capacity:
            merged = self._merge_summary_segments(self.raw_line_memory[0], self.raw_line_memory[1])
            self.raw_line_memory = self.raw_line_memory[2:]
            self.compressed_line_memory.append(merged)
        while len(self.compressed_line_memory) > self.spec.line_compressed_capacity:
            merged = self._merge_summary_segments(self.compressed_line_memory[0], self.compressed_line_memory[1])
            self.compressed_line_memory = [merged] + self.compressed_line_memory[2:]

    def _compress_line_memory_if_needed(self) -> None:
        self._compress_summary_memory_if_needed()

    def summary_memory_vector(self) -> Optional[torch.Tensor]:
        segments = self._all_summary_segments()
        if not segments:
            return None
        weighted: List[torch.Tensor] = []
        weights: List[float] = []
        total = len(segments)
        for idx, segment in enumerate(segments):
            recency_rank = total - idx - 1
            recency_weight = self.spec.line_summary_decay ** recency_rank
            span_weight = max(1.0, float(segment.get("span", 1)) ** 0.5)
            weight = recency_weight * span_weight
            weighted.append(segment["vec"].to(self.device) * weight)
            weights.append(weight)
        if not weighted:
            return None
        summary = torch.stack(weighted, dim=0).sum(dim=0) / max(sum(weights), 1e-8)
        return l2_normalize(summary)

    def _line_summary_vector(self) -> Optional[torch.Tensor]:
        return self.summary_memory_vector()

    def summary_prior(self) -> torch.Tensor:
        self._ensure_cache()
        if not self._cache_tokens:
            return torch.empty(0, dtype=torch.float32)
        summary_vec = self.summary_memory_vector()
        if summary_vec is None or self._cache_matrix is None:
            return torch.zeros(len(self._cache_tokens), dtype=torch.float32)
        sims = torch.mv(self._cache_matrix, summary_vec.to(self.device)).detach().cpu()
        return sims * float(self.spec.line_summary_score_weight)

    def _line_summary_prior(self) -> torch.Tensor:
        return self.summary_prior()

    def observe_sentence(self, tokens: List[str]) -> None:
        line_vec = self.sentence_summary_vector(tokens)
        if line_vec is None:
            return
        summary_vec = self.summary_memory_vector()
        if summary_vec is not None:
            alpha = self.alpha() * self.spec.line_summary_strength
            for tok in tokens:
                self._ensure_anchor_exists(tok)
                self.points.update_vector(tok, summary_vec, alpha, self.step)
                self.points.anchors[tok].importance += self.spec.line_summary_strength * self.spec.importance_observe
            self._mark_cache_dirty()
        self.raw_line_memory.append(
            {
                "vec": line_vec.detach().to(self.device),
                "span": 1,
                "depth": 0,
                "last_step": self.step,
            }
        )
        self._compress_summary_memory_if_needed()

    def observe_line(self, tokens: List[str]) -> None:
        self.observe_sentence(tokens)

    def reinforce_dialogue(self, question_tokens: List[str], answer_tokens: List[str]) -> None:
        q_vec = self.token_vector(question_tokens)
        if q_vec is None:
            return
        alpha = self.alpha() * self.spec.dialogue_strength
        for tok in answer_tokens:
            self._ensure_anchor_exists(tok)
            self.points.update_vector(tok, q_vec, alpha, self.step)
            self.points.anchors[tok].importance += self.spec.dialogue_strength
        for src, dst in zip(answer_tokens[:-1], answer_tokens[1:]):
            self.gravity.reinforce_pair(src, dst, self.spec.gravity_reinforce_amount * self.spec.dialogue_strength)
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
        """
        if tok in ("▁", "<unk>", "", " "):
            return False
        if "__loop_" in tok:
            return False
        if tok.startswith("<") and tok.endswith(">"):
            return False
        return True

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

    def _adaptive_prior(self, prior: torch.Tensor, stat_name: str, reference_scale: float) -> torch.Tensor:
        if prior.numel() == 0:
            return prior
        if not self.spec.use_adaptive_priors:
            return prior
        current = float(prior.abs().mean().item())
        beta = self.spec.adaptive_prior_beta
        prev = float(getattr(self.stats, stat_name))
        ema = current if prev == 0.0 else beta * prev + (1.0 - beta) * current
        setattr(self.stats, stat_name, ema)
        if ema <= 1e-8 or reference_scale <= 1e-8:
            return prior
        return prior * (reference_scale / ema)

    def scores_from_vector(self, vec: torch.Tensor) -> Optional[Tuple[List[str], torch.Tensor]]:
        self._ensure_cache()
        if self._cache_matrix is None or self._cache_importance is None:
            return None
        scores = self.predictor.score_candidates(
            l2_normalize(vec.to(self.device)),
            self._cache_matrix,
            self._cache_importance,
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

    def scores_from_context(self, fallback_token: Optional[str] = None) -> Optional[Tuple[List[str], torch.Tensor]]:
        query = self._context_query_vector(fallback_token)
        if query is None:
            return None
        self._ensure_cache()
        if self._cache_matrix is None or self._cache_importance is None:
            return None
        semantic = torch.mv(self._cache_matrix, l2_normalize(query.to(self.device))).detach().cpu()
        semantic_scale = float(semantic.abs().mean().item())
        beta = self.spec.adaptive_prior_beta
        prev_sem = self.stats.semantic_scale_ema
        self.stats.semantic_scale_ema = semantic_scale if prev_sem == 0.0 else beta * prev_sem + (1.0 - beta) * semantic_scale
        ref_scale = max(1e-6, self.stats.semantic_scale_ema)
        active_tokens = self.context.active_tokens(self.step)
        gravity_prior = self.gravity.context_prior(active_tokens, self._cache_tokens)
        transition_prior = self._transition_prior_from_path(self.context.state.path)
        orbit_map = self.orbits.query(self.context.recent_tokens(self.spec.orbit_max_length))
        orbit_prior = torch.tensor([orbit_map.get(tok, 0.0) for tok in self._cache_tokens], dtype=torch.float32)
        imitation_prior = self.imitation.token_prior(self._cache_tokens, self.context.recent_tokens(self.spec.context_window), self.step)
        gravity_prior = self._adaptive_prior(gravity_prior, "gravity_prior_ema", ref_scale)
        transition_prior = self._adaptive_prior(transition_prior, "transition_prior_ema", ref_scale)
        summary_prior = self.summary_prior()
        summary_prior = self._adaptive_prior(summary_prior, "summary_prior_ema", ref_scale)
        orbit_prior = self._adaptive_prior(orbit_prior, "orbit_prior_ema", ref_scale)
        imitation_prior = self._adaptive_prior(imitation_prior, "imitation_prior_ema", ref_scale)
        scores = self.predictor.score_candidates(
            query,
            self._cache_matrix,
            self._cache_importance,
            gravity_prior=gravity_prior + transition_prior,
            summary_prior=summary_prior,
            orbit_prior=orbit_prior,
            imitation_prior=imitation_prior,
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

    def _scores_from_state(self, state: ContextState, current_step: int, fallback_token: Optional[str] = None) -> Optional[Tuple[List[str], torch.Tensor]]:
        query = self._query_vector_for_state(state, current_step, fallback_token)
        if query is None:
            return None
        self._ensure_cache()
        if self._cache_matrix is None or self._cache_importance is None:
            return None
        semantic = torch.mv(self._cache_matrix, l2_normalize(query.to(self.device))).detach().cpu()
        semantic_scale = float(semantic.abs().mean().item())
        beta = self.spec.adaptive_prior_beta
        prev_sem = self.stats.semantic_scale_ema
        self.stats.semantic_scale_ema = semantic_scale if prev_sem == 0.0 else beta * prev_sem + (1.0 - beta) * semantic_scale
        ref_scale = max(1e-6, self.stats.semantic_scale_ema)
        temp_context = ContextField(spec=self.spec, state=state)
        active_tokens = temp_context.active_tokens(current_step)
        gravity_prior = self.gravity.context_prior(active_tokens, self._cache_tokens)
        transition_prior = self._transition_prior_from_path(state.path)
        orbit_map = self.orbits.query(temp_context.recent_tokens(self.spec.orbit_max_length))
        orbit_prior = torch.tensor([orbit_map.get(tok, 0.0) for tok in self._cache_tokens], dtype=torch.float32)
        imitation_prior = self.imitation.token_prior(self._cache_tokens, temp_context.recent_tokens(self.spec.context_window), self.step)
        gravity_prior = self._adaptive_prior(gravity_prior, "gravity_prior_ema", ref_scale)
        transition_prior = self._adaptive_prior(transition_prior, "transition_prior_ema", ref_scale)
        summary_prior = self.summary_prior()
        summary_prior = self._adaptive_prior(summary_prior, "summary_prior_ema", ref_scale)
        orbit_prior = self._adaptive_prior(orbit_prior, "orbit_prior_ema", ref_scale)
        imitation_prior = self._adaptive_prior(imitation_prior, "imitation_prior_ema", ref_scale)
        scores = self.predictor.score_candidates(
            query,
            self._cache_matrix,
            self._cache_importance,
            gravity_prior=gravity_prior + transition_prior,
            summary_prior=summary_prior,
            orbit_prior=orbit_prior,
            imitation_prior=imitation_prior,
            repeat_penalty=self.stats.repeat_ema,
        )
        scores, _, _, _ = self.predictor.normalize_scores(
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

    def _compute_activation(self, input_token: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_cache()
        if self._cache_matrix is None:
            empty_idx = torch.empty(0, dtype=torch.long, device=self.device)
            empty_scores = torch.empty(0, device=self.device)
            return empty_idx, empty_scores, empty_scores
        input_vec = self.points.anchors[input_token].vec
        sims = torch.mv(self._cache_matrix, input_vec)
        dists = 1.0 - sims
        sigma = self.sigma()
        strengths = torch.exp(-((dists * dists) / (2.0 * sigma * sigma)))
        k = min(self.spec.activation_top_k, strengths.numel())
        if k <= 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_idx, strengths, sims
        top_vals, top_idx = torch.topk(strengths, k=k)
        threshold = self.stop_threshold(sigma, strengths)
        relative_threshold = float(top_vals[0].item()) * float(self.spec.activation_keep_ratio)
        keep_mask = top_vals >= max(threshold, relative_threshold)
        if int(keep_mask.sum().item()) < self.spec.activation_min_keep:
            keep_mask[: min(self.spec.activation_min_keep, top_vals.numel())] = True
        active_idx = top_idx[keep_mask]
        return active_idx, strengths, sims

    def _update_nearest_distance(self, input_token: str, sims: torch.Tensor) -> None:
        if sims.numel() == 0 or input_token not in self._cache_index:
            self.stats.last_near_dist = 1.0
            return
        sims2 = sims.clone()
        sims2[self._cache_index[input_token]] = -1.0
        self.stats.last_near_dist = float(1.0 - sims2.max().item())

    def _update_importance(self, active_idx: torch.Tensor, strengths: torch.Tensor) -> None:
        self.points.decay_importance()
        if active_idx.numel() == 0:
            return
        ranked = sorted(
            active_idx.tolist(),
            key=lambda idx: float(strengths[idx].item()),
            reverse=True,
        )
        keep_n = max(1, min(self.spec.importance_top_k, len(ranked)))
        for idx in ranked[:keep_n]:
            token = self._cache_tokens[idx]
            strength = float(strengths[idx].item())
            anchor = self.points.anchors[token]
            anchor.importance += strength
            anchor.last_active_step = self.step
            self.dim_importance += torch.abs(anchor.vec) * strength
        self._mark_cache_dirty()

    def _update_error_stats(self, error: float) -> None:
        self.stats.error_history.append(error)
        if len(self.stats.error_history) > self.spec.error_window:
            self.stats.error_history = self.stats.error_history[-self.spec.error_window :]
        self.stats.last_error = error
        self.stats.error_count += 1
        delta = error - self.stats.error_mean
        self.stats.error_mean += delta / self.stats.error_count
        delta2 = error - self.stats.error_mean
        self.stats.error_m2 += delta * delta2
        if self.stats.error_count == 1:
            self.stats.ema_error = error
        else:
            beta = self.spec.ema_beta
            self.stats.ema_error = beta * self.stats.ema_error + (1.0 - beta) * error

    def _estimate_structure_loss(
        self,
        input_token: str,
        next_token: str,
        active_idx: torch.Tensor,
        sims: torch.Tensor,
    ) -> float:
        if sims.numel() == 0:
            return 0.0
        target_sim = 0.0
        if next_token in self._cache_index:
            target_sim = float(sims[self._cache_index[next_token]].item())
        active_mean = 0.0
        if active_idx.numel() > 0:
            active_mean = float(sims[active_idx].mean().item())
        gravity_bonus = self.gravity.get_forward_gravity(input_token, next_token)
        loss = max(0.0, 1.0 - target_sim) + max(0.0, active_mean - target_sim * self.spec.structure_loss_active_scale) - gravity_bonus * self.spec.structure_loss_gravity_scale
        return max(0.0, loss)

    def _update_prediction_metric(self, loss_value: float) -> None:
        self.stats.last_prediction_loss = loss_value
        beta = self.spec.ema_beta
        if self.stats.prediction_loss_ema == 0.0:
            self.stats.prediction_loss_ema = loss_value
        else:
            self.stats.prediction_loss_ema = beta * self.stats.prediction_loss_ema + (1.0 - beta) * loss_value

    def _update_structure_metric(self, loss_value: float) -> None:
        self.stats.last_structure_loss = loss_value
        beta = self.spec.ema_beta
        if self.stats.structure_loss_ema == 0.0:
            self.stats.structure_loss_ema = loss_value
        else:
            self.stats.structure_loss_ema = beta * self.stats.structure_loss_ema + (1.0 - beta) * loss_value

    def _update_imitation_metric(self, loss_value: float, alignment: float) -> None:
        beta = self.spec.ema_beta
        if self.stats.imitation_loss_ema == 0.0:
            self.stats.imitation_loss_ema = loss_value
        else:
            self.stats.imitation_loss_ema = beta * self.stats.imitation_loss_ema + (1.0 - beta) * loss_value
        self.stats.last_alignment_score = alignment

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

    def train_imitation_pair(self, input_tokens: List[str], target_tokens: List[str]) -> Dict[str, float]:
        if not input_tokens or not target_tokens:
            return {"imitation_loss": 0.0, "alignment": 0.0, "steps": 0.0}

        for tok in input_tokens:
            self._ensure_anchor_exists(tok)
        for tok in target_tokens:
            self._ensure_anchor_exists(tok)

        sim_state = ContextState(
            path=list(self.context.state.path),
            signatures=list(self.context.state.signatures),
        )
        sim_step = self.step
        for tok in input_tokens:
            sim_step = self._advance_state_token(sim_state, sim_step, tok)
        logits_list: List[torch.Tensor] = []
        target_indices: List[int] = []
        generated_tokens: List[str] = []
        prev_token = input_tokens[-1]

        for tok in target_tokens:
            scored = self._scores_from_state(sim_state, sim_step, fallback_token=prev_token)
            if scored is None:
                break
            cand_tokens, logits = scored
            if tok in cand_tokens:
                logits_list.append(logits)
                target_indices.append(cand_tokens.index(tok))
                pred_idx = int(torch.argmax(logits).item())
                generated_tokens.append(cand_tokens[pred_idx])
            sim_step = self._advance_state_token(sim_state, sim_step, tok)
            prev_token = tok

        imitation_loss = 0.0
        alignment = 0.0
        if logits_list and target_indices:
            loss = self.imitation.sequence_loss(logits_list, target_indices)
            imitation_loss = float(loss.item())
            alignment = self.imitation.alignment_score(generated_tokens, target_tokens[: len(generated_tokens)])
            self._update_imitation_metric(imitation_loss, alignment)

            ratio = self.imitation.imitation_ratio(self.step)
            q_vec = self.token_vector(input_tokens)
            if q_vec is not None:
                for tok in target_tokens:
                    self.points.update_vector(tok, q_vec, self.alpha() * ratio * self.spec.imitation_update_scale, self.step)
                    self.points.anchors[tok].importance += ratio * self.spec.imitation_importance_scale
            self.gravity.reinforce_sequence(input_tokens + target_tokens, self.spec.gravity_reinforce_amount * ratio)
            self.orbits.observe(input_tokens + target_tokens, self.step, max(0.0, alignment))
            self._mark_cache_dirty()

        return {
            "imitation_loss": float(imitation_loss),
            "alignment": float(alignment),
            "steps": float(len(logits_list)),
        }

    def _update_context_error(self, target_vec: torch.Tensor) -> None:
        ctx = self.context_vector()
        if ctx is None:
            self.stats.last_context_error = 0.0
            return
        self.stats.last_context_error = self.dist_cos(target_vec, ctx)

    def _next_loop_token(self, base_token: str) -> str:
        suffix = 1
        while True:
            token = f"{base_token}__loop_{suffix}"
            if token not in self.points.anchors:
                return token
            suffix += 1

    def _maybe_create_loop(self, next_token: str, target_vec: torch.Tensor, sims: torch.Tensor) -> None:
        if self.step < self.stabilization_steps or sims.numel() == 0:
            return
        novelty = 1.0 - float(sims.max().item())
        if novelty <= self.error_threshold() * self.spec.create_loop_threshold_scale:
            return
        self.add_anchor_near(self._next_loop_token(next_token), target_vec)

    def _maybe_emit_shock_wave(self, center_token: str, preserve_tokens: List[str]) -> None:
        # 안정화 전에는 발동하지 않음 — gravity가 형성되기 전에 decay하면 역효과
        if self.step < self.stabilization_steps:
            return

        crowding = self.gravity.forward_crowding(center_token)
        if crowding <= 0.0:
            return

        pred_ema = max(1e-6, float(self.stats.prediction_loss_ema))
        pred_signal = 0.0
        if self.stats.last_prediction_loss > 0.0:
            pred_signal = max(0.0, (float(self.stats.last_prediction_loss) - pred_ema) / pred_ema)
        error_signal = max(0.0, float(self.stats.last_error) - float(self.stats.ema_error))
        repeat_signal = max(0.0, float(self.stats.repeat_ema))

        instability = crowding * (pred_signal + error_signal) * (self.spec.shock_instability_base + repeat_signal)
        if instability <= 0.0:
            return

        if center_token not in self.points.anchors:
            return

        center_vec = self.points.anchors[center_token].vec.detach().clone().to(self.device)
        self.shock_waves.append(
            {
                "center_token": center_token,
                "center_vec": center_vec,
                "amplitude": min(self.spec.shock_amplitude_max, instability * self.spec.shock_amplitude_scale),
                "radius": 0.0,
                "speed": self.spec.shock_speed_base + self.spec.shock_speed_crowding * min(1.0, crowding),
                "decay": self.spec.shock_decay,
                "preserve_tokens": list(dict.fromkeys(preserve_tokens[-self.spec.orbit_max_length :])),
            }
        )

    def _propagate_shock_waves(self) -> None:
        if not self.shock_waves:
            return
        self._ensure_cache()
        if self._cache_matrix is None or not self._cache_tokens:
            self.shock_waves = []
            return

        remaining: List[Dict[str, Any]] = []
        distance_scale = max(0.08, 1.0 - self.spec.similarity_merge)
        for wave in self.shock_waves:
            amplitude = float(wave.get("amplitude", 0.0))
            if amplitude <= 1e-3:
                continue

            radius = float(wave.get("radius", 0.0)) + float(wave.get("speed", 0.0))
            center_vec = l2_normalize(wave["center_vec"].to(self.device))
            sims = torch.mv(self._cache_matrix, center_vec)
            dists = 1.0 - sims
            sigma = max(distance_scale, self.spec.shock_ring_cutoff + radius * 0.5)
            ring = torch.exp(-((dists - radius) ** 2) / (2.0 * sigma * sigma))
            preserve = list(wave.get("preserve_tokens", []))

            top_k = min(64, ring.numel())
            if top_k > 0:
                top_vals, top_idx = torch.topk(ring, k=top_k)
                for idx, ring_strength in zip(top_idx.tolist(), top_vals.tolist()):
                    if ring_strength < self.spec.shock_ring_cutoff:
                        continue
                    token = self._cache_tokens[idx]
                    if token in preserve:
                        continue
                    local_strength = amplitude * float(ring_strength) * self.spec.shock_gravity_scale
                    self.gravity.supernova_decay(token, preserve_tokens=preserve, strength=local_strength)

                    # 앵커 위치를 파동 중심 반대 방향으로 실제로 밀어냄
                    anchor = self.points.anchors.get(token)
                    if anchor is not None and self.spec.shock_repel_strength > 0.0:
                        repel_dir = anchor.vec - center_vec
                        repel_norm = float(torch.norm(repel_dir).item())
                        if repel_norm > 1e-6:
                            repel_dir = repel_dir / repel_norm
                            push = amplitude * float(ring_strength) * self.spec.shock_repel_strength
                            anchor.vec = l2_normalize(anchor.vec + repel_dir * push)

            amplitude *= float(wave.get("decay", self.spec.shock_decay))
            if amplitude > 1e-3 and radius < self.spec.shock_max_radius:
                updated = dict(wave)
                updated["radius"] = radius
                updated["amplitude"] = amplitude
                remaining.append(updated)

        if any(w not in remaining for w in self.shock_waves):
            self._mark_cache_dirty()
        self.shock_waves = remaining

    def _compute_repulsion(self, token: str, exclude_tokens: set) -> Optional[torch.Tensor]:
        """
        주어진 앵커 주변에 너무 가까운 앵커들이 있으면,
        그것들로부터 밀려나는 방향 벡터를 반환한다.
        코사인 유사도가 spec.repulsion_threshold 이상인 앵커만 대상.
        """
        anchor = self.points.anchors.get(token)
        if anchor is None or self._cache_matrix is None:
            return None
        sims = torch.mv(self._cache_matrix, anchor.vec.to(self.device))
        repel_vec = torch.zeros_like(anchor.vec)
        count = 0
        for i, sim_val in enumerate(sims.tolist()):
            if sim_val < self.spec.repulsion_threshold:
                continue
            cand = self._cache_tokens[i]
            if cand == token or cand in exclude_tokens:
                continue
            # 가까울수록(sim이 높을수록) 강하게 밀어냄
            push = anchor.vec - self._cache_matrix[i].to(self.device)
            repel_vec = repel_vec + push * float(sim_val)
            count += 1
        if count == 0:
            return None
        norm = float(torch.norm(repel_vec).item())
        if norm < 1e-6:
            return None
        return repel_vec / norm

    def _apply_learning_step(
        self,
        input_token: str,
        next_token: str,
        update_context: bool,
        recent_override: Optional[List[str]] = None,
    ) -> None:
        self._ensure_anchor_exists(input_token)
        self._ensure_anchor_exists(next_token)
        self._ensure_cache()
        if self._cache_matrix is None:
            return

        active_idx, strengths, sims = self._compute_activation(input_token)
        self._update_nearest_distance(input_token, sims)
        self._update_importance(active_idx, strengths)

        input_anchor = self.points.anchors[input_token]
        target_anchor = self.points.anchors[next_token]
        self._update_context_error(target_anchor.vec)

        prediction = self.scores_from_context(fallback_token=input_token)
        if prediction is not None:
            tokens, logits = prediction
            if next_token in tokens:
                loss = self.predictor.prediction_loss(logits, tokens.index(next_token))
                loss_value = float(loss.item())
                self._update_prediction_metric(loss_value)
                self.stats.last_alignment_score = self.imitation.alignment_score([tokens[int(torch.argmax(logits).item())]], [next_token])
        structure_loss = self._estimate_structure_loss(input_token, next_token, active_idx, sims)
        self._update_structure_metric(structure_loss)

        error = self.dist_euclid(target_anchor.vec, input_anchor.vec)
        self._update_error_stats(error)

        alpha = self.alpha()
        exclude = {input_token, next_token}
        for idx in active_idx.tolist():
            strength = float(strengths[idx].item())
            if strength < self.spec.min_activation_for_update:
                continue
            token = self._cache_tokens[idx]
            anchor = self.points.anchors[token]
            gravity_bonus = self.gravity.get_forward_gravity(input_token, token)
            attract_weight = alpha * strength * (1.0 + gravity_bonus)

            if self.spec.repulsion_strength > 0.0:
                repel_vec = self._compute_repulsion(token, exclude_tokens=exclude)
                if repel_vec is not None:
                    repel_weight = alpha * strength * self.spec.repulsion_strength
                    delta = attract_weight * (target_anchor.vec - anchor.vec) + repel_weight * repel_vec
                    anchor.vec = l2_normalize(anchor.vec + delta)
                    continue
            self.points.update_vector(token, target_anchor.vec, attract_weight, self.step)

        self.points.anchors[input_token].importance += self.spec.importance_attract
        self.points.anchors[next_token].importance += self.spec.importance_next
        self.gravity.reinforce_pair(input_token, next_token)
        if recent_override is None:
            recent = self.context.recent_tokens(self.spec.orbit_max_length - 1)
        else:
            recent = recent_override[-(self.spec.orbit_max_length - 1) :]
        if recent and recent[-1] == input_token:
            path = recent + [next_token]
        else:
            path = recent + [input_token, next_token]
        self.gravity.reinforce_sequence(path)
        self.orbits.observe(path, self.step, 1.0 - min(1.0, error))
        self._maybe_create_loop(next_token, target_anchor.vec, sims)
        self._mark_cache_dirty()

        if update_context:
            self.context.advance_token(input_token, self.step)
        self._repeat_pattern_similar()
        self._maybe_emit_shock_wave(input_token, preserve_tokens=path)
        self._propagate_shock_waves()

    def step_update(self, input_token: str, next_token: Optional[str] = None) -> None:
        self.step += 1
        self._ensure_anchor_exists(input_token)
        if next_token is None:
            self.context.advance_token(input_token, self.step)
            return
        self._apply_learning_step(input_token, next_token, update_context=True)

    def step_update_batch(self, pairs: List[Tuple[str, str]], fast: bool = True) -> None:
        recent_tokens: List[str] = []
        for input_token, next_token in pairs:
            self.step += 1
            if not recent_tokens or recent_tokens[-1] != input_token:
                recent_tokens = [input_token]
            local_recent = recent_tokens[-(self.spec.orbit_max_length - 1) :]
            self._apply_learning_step(
                input_token,
                next_token,
                update_context=not fast,
                recent_override=local_recent,
            )
            recent_tokens.append(next_token)

    def maybe_manage_dimensions(self) -> None:
        return

    def maybe_manage_anchors(self) -> None:
        return

    def set_emotion_bias(self, value: float) -> None:
        self.emotion_bias = max(-1.0, min(1.0, value))

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
                    "decay": float(item.get("decay", 0.72)),
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
                "semantic_scale_ema": self.stats.semantic_scale_ema,
                "gravity_prior_ema": self.stats.gravity_prior_ema,
                "transition_prior_ema": self.stats.transition_prior_ema,
                "orbit_prior_ema": self.stats.orbit_prior_ema,
                "imitation_prior_ema": self.stats.imitation_prior_ema,
                "summary_prior_ema": self.stats.summary_prior_ema,
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
            "emotion_pos": self.emotion_pos.detach().cpu(),
            "emotion_neg": self.emotion_neg.detach().cpu(),
            "emotion_bias": self.emotion_bias,
            "imitation_noise": self.imitation_noise,
        }

    def save(self, path: str) -> None:
        torch.save(self._build_state(), path)

    @staticmethod
    def load(path: str) -> "CWMCore":
        data = torch.load(path, map_location="cpu", weights_only=False)
        spec = CWMSpec(**data["spec"])
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
                "decay": float(item.get("decay", 0.72)),
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
        core.stats.semantic_scale_ema = float(stats.get("semantic_scale_ema", 0.0))
        core.stats.gravity_prior_ema = float(stats.get("gravity_prior_ema", 0.0))
        core.stats.transition_prior_ema = float(stats.get("transition_prior_ema", 0.0))
        core.stats.orbit_prior_ema = float(stats.get("orbit_prior_ema", 0.0))
        core.stats.imitation_prior_ema = float(stats.get("imitation_prior_ema", 0.0))
        core.stats.summary_prior_ema = float(stats.get("summary_prior_ema", 0.0))
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
        core.emotion_pos = l2_normalize(data.get("emotion_pos", core.emotion_pos).to(core.device))
        core.emotion_neg = l2_normalize(data.get("emotion_neg", core.emotion_neg).to(core.device))
        core.emotion_bias = float(data.get("emotion_bias", 0.0))
        core.imitation_noise = float(data.get("imitation_noise", core.spec.imitation_noise_init))
        core._mark_cache_dirty()
        return core