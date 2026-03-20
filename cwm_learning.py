"""
cwm_learning.py — 학습 엔진 (LearningMixin)

담당 역할:
  - Gaussian activation 계산: _compute_activation(), _compute_activation_batch()
  - 학습 스텝 실행: _apply_learning_step(), _apply_learning_step_precomputed()
  - 배치 학습 인터페이스: step_update(), step_update_batch()
  - Imitation 학습: train_imitation_pair()
  - 메트릭 갱신: _update_*_metric(), _update_error_stats()

CWMCore에 믹스인으로 사용됨.
self 속성 의존: spec, device, step, points, gravity, orbits, context,
               imitation, predictor, stats, stabilization_steps, dim_importance,
               _cache_tokens, _cache_index, _cache_matrix
self 메서드 의존: _ensure_cache, _ensure_anchor_exists, _mark_cache_dirty,
                _add_importance, alpha, sigma, error_threshold, stop_threshold,
                dist_euclid, dist_cos, context_vector, scores_from_context,
                _scores_from_state, _advance_state_token, add_anchor_near,
                token_vector,
                _maybe_emit_shock_wave, _propagate_shock_waves,  (ShockWaveMixin)
                _compute_repulsion,                              (ShockWaveMixin)
                _repeat_pattern_similar                          (CWMCore)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from cwm_types import ContextState, l2_normalize


class LearningMixin:

    # -------------------------------------------------------
    # Activation 계산
    # -------------------------------------------------------

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
        threshold = self.stop_threshold(sigma, strengths)
        keep_mask = strengths >= threshold
        if int(keep_mask.sum().item()) < self.spec.activation_min_keep:
            top_idx = torch.topk(strengths, k=min(self.spec.activation_min_keep, strengths.numel())).indices
            keep_mask[top_idx] = True
        active_idx = keep_mask.nonzero(as_tuple=True)[0]
        return active_idx, strengths, sims

    def _compute_activation_batch(
        self,
        input_tokens: List[str],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        여러 input_token의 activation을 mm() 한 번으로 계산.

        반환:
          active_idx_list : List[Tensor]  — 각 토큰의 활성 앵커 인덱스
          strengths_mat   : [N_anchors × B] — 전체 strength 행렬
          sims_mat        : [N_anchors × B] — 전체 유사도 행렬
        B = len(input_tokens)
        """
        self._ensure_cache()
        B = len(input_tokens)
        empty = (
            [torch.empty(0, dtype=torch.long, device=self.device)] * B,
            torch.empty(0, device=self.device),
            torch.empty(0, device=self.device),
        )
        if self._cache_matrix is None or B == 0:
            return empty

        vecs = []
        for tok in input_tokens:
            anchor = self.points.anchors.get(tok)
            if anchor is None:
                # zeros 대신 random unit vector — zeros이면 모든 앵커와 유사도 0으로
                # strength가 exp(-1/2σ²)로 균일하게 나와 top-k 선별이 무의미해짐.
                vecs.append(self.points._random_unit_vector().to(self.device))
            else:
                vecs.append(anchor.vec.to(self.device))
        input_stack = torch.stack(vecs, dim=1)  # [dim × B]

        # [N_anchors × B] — mm() 한 번으로 전체 유사도 행렬
        sims_mat = torch.mm(self._cache_matrix, input_stack)
        dists_mat = 1.0 - sims_mat
        sigma = self.sigma()
        strengths_mat = torch.exp(-(dists_mat ** 2) / (2.0 * sigma * sigma))

        active_idx_list: List[torch.Tensor] = []
        for b in range(B):
            col_vals = strengths_mat[:, b]
            col_threshold = self.stop_threshold(sigma, col_vals)
            keep_mask = col_vals >= col_threshold
            if int(keep_mask.sum().item()) < self.spec.activation_min_keep:
                top_idx = torch.topk(col_vals, k=min(self.spec.activation_min_keep, col_vals.numel())).indices
                keep_mask[top_idx] = True
            active_idx_list.append(keep_mask.nonzero(as_tuple=True)[0])

        return active_idx_list, strengths_mat, sims_mat

    # -------------------------------------------------------
    # 메트릭 갱신
    # -------------------------------------------------------

    def _update_nearest_distance(self, input_token: str, sims: torch.Tensor) -> None:
        if sims.numel() == 0 or input_token not in self._cache_index:
            self.stats.last_near_dist = 1.0
            return
        sims2 = sims.clone()
        sims2[self._cache_index[input_token]] = -1.0
        self.stats.last_near_dist = float(1.0 - sims2.max().item())

    def _update_importance(self, active_idx: torch.Tensor, strengths: torch.Tensor, fast: bool = False) -> None:
        # fast=True (배치 모드): decay/dim_importance/cache dirty는 배치 단위로 외부에서 처리.
        if not fast:
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
            if not fast:
                self.dim_importance += torch.abs(anchor.vec) * strength
        if not fast:
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
        loss = (
            max(0.0, 1.0 - target_sim)
            + max(0.0, active_mean - target_sim * self.spec.structure_loss_active_scale)
            - gravity_bonus * self.spec.structure_loss_gravity_scale
        )
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

    def _update_context_error(self, target_vec: torch.Tensor) -> None:
        ctx = self.context_vector()
        if ctx is None:
            self.stats.last_context_error = 0.0
            return
        self.stats.last_context_error = self.dist_cos(target_vec, ctx)

    # -------------------------------------------------------
    # 루프 앵커
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # Imitation 학습
    # -------------------------------------------------------

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
                    self._add_importance(tok, ratio * self.spec.imitation_importance_scale)
            self.gravity.reinforce_sequence(input_tokens + target_tokens, self.spec.gravity_reinforce_amount * ratio)
            self.orbits.observe(input_tokens + target_tokens, self.step, max(0.0, alignment))
            self._mark_cache_dirty()

        return {
            "imitation_loss": float(imitation_loss),
            "alignment": float(alignment),
            "steps": float(len(logits_list)),
        }

    # -------------------------------------------------------
    # 핵심 학습 스텝
    # -------------------------------------------------------

    def _apply_learning_step_precomputed(
        self,
        input_token: str,
        next_token: str,
        active_idx: torch.Tensor,
        strengths: torch.Tensor,
        sims: torch.Tensor,
        update_context: bool,
        recent_override: Optional[List[str]] = None,
        fast: bool = False,
    ) -> None:
        """
        _apply_learning_step과 동일하지만 activation(active_idx/strengths/sims)을
        외부에서 미리 계산해서 받음 — step_update_batch의 mm() 배치화에 사용.
        _compute_activation() 호출을 건너뛰므로 배치당 mv() 중복 제거.

        fast=True: scores_from_context 호출 생략 (prediction loss 측정 스킵).
        배치 학습에서 이 호출이 전체 시간의 ~99%를 차지하므로 fast 모드에서 제거.
        prediction_loss_ema는 fast=False인 step_update (단일 스텝)에서 계속 측정됨.
        """
        self._ensure_cache()
        if self._cache_matrix is None:
            return

        self._update_nearest_distance(input_token, sims)
        self._update_importance(active_idx, strengths, fast=fast)

        input_anchor = self.points.anchors[input_token]
        target_anchor = self.points.anchors[next_token]
        self._update_context_error(target_anchor.vec)

        # fast=True여도 pred_eval_interval 스텝마다 1번 pred 측정.
        # pred_ema가 고정되면 shock_wave pred_signal=0 → gravity 정리 안 됨.
        _do_pred_eval = (not fast) or (self.step % self.spec.pred_eval_interval == 0)
        if _do_pred_eval:
            prediction = self.scores_from_context(fallback_token=input_token)
            if prediction is not None:
                pred_tokens, logits = prediction
                if next_token in pred_tokens:
                    loss = self.predictor.prediction_loss(logits, pred_tokens.index(next_token))
                    self._update_prediction_metric(float(loss.item()))
                    self.stats.last_alignment_score = self.imitation.alignment_score(
                        [pred_tokens[int(torch.argmax(logits).item())]], [next_token]
                    )
                else:
                    # next_token이 emittable 필터에 걸려도 pred_ema는 살려둠
                    approx_loss = float(logits.max().item()) - float(logits.mean().item())
                    if approx_loss > 0.0:
                        self._update_prediction_metric(approx_loss)
        structure_loss = self._estimate_structure_loss(input_token, next_token, active_idx, sims)
        self._update_structure_metric(structure_loss)

        error = self.dist_euclid(target_anchor.vec, input_anchor.vec)
        self._update_error_stats(error)

        alpha = self.alpha() * self.spec.pair_learning_scale
        exclude = {input_token, next_token}
        for idx in active_idx.tolist():
            strength = float(strengths[idx].item())
            if strength < self.spec.min_activation_for_update:
                continue
            token = self._cache_tokens[idx]
            anchor = self.points.anchors[token]
            gravity_bonus = self.gravity.get_forward_gravity(input_token, token)
            attract_weight = alpha * (strength ** self.spec.activation_exponent) * (1.0 + gravity_bonus)

            self.points.update_vector(token, target_anchor.vec, attract_weight, self.step)

        self._add_importance(input_token, self.spec.importance_attract * self.spec.pair_learning_scale)
        self._add_importance(next_token, self.spec.importance_next * self.spec.pair_learning_scale)
        self.gravity.reinforce_pair(input_token, next_token,
            self.spec.gravity_reinforce_amount * self.spec.pair_gravity_scale)
        if recent_override is None:
            recent = self.context.recent_tokens(self.spec.orbit_max_length - 1)
        else:
            recent = recent_override[-(self.spec.orbit_max_length - 1) :]
        if recent and recent[-1] == input_token:
            path = recent + [next_token]
        else:
            path = recent + [input_token, next_token]
        self.gravity.reinforce_sequence(path,
            self.spec.gravity_reinforce_amount * self.spec.pair_gravity_scale)
        self.orbits.observe(path, self.step, 1.0 - min(1.0, error))
        self._maybe_create_loop(next_token, target_anchor.vec, sims)
        # fast=True 배치 모드: cache dirty / shock wave는 배치 종료 후 한 번만 처리.
        if not fast:
            self._mark_cache_dirty()

        if update_context:
            self.context.advance_token(input_token, self.step)
        # fast=True 배치 모드: context path가 안 쌓이므로 repeat 계산 의미없음.
        # shock wave는 배치 종료 후 step_update_batch에서 한 번만 emit+propagate.
        if not fast:
            self._repeat_pattern_similar()
            self._maybe_emit_shock_wave(input_token, preserve_tokens=path)
            self._propagate_shock_waves()

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
                self.stats.last_alignment_score = self.imitation.alignment_score(
                    [tokens[int(torch.argmax(logits).item())]], [next_token]
                )
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
            attract_weight = alpha * (strength ** self.spec.activation_exponent) * (1.0 + gravity_bonus)

            self.points.update_vector(token, target_anchor.vec, attract_weight, self.step)

        self._add_importance(input_token, self.spec.importance_attract)
        self._add_importance(next_token, self.spec.importance_next)
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

    # -------------------------------------------------------
    # 공개 학습 인터페이스
    # -------------------------------------------------------

    def step_update(self, input_token: str, next_token: Optional[str] = None) -> None:
        self.step += 1
        self._ensure_anchor_exists(input_token)
        if next_token is None:
            self.context.advance_token(input_token, self.step)
            return
        self._apply_learning_step(input_token, next_token, update_context=True)

    def step_update_batch(self, pairs: List[Tuple[str, str]], fast: bool = True) -> None:
        if not pairs:
            return

        for input_token, next_token in pairs:
            self._ensure_anchor_exists(input_token)
            self._ensure_anchor_exists(next_token)
        self._ensure_cache()

        # fast 배치 모드: decay를 배치당 1회만 수행.
        # 각 페어마다 호출하면 전체 앵커(8000+)를 32번 순회하는 낭비 발생.
        if fast:
            self.points.decay_importance()
            self.gravity.decay_all()

        # -------------------------------------------------------
        # 핵심 최적화: 유사도 행렬을 mm() 한 번에 계산
        # sims_mat     [N_anchors × B]
        # strengths_mat[N_anchors × B]
        # active_idx_list: List[Tensor] — 토큰별 활성 앵커 인덱스
        # -------------------------------------------------------
        input_tokens = [p[0] for p in pairs]
        active_idx_list, strengths_mat, sims_mat = self._compute_activation_batch(input_tokens)

        recent_tokens: List[str] = []
        for b, (input_token, next_token) in enumerate(pairs):
            self.step += 1
            if not recent_tokens or recent_tokens[-1] != input_token:
                recent_tokens = [input_token]
            local_recent = recent_tokens[-(self.spec.orbit_max_length - 1) :]

            active_idx = active_idx_list[b]
            sims = sims_mat[:, b] if sims_mat.numel() > 0 else torch.empty(0, device=self.device)
            strengths = strengths_mat[:, b] if strengths_mat.numel() > 0 else torch.empty(0, device=self.device)

            self._apply_learning_step_precomputed(
                input_token,
                next_token,
                active_idx=active_idx,
                strengths=strengths,
                sims=sims,
                update_context=not fast,
                recent_override=local_recent,
                fast=fast,
            )
            recent_tokens.append(next_token)

        # 배치 전체 처리 후 cache 한 번만 재빌드 + shock wave emit + 정리
        if fast:
            self._mark_cache_dirty()
            # fast 모드에서는 _apply_learning_step_precomputed 내부에서
            # _maybe_emit_shock_wave가 스킵되므로 배치 끝에 한 번 호출.
            if pairs:
                last_input = pairs[-1][0]
                last_path = recent_tokens[-(self.spec.orbit_max_length):]
                self._maybe_emit_shock_wave(last_input, preserve_tokens=last_path)
            self._propagate_shock_waves()
