"""
cwm_shockwave.py — 충격파 & 반발력 (ShockWaveMixin)

담당 역할:
  - 불안정성 감지 → 충격파 발생: _maybe_emit_shock_wave()
  - 충격파 전파 및 앵커/gravity 밀어내기: _propagate_shock_waves()
  - 활성 앵커 간 반발력 계산: _compute_repulsion()

CWMCore에 믹스인으로 사용됨.
self 속성 의존: spec, device, points, gravity, stats, shock_waves,
               stabilization_steps, step, similarity_threshold,
               _cache_matrix, _cache_tokens
self 메서드 의존: _ensure_cache, _mark_cache_dirty
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import torch

from cwm_types import l2_normalize


class ShockWaveMixin:

    def _maybe_emit_shock_wave(self, center_token: str, preserve_tokens: List[str]) -> None:
        # 안정화 전에는 발동하지 않음 — gravity가 형성되기 전에 decay하면 역효과
        if self.step < self.stabilization_steps:
            return

        if center_token not in self.points.anchors:
            return

        # 공간 기반 트리거: 너무 가까운 앵커가 있으면 발동
        # _compute_repulsion이 sim > repulsion_threshold인 앵커를 검출
        self._ensure_cache()
        if self._cache_matrix is None or not self._cache_tokens:
            return
        repel = self._compute_repulsion(center_token, set(preserve_tokens))
        if repel is None:
            return  # 아무도 repulsion_threshold 이상 가깝지 않음 → 발동 불필요

        # crowding은 파동 강도/속도에만 사용 (트리거 조건 아님)
        crowding = self.gravity.forward_crowding(center_token)
        amplitude = min(
            self.spec.shock_amplitude_max,
            self.spec.shock_amplitude_scale * (1.0 + crowding),
        )

        center_vec = self.points.anchors[center_token].vec.detach().clone().to(self.device)
        self.shock_waves.append(
            {
                "center_token": center_token,
                "center_vec": center_vec,
                "amplitude": amplitude,
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
        cache_changed = False
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

            top_k = min(self.spec.gravity_context_top_k, ring.numel())
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

                    anchor = self.points.anchors.get(token)
                    if anchor is not None and self.spec.shock_repel_strength > 0.0:
                        repel_dir = anchor.vec - center_vec
                        repel_norm = float(torch.norm(repel_dir).item())
                        if repel_norm > 1e-6:
                            repel_dir = repel_dir / repel_norm
                            push = amplitude * float(ring_strength) * self.spec.shock_repel_strength
                            anchor.vec = l2_normalize(anchor.vec + repel_dir * push)
                            cache_changed = True

            amplitude *= float(wave.get("decay", self.spec.shock_decay))
            if amplitude > 1e-3 and radius < self.spec.shock_max_radius:
                updated = dict(wave)
                updated["radius"] = radius
                updated["amplitude"] = amplitude
                remaining.append(updated)

        if cache_changed:
            self._mark_cache_dirty()
        self.shock_waves = remaining

    def _compute_repulsion(self, token: str, exclude_tokens: Set[str]) -> Optional[torch.Tensor]:
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
            push = anchor.vec - self._cache_matrix[i].to(self.device)
            repel_vec = repel_vec + push * float(sim_val)
            count += 1
        if count == 0:
            return None
        norm = float(torch.norm(repel_vec).item())
        if norm < 1e-6:
            return None
        return repel_vec / norm
