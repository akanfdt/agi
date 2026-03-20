"""
cwm_memory.py — 문장 요약 메모리 (MemoryMixin)

담당 역할:
  - 느낌 벡터(Feeling Vector, F) 계산: sentence_summary_vector()
  - 문장 관측 및 앵커 이동: observe_sentence()
  - raw/compressed line memory 관리: summary_memory_vector(), summary_prior()

CWMCore에 믹스인으로 사용됨.
self 속성 의존: spec, device, points, step, raw_line_memory,
               compressed_line_memory, _cache_tokens, _cache_matrix
self 메서드 의존: _ensure_anchor_exists, _mark_cache_dirty,
                _add_importance, alpha, _ensure_cache
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from cwm_types import l2_normalize


class MemoryMixin:

    def reset_line_memory(self) -> None:
        self.raw_line_memory = []
        self.compressed_line_memory = []

    def sentence_summary_vector(self, tokens: List[str]) -> Optional[torch.Tensor]:
        """
        문장의 느낌 벡터 F를 만든다.

        핵심 아이디어:
          - 문장 평균 벡터(배경)에서 가장 멀리 떨어진 토큰이
            이 문장의 핵심 느낌을 담고 있다.
          - deviation(평균과의 거리)을 softmax로 변환해서 가중치로 사용.
          - "나 오늘 너무 슬프다"에서 "슬프다"가 평균에서 가장 다르면
            F가 "슬프다" 방향을 강하게 반영한다.
          - 자주 나오는 조각 토큰("나", "오늘")은 평균 근처에 있으므로
            자동으로 낮은 가중치를 받는다.
        """
        if not tokens:
            return None

        vecs: List[torch.Tensor] = []
        valid_tokens: List[str] = []
        for tok in tokens:
            self._ensure_anchor_exists(tok)
            anchor = self.points.anchors.get(tok)
            if anchor is None:
                continue
            vecs.append(anchor.vec.to(self.device))
            valid_tokens.append(tok)

        if not vecs:
            return None

        if len(vecs) == 1:
            return l2_normalize(vecs[0])

        vec_stack = torch.stack(vecs, dim=0)  # [N, dim]

        # 1단계: 단순 평균 (배경 벡터)
        mean_vec = vec_stack.mean(dim=0)
        mean_norm = float(mean_vec.norm().item())
        if mean_norm < 1e-6:
            return l2_normalize(vec_stack.mean(dim=0))
        mean_vec = mean_vec / (mean_norm + 1e-8)

        # 2단계: 각 토큰이 평균에서 얼마나 다른가 (코사인 거리)
        sims_to_mean = torch.mv(vec_stack, mean_vec).clamp(-1.0, 1.0)
        deviations = 1.0 - sims_to_mean  # 클수록 배경과 다름 = 핵심 토큰

        # 3단계: deviation을 softmax로 가중치 변환
        temp = max(1e-4, self.spec.feeling_temperature)
        weights = torch.softmax(deviations * temp, dim=0)

        min_w = self.spec.feeling_min_weight / max(1, len(vecs))
        weights = weights.clamp(min=min_w)
        weights = weights / weights.sum()

        # 4단계: 가중합으로 느낌 벡터 F 구성
        F = (vec_stack * weights.unsqueeze(-1)).sum(dim=0)
        return l2_normalize(F)

    # -------------------------------------------------------
    # line memory 내부 관리
    # -------------------------------------------------------

    def _all_summary_segments(self) -> List[Dict[str, Any]]:
        return list(self.compressed_line_memory) + list(self.raw_line_memory)

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

    def _compress_summary_memory_if_needed(self) -> None:
        while len(self.raw_line_memory) > self.spec.line_recent_capacity:
            merged = self._merge_summary_segments(self.raw_line_memory[0], self.raw_line_memory[1])
            self.raw_line_memory = self.raw_line_memory[2:]
            self.compressed_line_memory.append(merged)
        while len(self.compressed_line_memory) > self.spec.line_compressed_capacity:
            merged = self._merge_summary_segments(self.compressed_line_memory[0], self.compressed_line_memory[1])
            self.compressed_line_memory = [merged] + self.compressed_line_memory[2:]

    # -------------------------------------------------------
    # 공개 메모리 쿼리
    # -------------------------------------------------------

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

    def summary_prior(self) -> torch.Tensor:
        self._ensure_cache()
        if not self._cache_tokens:
            return torch.empty(0, dtype=torch.float32)
        summary_vec = self.summary_memory_vector()
        if summary_vec is None or self._cache_matrix is None:
            return torch.zeros(len(self._cache_tokens), dtype=torch.float32)
        sims = torch.mv(self._cache_matrix, summary_vec.to(self.device)).detach().cpu()
        return sims * float(self.spec.line_summary_score_weight)

    # -------------------------------------------------------
    # 문장 관측: 느낌 벡터로 앵커 이동 + 메모리 등록
    # -------------------------------------------------------

    def observe_sentence(self, tokens: List[str]) -> None:
        """
        문장 전체를 느낌 벡터 F로 압축하고,
        F 방향으로 문장 내 토큰들을 당긴다.

        핵심 변화:
          - 이전: summary_memory(과거 기억)로 토큰을 당김
          - 지금: 이 문장 자체의 느낌 F로 토큰을 당김
          - 효과: "슬프다"가 포함된 문장들이 공간에서 같은 방향으로 모임
          - deviation 가중치를 재활용해서 핵심 토큰은 강하게,
            배경 토큰은 약하게 당김
        """
        if not tokens:
            return

        vecs: List[torch.Tensor] = []
        valid_tokens: List[str] = []
        for tok in tokens:
            self._ensure_anchor_exists(tok)
            anchor = self.points.anchors.get(tok)
            if anchor is None:
                continue
            vecs.append(anchor.vec.to(self.device))
            valid_tokens.append(tok)

        if not vecs:
            return

        vec_stack = torch.stack(vecs, dim=0)

        if len(vecs) == 1:
            F = l2_normalize(vecs[0])
            token_weights = torch.ones(1, device=self.device)
        else:
            mean_vec = l2_normalize(vec_stack.mean(dim=0))
            sims_to_mean = torch.mv(vec_stack, mean_vec)
            deviations = 1.0 - sims_to_mean
            temp = max(1e-4, self.spec.feeling_temperature)
            token_weights = torch.softmax(deviations * temp, dim=0)
            min_w = self.spec.feeling_min_weight / max(1, len(vecs))
            token_weights = token_weights.clamp(min=min_w)
            token_weights = token_weights / token_weights.sum()
            F = l2_normalize((vec_stack * token_weights.unsqueeze(-1)).sum(dim=0))

        # F 방향으로 각 토큰을 당김 (핵심 토큰은 강하게, 배경 토큰은 약하게)
        base_alpha = self.alpha() * self.spec.line_summary_strength * self.spec.feeling_attract_scale
        for i, tok in enumerate(valid_tokens):
            w = float(token_weights[i].item())
            attract_alpha = base_alpha * w * len(valid_tokens)
            self.points.update_vector(tok, F, attract_alpha, self.step)
            self._add_importance(
                tok,
                self.spec.line_summary_strength * self.spec.importance_observe * w,
            )
        self._mark_cache_dirty()

        self.raw_line_memory.append(
            {
                "vec": F.detach().to(self.device),
                "span": 1,
                "depth": 0,
                "last_step": self.step,
            }
        )
        self._compress_summary_memory_if_needed()
