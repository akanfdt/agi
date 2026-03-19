from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List

import torch

from cwm_types import CWMSpec


GravityMap = Dict[str, Dict[str, float]]


def _nested_get(table: GravityMap, src: str, dst: str) -> float:
    return float(table.get(src, {}).get(dst, 0.0))


def _nested_add(table: GravityMap, src: str, dst: str, amount: float) -> None:
    bucket = table.setdefault(src, {})
    bucket[dst] = max(0.0, min(1.0, float(bucket.get(dst, 0.0) + amount)))


@dataclass
class GravityField:
    spec: CWMSpec
    base_gravity: GravityMap = field(default_factory=dict)
    forward_gravity: GravityMap = field(default_factory=dict)

    def get_gravity(self, src: str, dst: str) -> float:
        return _nested_get(self.base_gravity, src, dst)

    def get_forward_gravity(self, src: str, dst: str) -> float:
        return _nested_get(self.forward_gravity, src, dst)

    def reinforce_pair(self, src: str, dst: str, amount: float | None = None) -> None:
        delta = self.spec.gravity_reinforce_amount if amount is None else amount
        _nested_add(self.base_gravity, src, dst, delta)
        _nested_add(self.base_gravity, dst, src, delta * self.spec.gravity_reverse_ratio)
        _nested_add(self.forward_gravity, src, dst, delta)

    def reinforce_sequence(self, path: List[str], amount: float | None = None) -> None:
        if len(path) < 2:
            return
        delta = self.spec.gravity_reinforce_amount if amount is None else amount
        for src, dst in zip(path[:-1], path[1:]):
            self.reinforce_pair(src, dst, delta)

    def context_prior(self, active_tokens: Iterable[str], candidate_tokens: List[str]) -> torch.Tensor:
        scores = []
        active = list(active_tokens)
        active_count = max(1, len(active))
        for cand in candidate_tokens:
            total = 0.0
            for tok in active:
                total += self.get_gravity(tok, cand)
                total += self.get_forward_gravity(tok, cand) * self.spec.gravity_forward_prior_scale
            scores.append(math.tanh(total / active_count))
        if not scores:
            return torch.empty(0)
        return torch.tensor(scores, dtype=torch.float32)

    def forward_crowding(self, token: str) -> float:
        edges = self.forward_gravity.get(token, {})
        if not edges:
            return 0.0
        total = sum(edges.values())
        count = len(edges)
        avg_strength = total / count
        edge_scale = math.log1p(count) / math.log1p(self.spec.gravity_context_top_k)
        return float(min(1.0, avg_strength * edge_scale))

    def supernova_decay(self, token: str, preserve_tokens: List[str], strength: float) -> None:
        preserve_set = set(preserve_tokens)
        for table in (self.base_gravity, self.forward_gravity):
            edges = table.get(token, {})
            for dst in list(edges.keys()):
                if dst in preserve_set:
                    continue
                edges[dst] = max(0.0, edges[dst] - strength)
                if edges[dst] == 0.0:
                    del edges[dst]
