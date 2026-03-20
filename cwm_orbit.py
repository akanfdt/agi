from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from cwm_types import CWMSpec


@dataclass
class OrbitMemory:
    spec: CWMSpec
    path_counts: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    path_strength: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    path_last_seen_step: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    # prefix → {next_token: path} 인덱스 — query를 O(n) → O(1)로 단축
    _prefix_index: Dict[Tuple[str, ...], Dict[str, Tuple[str, ...]]] = field(default_factory=dict)

    def observe(self, path: List[str], step: int, score: float) -> None:
        max_len = min(self.spec.orbit_max_length, len(path))
        for length in range(2, max_len + 1):
            key = tuple(path[-length:])
            self.path_counts[key] = self.path_counts.get(key, 0) + 1
            prev = self.path_strength.get(key, 0.0)
            count = self.path_counts[key]
            self.path_strength[key] = prev + (score - prev) / count
            self.path_last_seen_step[key] = step
            # 인덱스 갱신
            prefix = key[:-1]
            next_token = key[-1]
            if prefix not in self._prefix_index:
                self._prefix_index[prefix] = {}
            self._prefix_index[prefix][next_token] = key

    def query(self, prefix: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not prefix:
            return out
        max_len = min(self.spec.orbit_max_length - 1, len(prefix))
        for length in range(1, max_len + 1):
            key_prefix = tuple(prefix[-length:])
            candidates = self._prefix_index.get(key_prefix)
            if not candidates:
                continue
            for next_token, path in candidates.items():
                count = self.path_counts.get(path, 0)
                if count < self.spec.orbit_min_count:
                    continue
                score = float(self.path_strength.get(path, 0.0)) * min(1.0, count / self.spec.orbit_count_scale)
                out[next_token] = max(out.get(next_token, 0.0), score)
        return out