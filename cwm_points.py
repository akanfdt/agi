from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch

from cwm_types import Anchor, CWMSpec, l2_normalize


@dataclass
class PointStore:
    spec: CWMSpec
    device: torch.device
    anchors: Dict[str, Anchor] = field(default_factory=dict)
    char_anchors: Dict[str, Anchor] = field(default_factory=dict)

    def _random_unit_vector(self) -> torch.Tensor:
        raw = (torch.rand(self.spec.dim, device=self.device) * 2 - 1) * self.spec.init_range
        return l2_normalize(raw)

    def _neighbor_profile(
        self,
        vec: torch.Tensor,
        exclude_token: Optional[str] = None,
    ) -> tuple[Optional[torch.Tensor], float]:
        tokens = [tok for tok in self.anchors.keys() if tok != exclude_token]
        if not tokens:
            return None, 0.0

        query = l2_normalize(vec.to(self.device))
        matrix = torch.stack([self.anchors[tok].vec for tok in tokens], dim=0).to(self.device)
        sims = torch.mv(matrix, query)
        k = min(len(tokens), max(4, int(len(tokens) ** 0.5)))
        top_vals, top_idx = torch.topk(sims, k=k)

        positive = torch.clamp(top_vals, min=0.0)
        if float(positive.sum().item()) > 0.0:
            center = l2_normalize(torch.sum(matrix[top_idx] * positive.unsqueeze(1), dim=0))
        else:
            center = l2_normalize(torch.mean(matrix[top_idx], dim=0))

        local_mean = float(top_vals.mean().item())
        global_mean = float(sims.mean().item())
        crowding = max(0.0, local_mean - global_mean)
        return center, crowding

    def _repulsion_direction(self, vec: torch.Tensor, center: Optional[torch.Tensor]) -> torch.Tensor:
        if center is None:
            return torch.zeros_like(vec)
        repel = vec - center.to(self.device)
        norm = float(repel.norm().item())
        if norm > 1e-8:
            return repel / norm

        noise = self._random_unit_vector()
        projected = noise - torch.dot(noise, vec) * vec
        proj_norm = float(projected.norm().item())
        if proj_norm <= 1e-8:
            return noise
        return projected / proj_norm

    def init_vocab(self, vocab: Iterable[str]) -> None:
        for tok in vocab:
            if tok not in self.anchors:
                self.add_anchor_near(tok, self._random_unit_vector())
        self._init_char_anchors(vocab)

    def _init_char_anchors(self, vocab: Iterable[str]) -> None:
        chars = {ch for tok in vocab for ch in tok}
        for ch in chars:
            if ch not in self.char_anchors:
                self.char_anchors[ch] = Anchor(token=ch, vec=self._random_unit_vector())

    def ensure_char_anchor(self, ch: str) -> None:
        if ch not in self.char_anchors:
            self.char_anchors[ch] = Anchor(token=ch, vec=self._random_unit_vector())

    def oov_vector_from_chars(self, token: str) -> Optional[torch.Tensor]:
        vecs: List[torch.Tensor] = []
        for ch in token:
            self.ensure_char_anchor(ch)
            vecs.append(self.char_anchors[ch].vec)
        if not vecs:
            return None
        return l2_normalize(torch.stack(vecs, dim=0).mean(dim=0))

    def add_anchor_near(self, token: str, base_vec: torch.Tensor) -> bool:
        if token in self.anchors:
            return False
        if self.spec.max_anchors > 0 and len(self.anchors) >= self.spec.max_anchors:
            return False
        noise = (torch.rand(self.spec.dim, device=self.device) * 2 - 1) * (self.spec.init_range * 0.1)
        base = l2_normalize(base_vec.to(self.device))
        center, crowding = self._neighbor_profile(base)
        repel = self._repulsion_direction(base, center)
        vec = l2_normalize(base + noise + repel * crowding)
        self.anchors[token] = Anchor(token=token, vec=vec)
        for ch in token:
            self.ensure_char_anchor(ch)
        return True

    def ensure_anchor(self, token: str) -> None:
        if token in self.anchors:
            return
        base_vec = self.oov_vector_from_chars(token)
        if base_vec is None:
            base_vec = self._random_unit_vector()
        self.add_anchor_near(token, base_vec)

    def token_vector(self, tokens: List[str]) -> Optional[torch.Tensor]:
        vecs = [self.anchors[tok].vec for tok in tokens if tok in self.anchors]
        if not vecs:
            return None
        return l2_normalize(torch.stack(vecs, dim=0).mean(dim=0))

    def update_vector(self, token: str, target: torch.Tensor, weight: float, step: int) -> None:
        self.ensure_anchor(token)
        anchor = self.anchors[token]
        anchor.vec = l2_normalize(anchor.vec + weight * (target.to(self.device) - anchor.vec))
        anchor.last_active_step = step

    def decay_importance(self) -> None:
        for anchor in self.anchors.values():
            anchor.importance *= self.spec.importance_decay
