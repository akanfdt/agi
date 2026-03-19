from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from cwm_types import CWMSpec, ContextState, l2_normalize


@dataclass
class ContextField:
    spec: CWMSpec
    state: ContextState

    def _context_weight(self, current_step: int, seen_step: int) -> float:
        return self.spec.context_decay ** max(0, current_step - seen_step)

    def advance_token(self, token: str, step: int) -> None:
        self.state.path.append((token, step))
        if len(self.state.path) > self.spec.context_window:
            self.state.path = self.state.path[-self.spec.context_window :]
        sig_tokens = [tok for tok, _ in self.state.path[-self.spec.repeat_sig_len :]]
        if sig_tokens:
            self.state.signatures.append((tuple(sig_tokens), step))
        cutoff = step - self.spec.repeat_window
        self.state.signatures = [(sig, t) for sig, t in self.state.signatures if t >= cutoff]

    def active_tokens(self, current_step: int) -> List[str]:
        weighted = self.active_weights(current_step)
        ranked = sorted(weighted.items(), key=lambda item: item[1], reverse=True)
        return [token for token, _ in ranked[: self.spec.gravity_context_top_k]]

    def active_weights(self, current_step: int) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for token, seen_step in self.state.path:
            weights[token] = weights.get(token, 0.0) + self._context_weight(current_step, seen_step)
        return weights

    def build_context_vector(
        self,
        anchors: Dict[str, torch.Tensor],
        current_step: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        weighted: List[torch.Tensor] = []
        weights: List[float] = []
        for token, seen_step in self.state.path:
            vec = anchors.get(token)
            if vec is None:
                continue
            weight = self._context_weight(current_step, seen_step)
            if weight <= 0.0:
                continue
            weighted.append(vec.to(device) * weight)
            weights.append(weight)
        if not weighted:
            return None
        total = torch.stack(weighted, dim=0).sum(dim=0)
        return l2_normalize(total / max(sum(weights), 1e-8))

    def recent_tokens(self, limit: int) -> List[str]:
        return [token for token, _ in self.state.path[-limit:]]

    def repeat_signature(self) -> tuple[str, ...]:
        if not self.state.signatures:
            return ()
        return self.state.signatures[-1][0]
