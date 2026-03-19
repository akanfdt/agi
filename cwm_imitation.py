from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from cwm_types import CWMSpec


@dataclass
class ImitationTrainer:
    spec: CWMSpec

    def imitation_ratio(self, step: int) -> float:
        if step <= self.spec.imitation_warmup_steps:
            return self.spec.imitation_ratio_start
        if step >= self.spec.imitation_cooldown_steps:
            return self.spec.imitation_ratio_end
        span = max(1, self.spec.imitation_cooldown_steps - self.spec.imitation_warmup_steps)
        progress = (step - self.spec.imitation_warmup_steps) / span
        mid_mix = self.spec.imitation_ratio_start + (self.spec.imitation_ratio_mid - self.spec.imitation_ratio_start) * min(progress * 2.0, 1.0)
        end_progress = max(0.0, progress * 2.0 - 1.0)
        return mid_mix + (self.spec.imitation_ratio_end - self.spec.imitation_ratio_mid) * end_progress

    def alignment_score(self, generated: List[str], target: List[str]) -> float:
        if not target and not generated:
            return 1.0
        if not target or not generated:
            return 0.0
        matches = sum(1 for a, b in zip(generated, target) if a == b)
        return matches / max(len(target), len(generated))

    def token_prior(self, candidate_tokens: List[str], recent_tokens: List[str], step: int) -> torch.Tensor:
        ratio = self.imitation_ratio(step)
        recent_set = set(recent_tokens)
        scores = [ratio if token in recent_set else 0.0 for token in candidate_tokens]
        if not scores:
            return torch.empty(0)
        return torch.tensor(scores, dtype=torch.float32)

    def sequence_loss(self, logits_list: List[torch.Tensor], target_indices: List[int]) -> torch.Tensor:
        if not logits_list or not target_indices:
            return torch.tensor(0.0)
        losses = []
        for logits, target_index in zip(logits_list, target_indices):
            target = torch.tensor([target_index], dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        return torch.stack(losses).mean()
