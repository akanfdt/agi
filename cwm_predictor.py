from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from cwm_types import CWMSpec


@dataclass
class Predictor:
    spec: CWMSpec
    device: torch.device

    def score_candidates(
        self,
        query: torch.Tensor,
        candidate_matrix: torch.Tensor,
        importance_vec: torch.Tensor,
        gravity_prior: torch.Tensor | None = None,
        summary_prior: torch.Tensor | None = None,
        orbit_prior: torch.Tensor | None = None,
        imitation_prior: torch.Tensor | None = None,
        repeat_penalty: float = 0.0,
    ) -> torch.Tensor:
        query = query.to(self.device)
        matrix = candidate_matrix.to(self.device)
        scores = torch.mv(matrix, query)
        if importance_vec.numel() == scores.numel():
            scores = scores + self.spec.output_importance_weight * torch.sigmoid(importance_vec.to(self.device))
        if gravity_prior is not None and gravity_prior.numel() == scores.numel():
            scores = scores + gravity_prior.to(self.device)
        if summary_prior is not None and summary_prior.numel() == scores.numel():
            scores = scores + summary_prior.to(self.device)
        if orbit_prior is not None and orbit_prior.numel() == scores.numel():
            scores = scores + orbit_prior.to(self.device)
        if imitation_prior is not None and imitation_prior.numel() == scores.numel():
            scores = scores + imitation_prior.to(self.device)
        if repeat_penalty > 0.0:
            scores = scores / (1.0 + repeat_penalty)
        return scores

    def prediction_loss(self, logits: torch.Tensor, target_index: int) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        target = torch.tensor([target_index], dtype=torch.long, device=logits.device)
        return torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)

    def normalize_scores(
        self,
        scores: torch.Tensor,
        score_mean: float,
        score_m2: float,
        score_count: int,
        repeat_ema: float,
    ) -> tuple[torch.Tensor, float, float, int]:
        if scores.numel() == 0:
            return scores, score_mean, score_m2, score_count
        mean = float(scores.mean().item())
        var = float(scores.var(unbiased=False).item())
        n = int(scores.numel())
        total = score_count + n
        if total > 0:
            delta = mean - score_mean
            score_mean += delta * n / total
            score_m2 += var * n + (delta * delta) * score_count * n / total
            score_count = total
        std = math.sqrt(max(1e-8, score_m2 / max(1, score_count)))
        normalized = (scores - score_mean) / std
        if repeat_ema > 0.0:
            normalized = normalized / (1.0 + repeat_ema)
        return normalized, score_mean, score_m2, score_count
