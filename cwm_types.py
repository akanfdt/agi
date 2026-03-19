from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim == 1:
        return x / (x.norm() + eps)
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@dataclass
class Anchor:
    token: str
    vec: torch.Tensor
    importance: float = 0.0
    last_active_step: int = 0

    def normalize(self) -> None:
        self.vec = l2_normalize(self.vec)


@dataclass
class CWMSpec:
    dim: int = 128
    init_range: float = 0.1

    error_window: int = 100
    ema_beta: float = 0.99
    stabilization_ratio: float = 0.1

    alpha_init: float = 0.08
    alpha_final: float = 0.02
    total_steps: int = 1_000_000

    sigma_init: float = 0.35
    sigma_final: float = 0.18
    sigma_dim_ratio: float = 0.1

    similarity_merge: float = 0.92

    dim_cooldown_steps: int = 100
    dim_min_lifetime_steps: int = 100

    context_decay: float = 0.92
    context_window: int = 32

    repeat_window: int = 20
    repeat_theta: int = 2
    repeat_sig_len: int = 3

    importance_decay: float = 0.995

    min_freq: int = 2
    epoch_steps: int = 5000
    max_anchors: int = 50000
    anchor_merge_corr: float = 0.985
    anchor_merge_every: int = 20000

    vram_target_gb: float = 8.0

    emotion_intensity: float = 0.0
    emotion_bias_step: float = 0.0

    context_weight: float = 0.65
    relation_weight: float = 0.35

    output_importance_weight: float = 0.1

    qa_key_len: int = 6
    qa_weight: float = 0.25
    qa_memory_decay: float = 0.9

    dim_compress_check_every: int = 1000
    dim_compress_corr: float = 0.95
    dim_importance_quantile: float = 0.1
    dim_min_anchors_for_compress: int = 200

    cache_rebuild_every: int = 100
    similarity_threshold_every: int = 2000
    similarity_sample_ratio: float = 0.1
    similarity_sample_max: int = 200
    similarity_sample_min: int = 200
    activation_top_k: int = 128
    activation_keep_ratio: float = 0.72
    activation_min_keep: int = 8
    importance_top_k: int = 8

    imitation_noise_init: float = 0.2
    imitation_noise_min: float = 0.05
    imitation_noise_max: float = 0.6
    imitation_target_ratio: float = 0.2
    imitation_adjust_rate: float = 0.1
    imitation_ratio_start: float = 0.8
    imitation_ratio_mid: float = 0.5
    imitation_ratio_end: float = 0.2
    imitation_warmup_steps: int = 5_000
    imitation_cooldown_steps: int = 50_000

    dialogue_strength: float = 0.2

    auto_calibrate: bool = False
    auto_calibrate_steps: int = 2000
    auto_calibrate_sample_max: int = 5000

    create_loop_threshold_scale: float = 1.5
    min_activation_for_update: float = 1e-4
    exclude_loop_tokens_from_output: bool = True

    gravity_base_decay: float = 0.9995
    gravity_forward_decay: float = 0.999
    gravity_reinforce_amount: float = 0.05
    gravity_context_top_k: int = 64
    gravity_reverse_ratio: float = 0.5
    gravity_forward_prior_scale: float = 1.5
    orbit_min_count: int = 3
    orbit_max_length: int = 8
    line_recent_capacity: int = 8
    line_compressed_capacity: int = 24
    line_summary_decay: float = 0.82
    line_summary_strength: float = 0.12
    line_summary_score_weight: float = 0.6
    use_adaptive_priors: bool = False
    adaptive_prior_beta: float = 0.98

    repulsion_threshold: float = 0.85
    repulsion_strength: float = 0.25

    shock_repel_strength: float = 0.06
    shock_amplitude_scale: float = 1.8
    shock_amplitude_max: float = 2.5
    shock_speed_base: float = 0.26
    shock_speed_crowding: float = 0.18
    shock_decay: float = 0.82
    shock_ring_cutoff: float = 0.12
    shock_gravity_scale: float = 1.6
    shock_instability_base: float = 0.25
    shock_max_radius: float = 2.0

    structure_loss_active_scale: float = 0.5
    structure_loss_gravity_scale: float = 0.1

    query_weight_input: float = 1.0
    query_weight_memory: float = 0.7
    query_weight_context: float = 0.45
    query_weight_fallback: float = 0.2

    importance_attract: float = 0.01
    importance_next: float = 0.02
    importance_observe: float = 0.02
    imitation_update_scale: float = 0.25
    imitation_importance_scale: float = 0.1

    sentence_recency_base: float = 0.7
    sentence_recency_range: float = 0.3

    transition_hop_base: float = 0.5
    transition_hop_range: float = 1.5


@dataclass
class ContextState:
    path: List[Tuple[str, int]] = field(default_factory=list)
    signatures: List[Tuple[Tuple[str, ...], int]] = field(default_factory=list)


@dataclass
class StatsState:
    error_history: List[float] = field(default_factory=list)
    error_count: int = 0
    error_mean: float = 0.0
    error_m2: float = 0.0
    ema_error: float = 0.0
    last_error: float = 0.0
    last_near_dist: float = 1.0
    last_context_error: float = 0.0
    repeat_ema: float = 0.0
    score_count: int = 0
    score_mean: float = 0.0
    score_m2: float = 0.0
    semantic_scale_ema: float = 0.0
    gravity_prior_ema: float = 0.0
    transition_prior_ema: float = 0.0
    orbit_prior_ema: float = 0.0
    imitation_prior_ema: float = 0.0
    summary_prior_ema: float = 0.0
    last_prediction_loss: float = 0.0
    prediction_loss_ema: float = 0.0
    imitation_loss_ema: float = 0.0
    last_structure_loss: float = 0.0
    structure_loss_ema: float = 0.0
    last_alignment_score: float = 0.0


@dataclass
class QAMemoryState:
    memory: Dict[Tuple[str, ...], torch.Tensor] = field(default_factory=dict)
    counts: Dict[Tuple[str, ...], int] = field(default_factory=dict)
