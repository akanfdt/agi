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
    dim: int = 256
    init_range: float = 0.1

    error_window: int = 100
    ema_beta: float = 0.99
    stabilization_ratio: float = 0.1

    alpha_init: float = 0.02
    alpha_final: float = 0.005
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

    output_importance_weight: float = 0.0

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
    activation_top_k: int = 32
    activation_keep_ratio: float = 0.5
    activation_min_keep: int = 3
    activation_exponent: float = 2.0
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
    gravity_reverse_ratio: float = 0.5      # reinforce_pair 역방향 강도 (순방향 대비)
    gravity_forward_prior_scale: float = 1.5  # context_prior에서 forward gravity 가중치
    orbit_min_count: int = 3
    orbit_max_length: int = 8
    orbit_count_scale: float = 10.0   # query 강도 정규화 기준 count
    line_recent_capacity: int = 8
    line_compressed_capacity: int = 24
    line_summary_decay: float = 0.82
    line_summary_strength: float = 0.12
    line_summary_score_weight: float = 0.6
    use_adaptive_priors: bool = False
    adaptive_prior_beta: float = 0.98

    # -------------------------------------------------------
    # 반발력 (군집 분리)
    # -------------------------------------------------------
    # 활성화된 앵커들 사이의 반발: 코사인 유사도가 이 값 이상이면 밀어냄
    repulsion_threshold: float = 0.85
    # 반발력 세기. 0이면 반발 없음 (기존 동작 유지)
    repulsion_strength: float = 0.00

    # -------------------------------------------------------
    # Shock wave (초신성 파동)
    # -------------------------------------------------------
    # 파동이 앵커 위치를 실제로 밀어내는 세기
    shock_repel_strength: float = 0.06
    # instability → amplitude 변환 계수
    shock_amplitude_scale: float = 1.8
    # amplitude 상한
    shock_amplitude_max: float = 2.5
    # 파동 기본 속도
    shock_speed_base: float = 0.26
    # crowding에 따른 속도 보정
    shock_speed_crowding: float = 0.18
    # 매 스텝 amplitude 감쇠율
    shock_decay: float = 0.82
    # ring 강도 컷오프 (이하 무시)
    shock_ring_cutoff: float = 0.12
    # gravity decay 강도 스케일
    shock_gravity_scale: float = 1.6
    # instability 계산의 repeat_signal 기저
    shock_instability_base: float = 0.25
    # 파동 소멸 반경 상한
    shock_max_radius: float = 2.0

    # -------------------------------------------------------
    # 구조 손실 계수
    # -------------------------------------------------------
    structure_loss_active_scale: float = 0.5
    structure_loss_gravity_scale: float = 0.1

    # -------------------------------------------------------
    # 컨텍스트 쿼리 가중치
    # -------------------------------------------------------
    query_weight_input: float = 1.0
    query_weight_memory: float = 0.7
    query_weight_context: float = 0.45
    query_weight_fallback: float = 0.2

    # -------------------------------------------------------
    # 중요도 증가 단위
    # -------------------------------------------------------
    importance_attract: float = 0.01   # input 앵커
    importance_next: float = 0.02      # next 앵커
    importance_observe: float = 0.02   # observe_sentence 내 라인 요약 강도 계수
    imitation_update_scale: float = 0.25   # train_imitation_pair 앵커 이동 세기
    imitation_importance_scale: float = 0.1  # train_imitation_pair importance 증가 단위

    # -------------------------------------------------------
    # 문장 내 토큰 위치 가중치
    # -------------------------------------------------------
    sentence_recency_base: float = 0.7    # 모든 토큰의 기저 가중치
    sentence_recency_range: float = 0.3   # 마지막 토큰에 추가되는 최대 가중치

    # -------------------------------------------------------
    # Context path hop 가중치
    # -------------------------------------------------------
    transition_hop_base: float = 0.5      # 가장 오래된 hop의 가중치
    transition_hop_range: float = 1.5     # 가장 최근 hop에 추가되는 가중치

    # -------------------------------------------------------
    # fast=True 배치 학습 중 pred 측정 주기
    # -------------------------------------------------------
    # fast 모드에서도 N 스텝마다 1번 scores_from_context를 호출해
    # prediction_loss_ema / shock_wave 신호를 살아있게 유지.
    # 100 = 배치 32쌍 기준 약 3배치마다 1회 → 비용 ~1%
    pred_eval_interval: int = 100

    # -------------------------------------------------------
    # 느낌 벡터 (sentence feeling vector) 파라미터
    # -------------------------------------------------------
    # deviation softmax 온도 — 높을수록 핵심 토큰 집중, 낮을수록 균등
    # 3.0은 너무 날카로워서 ctx_err 이상값 발생 → 2.0으로 낮춤
    feeling_temperature: float = 2.0
    # observe_sentence에서 느낌 벡터로 토큰을 당기는 강도
    feeling_attract_scale: float = 2.0
    # 배경 토큰(deviation 낮은 토큰) 최소 가중치
    feeling_min_weight: float = 0.1

    # -------------------------------------------------------
    # 토큰 쌍 학습 보조 역할 스케일
    # -------------------------------------------------------
    # step_update_batch의 alpha에 곱해지는 스케일.
    # 1.0 = 기존과 동일 (주역할)
    # 0.3 = 보조 역할 (느낌 학습이 주, 인접 흐름은 보조)
    pair_learning_scale: float = 0.3
    # step_update_batch의 gravity reinforce에 곱해지는 스케일.
    # 0.3이면 decay 속도보다 강화 속도가 느려 gravity 순감소.
    # 0.6으로 올려서 decay와 균형을 맞춤.
    pair_gravity_scale: float = 0.6

    # -------------------------------------------------------
    # importance 상한 (조각/구두점 독점 방지)
    # -------------------------------------------------------
    # subword 조각(▁ 없이 시작, 길이 짧은 토큰)의 importance 상한
    subword_importance_cap: float = 8.0
    # 단일 구두점/특수문자의 importance 상한
    punct_importance_cap: float = 4.0
    # gravity reinforce 시 짧은 토큰 쌍에 적용할 최소 가중치
    gravity_short_token_min_weight: float = 0.3

    # -------------------------------------------------------
    # 생성 (sampling) 파라미터
    # -------------------------------------------------------
    # softmax temperature: 높을수록 다양, 낮을수록 집중
    gen_temperature: float = 1.5
    # 최근 N개 생성 토큰에 적용할 반복 패널티 (score에서 차감)
    gen_repeat_penalty: float = 3.0
    # 반복 패널티 적용 윈도우 크기
    gen_repeat_window: int = 4


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