# CWM v1.1 구현 명세서

## 1. 목적

이 문서는 CWM v1.1을 실제로 구현하기 위한 기준 명세서다.  
철학적 설명보다 구현 가능한 구조, 데이터, 학습 규칙, 모듈 경계, 입출력 계약을 우선한다.

구현 목표는 다음과 같다.

- 언어를 점과 중력으로 이루어진 구조로 표현한다.
- 문맥과 문법을 next-token prediction으로 학습한다.
- 모방 학습을 통해 초기 언어 습득과 표현 복사를 가능하게 한다.
- 추가 학습 시 기존 구조 위에 점진적으로 덧입히는 방식을 사용한다.


## 2. 핵심 정의

### 2.1 점(Point)

점은 토큰 또는 개념의 기본 표현 단위다.

- 기본 단위는 토큰이다.
- 확장 시 문맥 상태 점을 허용할 수 있다.
- 모든 점은 고정 차원 벡터를 가진다.

필수 속성:

- `id`
- `token`
- `vector`
- `importance`
- `created_step`
- `last_active_step`

### 2.2 거리(Distance)

거리는 두 점의 의미적 위치 차이다.

- 기본 metric은 cosine distance를 사용한다.
- 필요 시 euclidean distance는 보조 metric으로만 사용한다.

정의:

- `distance(a, b) = 1 - cosine_similarity(a, b)`

### 2.3 중력(Gravity)

중력은 두 점이 실제 경험 속에서 얼마나 강하게 서로 영향을 주는지를 나타내는 연결 강도다.

중력은 거리와 별개로 저장한다.

중력의 종류:

- `base_gravity`: 장기 누적 관계
- `forward_gravity`: 순차 관계
- `context_gravity`: 현재 문맥에서 일시적으로 강화된 관계

중력 값 범위:

- 기본 구현은 `0.0 ~ 1.0`

### 2.4 선(Line)

선은 중력이 반복적으로 작용한 결과로 생긴 경로의 흔적이다.

구현상 선은 독립 객체보다 경로 기록과 중력 분포의 조합으로 해석한다.

즉:

- 선 자체를 먼저 저장하지 않는다.
- 점 시퀀스와 중력 강도의 누적으로부터 선을 파생한다.

### 2.5 궤도(Orbit)

궤도는 반복적으로 나타난 안정 경로다.

구현상 궤도는 다음으로 정의한다.

- 길이 2 이상 토큰 경로
- 최소 등장 횟수 이상
- 평균 중력 또는 평균 점수 기준 이상

### 2.6 문맥장(Context Field)

문맥장은 현재 입력과 최근 생성 결과를 바탕으로 활성화된 점들과 그 가중치 집합이다.

문맥장은 다음을 포함한다.

- 최근 토큰 시퀀스
- 활성 점 목록
- 각 점의 활성 가중치
- 현재 문맥 벡터


## 3. 시스템 범위

v1.1에서 반드시 구현할 범위:

- 점 저장소
- 중력 저장소
- 문맥장
- next-token predictor
- imitation trainer
- orbit memory
- save/load

v1.1에서 제외하거나 noop로 둘 범위:

- dynamic dimension add/delete/compress
- emotion auto tuning
- aggressive self calibration
- 복잡한 loop self-organization


## 4. 모듈 구조

## 4.1 `PointStore`

책임:

- 점 생성
- 점 조회
- OOV fallback 초기화
- 점 벡터 업데이트
- 점 importance 관리

필수 메서드:

- `ensure_point(token: str) -> PointId`
- `get_point(token: str) -> Point`
- `get_vector(token: str) -> Tensor`
- `update_vector(token: str, target: Tensor, weight: float) -> None`
- `decay_importance() -> None`

구현 규칙:

- 점은 단일 진실 원천이다.
- 캐시는 파생 상태이며 점 자체를 대체하지 않는다.

## 4.2 `GravityField`

책임:

- 점 사이 중력 저장
- 순차 중력 저장
- 문맥 기반 임시 중력 계산
- 중력 감쇠 및 강화

필수 데이터:

- `base_gravity[src, dst]`
- `forward_gravity[src, dst]`

필수 메서드:

- `reinforce_pair(src: PointId, dst: PointId, amount: float) -> None`
- `reinforce_sequence(path: list[PointId], amount: float) -> None`
- `get_gravity(src: PointId, dst: PointId) -> float`
- `get_forward_gravity(src: PointId, dst: PointId) -> float`
- `build_context_gravity(active_points: list[PointId]) -> Tensor`

구현 규칙:

- base와 forward는 분리 저장한다.
- context_gravity는 저장하지 않고 매 스텝 계산한다.
- sparse 구조를 우선 사용한다.

## 4.3 `ContextField`

책임:

- 최근 문맥 유지
- 활성 점 계산
- 문맥 벡터 계산
- 반복 패턴 추적

필수 데이터:

- `recent_tokens`
- `recent_point_ids`
- `activation_weights`
- `context_vector`

필수 메서드:

- `advance(tokens: list[str]) -> None`
- `build_context_vector() -> Tensor | None`
- `active_points() -> list[PointId]`
- `repeat_signature() -> tuple[PointId, ...]`

구현 규칙:

- 감쇠 기반 window를 사용한다.
- 최근 입력일수록 높은 가중치를 가진다.
- 문맥장은 저장 가능한 최소 상태만 남긴다.

## 4.4 `OrbitMemory`

책임:

- 반복 경로 저장
- 자주 쓰인 문장 흐름 유지
- predictor에 prior 제공

필수 데이터:

- `path_counts`
- `path_strength`
- `path_last_seen_step`

필수 메서드:

- `observe(path: list[PointId], score: float) -> None`
- `query(prefix: list[PointId]) -> list[tuple[PointId, float]]`

구현 규칙:

- v1.1에서는 길이 2~8 경로만 저장한다.
- 최소 등장 횟수 미만 경로는 orbit로 승격하지 않는다.

## 4.5 `Predictor`

책임:

- next-token 후보 계산
- prediction logits 계산
- prediction loss 계산
- imitation prior와 orbit prior 혼합

입력:

- context vector
- active points
- gravity scores
- orbit priors

출력:

- candidate tokens
- logits
- probability distribution

필수 메서드:

- `score_candidates(context) -> Tensor`
- `predict_next(context, top_k) -> list[tuple[token, score]]`
- `prediction_loss(logits, target_id) -> Tensor`

구현 규칙:

- 후보 전체 vocab를 볼 수 있다.
- 이후 최적화 단계에서 candidate pruning을 추가할 수 있다.

## 4.6 `ImitationTrainer`

책임:

- 입력-목표 쌍 비교
- imitation loss 계산
- 표현 복사 성향 강화
- imitation ratio 스케줄 관리

입력:

- input tokens
- generated tokens
- target tokens

출력:

- imitation loss
- copy alignment score

필수 메서드:

- `imitation_loss(pred_logits, target_tokens) -> Tensor`
- `alignment_score(generated, target) -> float`
- `imitation_ratio(step: int) -> float`

구현 규칙:

- 초기엔 imitation ratio를 높게 둔다.
- 후기로 갈수록 prediction 비중을 높인다.


## 5. 데이터 구조 명세

## 5.1 Point

```python
Point = {
    "id": int,
    "token": str,
    "vector": Tensor[dim],
    "importance": float,
    "created_step": int,
    "last_active_step": int,
}
```

## 5.2 Gravity Entry

```python
GravityEntry = {
    "value": float,
    "count": int,
    "last_updated_step": int,
}
```

## 5.3 Orbit Entry

```python
OrbitEntry = {
    "path": tuple[int, ...],
    "count": int,
    "strength": float,
    "last_seen_step": int,
}
```

## 5.4 Saved State

저장 포맷은 최소한 다음을 포함한다.

```python
SavedState = {
    "format_version": int,
    "spec": dict,
    "step": int,
    "points": ...,
    "gravity": ...,
    "orbit_memory": ...,
    "context_state": ...,
    "stats": ...,
}
```

저장하지 않는 항목:

- 재생성 가능한 dense cache
- 임시 문맥 중력
- 디버그 전용 로그


## 6. 학습 파이프라인

## 6.1 입력 단위

v1.1은 다음 세 가지 입력을 지원한다.

- 일반 문장열
- 대화 쌍
- 모방용 입력-목표 쌍

## 6.2 1 스텝 학습 절차

입력:

- `current_token`
- `next_token`
- 선택적으로 `target_sequence`

절차:

1. `PointStore`에서 현재 토큰과 다음 토큰 점을 확보한다.
2. `ContextField`를 갱신한다.
3. 활성 점과 문맥 벡터를 계산한다.
4. `GravityField`에서 관련 중력을 읽는다.
5. `Predictor`가 다음 토큰 logits를 계산한다.
6. prediction loss를 계산한다.
7. imitation target이 있으면 imitation loss를 계산한다.
8. structure loss를 계산한다.
9. 점 벡터와 중력을 갱신한다.
10. orbit memory를 관측 업데이트한다.

## 6.3 배치 학습 절차

입력:

- 토큰 쌍 리스트

절차:

1. pair마다 point id를 만든다.
2. batch context를 구성한다.
3. predictor logits를 계산한다.
4. prediction/imitation/structure loss를 배치 단위로 합산한다.
5. optimizer step을 적용한다.
6. orbit memory와 stats를 갱신한다.


## 7. 학습 규칙

## 7.1 점 업데이트

업데이트 목적:

- 관련 점은 가까워지게
- 무관한 점은 필요 이상 가까워지지 않게

기본 규칙:

- positive relation이면 target 방향으로 이동
- negative sample이면 멀어지는 방향으로 이동

v1.1 기본식:

- `v_i <- normalize(v_i + lr * w * (target - v_i))`

## 7.2 중력 업데이트

중력 강화 조건:

- 같은 문맥에서 활성화
- 순차적으로 이어짐
- 모방 성공
- 예측 성공

기본 규칙:

- `gravity <- clamp(gravity + delta, 0, 1)`

감쇠 규칙:

- 장기간 쓰이지 않는 중력은 EMA 또는 decay를 적용한다.

## 7.3 구조 loss

structure loss는 최소 두 부분으로 구성한다.

- positive pair pull
- negative pair push

v1.1 기본 형태:

- positive는 cosine similarity 상승
- negative는 margin 이하로 벌어짐

## 7.4 imitation ratio 스케줄

기본 스케줄:

- 초기: `0.8`
- 중기: `0.5`
- 후기: `0.2`

구현 함수:

- step 기반 piecewise schedule


## 8. Loss 명세

## 8.1 Prediction Loss

입력:

- `logits`
- `target_token_id`

함수:

- cross entropy

## 8.2 Imitation Loss

입력:

- `generated_logits`
- `target_sequence`

함수:

- sequence cross entropy

보조 항목:

- alignment bonus 또는 edit-distance penalty

## 8.3 Structure Loss

입력:

- positive pairs
- negative pairs

함수:

- contrastive 또는 cosine margin loss

## 8.4 총 loss

```python
total_loss = prediction_loss + 0.5 * imitation_loss + 0.1 * structure_loss
```

초기 구현에서는 이 가중치를 고정하고, 이후 튜닝 대상으로 둔다.


## 9. 추론 명세

입력:

- 현재 사용자 입력
- 현재 context state

절차:

1. 입력 토큰을 점으로 변환한다.
2. context field를 갱신한다.
3. gravity field와 orbit memory를 이용해 후보를 점수화한다.
4. imitation prior와 prediction prior를 혼합한다.
5. 다음 토큰을 샘플링 또는 argmax로 선택한다.
6. 문장 종료 조건까지 반복한다.

종료 조건:

- EOS 토큰
- 최대 길이 도달
- score threshold 이하


## 10. 저장/로드 명세

저장 대상:

- spec
- step
- points
- gravity field
- orbit memory
- context state
- stats

로드 요구사항:

- 이전 포맷과 완전 호환은 요구하지 않는다.
- `format_version` 기반 분기 로드를 허용한다.
- dense cache는 로드 후 재생성한다.


## 11. 테스트 명세

## 11.1 단위 테스트

- 점 생성 시 차원 일치
- OOV 초기화 동작
- gravity 강화/감쇠 동작
- context vector 계산 동작
- orbit 승격 조건 동작
- save/load roundtrip

## 11.2 통합 테스트

- 짧은 문장열로 next-token prediction 학습
- 대화 쌍으로 imitation 학습
- 저장 후 재로드 뒤 추론 가능
- 추가 학습 후 기존 구조 유지 여부 확인

## 11.3 회귀 테스트

- NaN 없음
- device mismatch 없음
- invalid index 없음
- 빈 context에서도 안전 동작
- 미등록 토큰에도 안전 동작


## 12. 구현 순서

### 1단계

- `PointStore`
- `ContextField`
- `Predictor`
- prediction loss

### 2단계

- `GravityField`
- 점/중력 동시 업데이트
- structure loss

### 3단계

- `ImitationTrainer`
- imitation ratio
- sequence imitation loss

### 4단계

- `OrbitMemory`
- orbit prior
- 반복 경로 강화

### 5단계

- 저장 포맷 고정
- 통합 테스트
- 추가 학습 안정화


## 13. 미결정 사항

아래 항목은 v1.1 구현 중 실험으로 확정한다.

- gravity를 dense로 둘지 sparse로 둘지
- orbit path 최대 길이
- negative sampling 비율
- imitation ratio 감소 스케줄 상세값
- structure loss margin 값


## 14. 최종 구현 기준

CWM v1.1은 다음 조건을 만족하면 구현 완료로 본다.

- 점과 중력이 분리된 구조로 저장된다.
- next-token prediction이 동작한다.
- imitation loss가 실제 학습에 반영된다.
- orbit memory가 반복 경로를 저장한다.
- save/load 후 추가 학습이 가능하다.
- 문맥 기반 생성이 가능하다.

이 문서는 v1.1 구현의 기준 명세서이며, 이후 코드는 이 문서에 맞춰 모듈 단위로 분리 구현한다.
