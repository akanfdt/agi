# CWM 프로젝트 인계 문서 (즉시 투입용)

이 문서는 다른 AI/개발자가 바로 이어받아 작업할 수 있도록 현재 상태와 다음 할 일을 정리한 것입니다.

## 1) 현재 목표
- CWM v0.9 설계 내용을 코드로 구현하고, `cwm_train.py` 학습 및 `cwm_chat.py` 대화가 정상 동작하도록 안정화.
- `cwm_v0_9_review.md` 체크리스트를 순서대로 처리.

## 2) 핵심 파일
- `cwm_core.py`: CWM 코어 (앵커, 업데이트, 컨텍스트, 감정, 동적 차원, 성능 최적화)
- `cwm_train.py`: 학습 스크립트 (script + namuwiki 로딩, 로그/저장)
- `cwm_chat.py`: 대화 실행기 (컨텍스트 기반 예측 출력)
- `cwm_v0_9_review.md`: v0.9 리뷰 및 구현 체크리스트
- `script.txt`: 2인 대화 스크립트 데이터 (한국어, 인코딩 주의)
- `namuwiki_pretrain.txt`: 나무위키 대용량 텍스트

## 3) 최근 적용된 변화 (요약)
- EMA+표준편차 기반 `error_threshold`, 안정화 구간(stabilization).
- 반복 패턴 억제, 컨텍스트 벡터 기반 예측.
- 감정 극점(좋다/나쁘다) 가중치 반영.
- OOV 문자 앵커(문자 임베딩 기반) 및 저장/로드 지원.
- 벡터화 업데이트 + top-k 활성화 마스킹.
- 유사도 임계값 샘플링, 차원 압축을 CPU에서 계산.
- 동적 차원 추가/삭제/압축 로직 구현.

## 4) 지금 발생 중인 오류 (최우선)
### 증상
`python cwm_train.py` 실행 중 중간 스텝에서 CUDA device-side assert 발생.
스택트레이스는 `_update_similarity_threshold` 내부에서 터짐.

### 마지막 로그 패턴
```
steps=13998 ... (중략)
RuntimeError: CUDA error: device-side assert triggered
  at cwm_core.py:_update_similarity_threshold
```

### 현재 코드에서 적용된 회피책
`_update_similarity_threshold`는 CPU에서 계산하도록 변경됨.
```
idx = torch.randperm(n)[:sample_n]            # CPU idx
vecs = self._anchor_matrix.detach().cpu()[idx]
```
그럼에도 device-side assert가 남아 있음.

### 원인 후보
- `idx` 범위가 `_anchor_matrix` 크기와 일시적으로 어긋나는 타이밍.
- `_anchor_matrix`가 GPU에서 업데이트 중인데 동시에 CPU로 옮겨지는 순간.
- `activated_idx`가 비정상 범위를 포함해 다른 연산에서 CUDA assert를 유발.

## 5) 다음 우선 작업
1) `_update_similarity_threshold`를 완전히 안전하게 만들기
   - 임시로 `if self.device.type == "cuda": return`로 CUDA에서 비활성화하고
     학습이 끝까지 도는지 확인.
   - 또는 `_anchor_matrix`를 CPU 복사하기 전에 `_ensure_cache(force=True)`로 안정화.
2) `activated_idx` 안전성 재점검
   - top-k 마스킹 이후 인덱스 클램프가 충분한지 확인.
3) 오류 해결 후 학습 재실행
   - `python cwm_train.py`로 2~3만 스텝 이상 안정 확인.

## 6) 실행 방법
### 학습
```
python cwm_train.py
```
기본 동작:
- `script.txt`, `namuwiki_pretrain.txt` 로딩
- 200 스텝마다 저장
- 로그에 타임스탬프 출력

### 대화
```
python cwm_chat.py
```
컨텍스트 기반 예측 후보를 출력함.

## 7) 성능 관련 주요 파라미터
`cwm_core.py` (CWMSpec):
- `activation_top_k`: 활성 앵커 상한 (속도/품질 트레이드오프)
- `cache_rebuild_every`: 캐시 재구축 주기
- `similarity_threshold_every`: 유사도 임계 갱신 주기
- `similarity_sample_ratio/min/max`: 샘플링 규모
- `dim_compress_check_every`: 차원 압축 주기

`cwm_train.py`:
- `MAX_VOCAB`, `max_steps`, `SAVE_EVERY_STEPS`

## 8) 미완성 기능 (설계와 차이)
- `maybe_manage_anchors`에 실제 병합/삭제 정책 미구현.
- 다중 토큰 응답 생성(짧은 문장 생성 루틴) 미구현.
- `char_anchors` 학습 규칙은 최소화됨 (정교화 필요).

## 9) 빠른 확인 체크리스트
- `python -c "import torch; print(torch.cuda.is_available())"`
- `python cwm_train.py`가 1만 스텝 이상 오류 없이 진행되는지
- `python cwm_chat.py`에서 후보가 정상 출력되는지

## 10) 파일 스냅샷 (참고)
- `cwm_core.py`는 v0.9 핵심 로직 대부분 반영됨.
- `cwm_v0_9_review.md`에 체크리스트가 있으며, 다음 작업 순서를 명확히 안내.

