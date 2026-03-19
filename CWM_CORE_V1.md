# CWM Core V1

이 버전의 코어는 유지보수성과 안정성을 우선한 최소 구현입니다.

## 원칙
- `anchors`가 유일한 진실 원천이다.
- 캐시(`token list`, `index map`, `matrix`, `importance vec`)는 항상 재생성 가능한 파생 상태다.
- 학습은 원본 anchor를 갱신한 뒤 캐시를 무효화한다.
- 추론과 학습은 단일 device 정책을 따른다.

## 포함된 기능
- anchor 초기화와 OOV fallback
- context 누적과 context 기반 점수 계산
- 기본 activation 기반 학습 업데이트
- QA memory
- save/load
- 최소 loop anchor 생성

## 의도적으로 미룬 기능
- dynamic dimension add/delete/compress
- 공격적인 auto calibration
- anchor merge/prune 자동화
- emotion bias 자동 조정

## 저장 포맷
- 새 저장 포맷은 `format_version = 2`를 사용한다.
- 캐시는 저장하지 않는다.
- 저장 항목은 spec, step, anchors, importance, context, QA memory, 주요 통계다.

## 다음 단계
- 학습 품질 검증 후 loop 정책 조정
- output scoring 정교화
- anchor lifecycle 정책 복원
- 필요 시 dynamic dimension 기능을 별도 모듈로 추가
