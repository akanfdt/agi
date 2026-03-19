# Script-Only Long Run Report

Generated: 2026-03-19

## Summary

`script.txt`만 사용해 fresh init으로 `6000`, `12000` 토큰까지 학습한 체크포인트를 비교했다.  
결론은, 더 학습할수록 `gravity`와 `orbit` 수는 늘지만 실제 출력은 거의 개선되지 않았고, 여전히 짧은 말버릇 반복에 강하게 잠긴다.

즉 현재 구조에서는 `script` epoch 증가만으로 자연스러운 문장 생성이 자동으로 좋아진다고 보기 어렵다.

## Token Budget 6000

- Model: `tmp_script_only_6000.pt`
- Metrics: `pred=4.3449 pred_ema=7.0140 struct=0.3975 struct_ema=0.8262 imit_ema=825.2232 align=0.200 err_ema=0.6571 ctx_err=0.0000 repeat=0.000 gravity=2257 orbits=1247 imit_ratio=0.79`

### Quick Eval

- Prompt: `안녕, 네 이름은 뭐야?`
- Output: `나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는`

- Prompt: `오늘 기분이 어때?`
- Output: `나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는`

- Prompt: `짧게 자기소개를 해줘.`
- Output: `나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도`

- Prompt: `네 이름은 릴리아야. 기억해 줘.`
- Output: `나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도`

## Token Budget 12000

- Model: `tmp_script_only_12000.pt`
- Metrics: `pred=5.1380 pred_ema=6.8053 struct=0.4954 struct_ema=0.7685 imit_ema=559.4533 align=0.200 err_ema=0.5738 ctx_err=0.0000 repeat=0.000 gravity=3579 orbits=3022 imit_ratio=0.71`

### Quick Eval

- Prompt: `안녕, 네 이름은 뭐야?`
- Output: `나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는`

- Prompt: `오늘 기분이 어때?`
- Output: `나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는 나는`

- Prompt: `짧게 자기소개를 해줘.`
- Output: `나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도`

- Prompt: `네 이름은 릴리아야. 기억해 줘.`
- Output: `나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도 나도`

## Interpretation

- `12000`은 `6000`보다 `gravity`와 `orbit`가 더 많이 자랐다.
- 하지만 출력은 사실상 동일하다.
- 따라서 현재 병목은 학습량 부족만이 아니라, 생성 단계가 소수의 강한 응답 패턴으로 붕괴하는 구조 문제일 가능성이 높다.

## Recommendation

- 당장은 `namuwiki_pretrain.txt`를 섞기보다 현재 구조 병목을 먼저 해결하는 편이 낫다.
- 다음 실험은 단순 epoch 증가보다 다음 중 하나가 더 가치 있다.
  - summary가 실제 query를 얼마나 지배하는지 추가 디버깅
  - `나는`, `나도` 같은 반복 응답이 왜 고정되는지 후보 점수 분해
  - imitation이 표현 다양성을 늘리는 대신 특정 응답 습관을 굳히는지 점검
