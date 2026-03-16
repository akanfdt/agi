# CWM 형태소 학습기 v3
# 공백 분리 기반 토크나이저
# PMI 병합 제거 → 공백으로 나눈 단어 자체가 토큰

import re
import json
from collections import defaultdict

SCRIPT_PATH = "./script.txt"
SAVE_PATH   = "./cwm_tokens.json"

# ── 1. 스크립트 로드 ──────────────────────────────────────────
def load_script(path):
    lines = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            if line:
                lines.append(line)
    return lines

# ── 2. 공백 기반 토큰화 ───────────────────────────────────────
def tokenize(line):
    """
    공백으로 나누고 문장부호만 분리
    "만나서 반가워!" → ["만나서", "반가워", "!"]
    한국어는 공백 자체가 의미 단위 경계
    """
    # 한글/영문/숫자 덩어리와 문장부호 분리
    pattern = re.compile(r'[가-힣a-zA-Z0-9]+|[^가-힣a-zA-Z0-9\s]')
    return pattern.findall(line)

def tokenize_korean_only(line):
    """한글 단어만 추출 (학습용)"""
    tokens = tokenize(line)
    return [t for t in tokens if re.match(r'^[가-힣]+$', t)]

# ── 3. 어휘집 구축 ────────────────────────────────────────────
def build_vocabulary(lines, min_freq=2):
    """
    공백으로 나눈 단어들의 빈도 계산
    min_freq 이상 등장한 단어만 어휘집에 포함
    """
    freq = defaultdict(int)
    for line in lines:
        tokens = tokenize(line)
        for tok in tokens:
            freq[tok] += 1

    # 빈도순 정렬
    vocab = sorted(freq.items(), key=lambda x: -x[1])
    vocab = [(tok, cnt) for tok, cnt in vocab if cnt >= min_freq]

    return vocab

# ── 4. 저장 ───────────────────────────────────────────────────
def save_results(vocab, lines, path):
    # 샘플 토큰화
    samples = []
    for line in lines[:10]:
        tokens = tokenize(line)
        samples.append(" | ".join(tokens))

    data = {
        "vocabulary": [{"token": tok, "freq": freq} for tok, freq in vocab],
        "sample_tokenized": samples
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {path}")

# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("CWM 형태소 학습기 v3")
    print("공백 분리 기반 토크나이저")
    print("=" * 50)

    print(f"\n1. 스크립트 로드")
    lines = load_script(SCRIPT_PATH)
    print(f"   대사 {len(lines)}개")

    print(f"\n2. 어휘집 구축")
    vocab = build_vocabulary(lines, min_freq=2)

    # 전체 토큰 수
    total_tokens = sum(cnt for _, cnt in vocab)
    korean_vocab = [(tok, cnt) for tok, cnt in vocab
                    if re.match(r'^[가-힣]+$', tok)]

    print(f"   전체 어휘: {len(vocab)}개")
    print(f"   한글 어휘: {len(korean_vocab)}개")
    print(f"   총 토큰 수: {total_tokens}개")

    print(f"\n발견된 한글 단어 TOP 30:")
    print("-" * 40)
    for tok, cnt in korean_vocab[:30]:
        bar = "█" * min(cnt // 10, 20)
        print(f"  {tok:12s} {bar} ({cnt})")

    print(f"\n3. 토큰화 샘플 (상위 5개):")
    print("-" * 40)
    for line in lines[:5]:
        tokens = tokenize(line)
        print(f"  원문: {line}")
        print(f"  토큰: {' | '.join(tokens)}")
        print()

    save_results(vocab, lines, SAVE_PATH)

    print(f"\n한글 단어 총 {len(korean_vocab)}개")
    print("완료! 다음 단계: python anchor.py")

if __name__ == "__main__":
    main()