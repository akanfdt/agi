from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable

from bpe_tokenizer import (
    WORD_ONLY, WORD_PATTERN,
    train_bpe,
    save_word_counts, load_word_counts,
)


CORPUS_PATH    = "./namuwiki_pretrain.txt"
SAVE_PATH      = "./cwm_tokens.json"
VOCAB_SIZE     = 8000
MIN_FREQ       = 2
MAX_WORD_TYPES = 200000
COUNTS_CACHE   = "./bpe_word_counts.json"


def iter_lines(path: str) -> Iterable[str]:
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            with open(path, encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
            return
        except UnicodeDecodeError:
            continue


def main() -> None:
    print("=" * 60)
    print("CWM BPE Tokenizer Builder")
    print(f"corpus    : {CORPUS_PATH}")
    print(f"vocab_size: {VOCAB_SIZE}  |  min_freq: {MIN_FREQ}")
    print("=" * 60)

    # ── 수정 핵심 ────────────────────────────────────────────
    # 이전 코드는 캐시 있을 때 train_bpe_from_counts()를 불렀는데
    # 그 함수가 merges=[] 로 반환 → BPE 병합이 전혀 안 됐음.
    # 수정: 캐시 유무와 관계없이 항상 bpe_tokenizer.train_bpe() 호출.
    # ─────────────────────────────────────────────────────────

    if os.path.exists(COUNTS_CACHE):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시 로드: {COUNTS_CACHE}")
        word_counts, punct_counts = load_word_counts(COUNTS_CACHE)

        # 캐시된 word_counts를 가짜 코퍼스 줄로 변환해서 train_bpe에 넘김
        # (train_bpe가 내부에서 카운팅을 다시 하므로 이렇게 넘겨야 함)
        def lines_from_cache() -> Iterable[str]:
            for word, freq in word_counts.items():
                yield " ".join([word] * min(freq, 50))

        print(f"[{datetime.now().strftime('%H:%M:%S')}] BPE 병합 학습 중...")
        symbols, merges = train_bpe(
            lines_from_cache(),
            vocab_size=VOCAB_SIZE,
            max_word_types=MAX_WORD_TYPES,
            min_freq=1,           # 캐시에서 이미 min_freq 필터됨
            log_every_merges=200,
        )

    else:
        # 캐시 없음 → 나무위키 전체 읽기 (첫 실행 시 한 번만)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 나무위키 카운팅 + BPE 학습 중...")
        word_counts: dict = {}
        punct_counts: dict = {}

        def counting_lines() -> Iterable[str]:
            for i, line in enumerate(iter_lines(CORPUS_PATH), 1):
                for raw in WORD_PATTERN.findall(line):
                    if not raw.strip():
                        continue
                    if WORD_ONLY.match(raw):
                        word_counts[raw] = word_counts.get(raw, 0) + 1
                    else:
                        punct_counts[raw] = punct_counts.get(raw, 0) + 1
                if i % 200000 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] lines={i}")
                yield line

        symbols, merges = train_bpe(
            counting_lines(),
            vocab_size=VOCAB_SIZE,
            max_word_types=MAX_WORD_TYPES,
            min_freq=MIN_FREQ,
            log_every_merges=200,
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 캐시 저장: {COUNTS_CACHE}")
        save_word_counts(COUNTS_CACHE, word_counts, punct_counts)

    # ── 저장 ──────────────────────────────────────────────────
    data = {
        "type": "bpe",
        "vocab_size": VOCAB_SIZE,
        "unk_token": "<unk>",
        "vocab": symbols,
        "merges": [f"{a} {b}" for a, b in merges],
    }
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 저장: {SAVE_PATH}")
    print(f"symbols: {len(symbols)}  |  merges: {len(merges)}")

    # ── 동작 확인 ─────────────────────────────────────────────
    if merges:
        from bpe_tokenizer import BPETokenizer
        bpe = BPETokenizer.load(SAVE_PATH)
        print("\n[토큰화 샘플]")
        for t in ["안녕하세요", "자연어처리", "딥러닝", "대한민국"]:
            print(f"  {t!r:12s} → {bpe.encode(t)}")
    else:
        print("\n경고: merges가 여전히 0입니다. bpe_tokenizer.py를 확인하세요.")

    print("\n다음 단계: python cwm_train.py")


if __name__ == "__main__":
    main()