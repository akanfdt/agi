"""
Fresh 학습 후 앵커 공간 시각화.
cwm_model.pt 덮어쓰지 않음 — cwm_fresh_vis.pt 에 저장.
"""
from __future__ import annotations

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from datetime import datetime
from typing import List, Tuple

from bpe_tokenizer import BPETokenizer
from cwm_core import CWMCore, CWMSpec
from cwm_textio import parse_dialogue_line
from cwm_visualize import build_plot
from pathlib import Path

VOCAB_PATH = "cwm_tokens.json"
SAVE_PATH   = "cwm_fresh_vis.pt"
NAMUWIKI_PATH = "namuwiki_pretrain.txt"
TOKEN_LIMIT = 50_000
BATCH_SIZE  = 32


def iter_content_lines(path: str):
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            with open(path, encoding=enc) as f:
                for line in f:
                    content = parse_dialogue_line(line).content
                    if content:
                        yield content
            return
        except UnicodeDecodeError:
            pass


def train(core: CWMCore, bpe: BPETokenizer) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] train start | limit={TOKEN_LIMIT} tokens")
    token_count = 0
    for content in iter_content_lines(NAMUWIKI_PATH):
        tokens = bpe.encode(content)
        if not tokens:
            continue
        token_count += len(tokens)
        core.observe_sentence(tokens)
        pairs: List[Tuple[str, str]] = list(zip(tokens[:-1], tokens[1:]))
        for i in range(0, max(1, len(pairs)), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            if batch:
                core.step_update_batch(batch, fast=True)
        core.advance_context(tokens)
        if token_count >= TOKEN_LIMIT:
            break
        if token_count % 10_000 < len(tokens):
            print(f"  tokens={token_count} | step={core.step} | anchors={len(core.anchors)}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] done | tokens={token_count} | step={core.step} | anchors={len(core.anchors)}")


def main() -> None:
    bpe = BPETokenizer.load(VOCAB_PATH)
    core = CWMCore(bpe.vocab, spec=CWMSpec())

    train(core, bpe)
    core.save(SAVE_PATH)
    print(f"saved → {SAVE_PATH}")

    output = Path("cwm_fresh_universe.png")
    build_plot(
        core=core,
        output_path=output,
        max_points=1200,
        label_top_n=80,
        edge_top_n=120,
        include_internal=False,
        random_seed=42,
    )
    print(f"visualized → {output}")


if __name__ == "__main__":
    main()
