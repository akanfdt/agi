"""
Minimal chat loop for CWMCore.
Loads a trained model and prints top-k next-token candidates.
Uses BPE tokenizer (cwm_tokens.json).
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import torch

from bpe_tokenizer import BPETokenizer
from cwm_core import CWMCore


MODEL_PATH = "tmp_birth_only_expansion_check.pt"
VOCAB_PATH = "cwm_tokens.json"


def edit_distance(a: List[str], b: List[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, tok_b in enumerate(b, start=1):
            cur = dp[j]
            if tok_a == tok_b:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[-1]


def sample_token(tokens: List[str], scores: torch.Tensor) -> Optional[str]:
    if scores.numel() == 0:
        return None
    probs = torch.softmax(scores, dim=0)
    idx = int(torch.multinomial(probs, 1).item())
    return tokens[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a trained CWM model.")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the model checkpoint")
    parser.add_argument("--vocab", type=str, default=VOCAB_PATH, help="Path to the tokenizer vocab file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bpe = BPETokenizer.load(args.vocab)
    core = CWMCore.load(args.model)
    print("CWM chat (type 'quit' to exit)")
    print(f"model> {args.model}")
    while True:
        text = input("you> ").strip()
        if text.lower() == "quit":
            break
        tokens = bpe.encode(text)
        if not tokens:
            print("assistant> ...")
            continue
        core.advance_context(tokens)
        last_tok = tokens[-1]
        generated: List[str] = []
        max_len = max(4, int(len(tokens) * 0.6) + 4)
        for _ in range(max_len):
            scored = core.scores_from_context(fallback_token=last_tok)
            if scored is None:
                break
            cand_tokens, scores = scored
            next_tok = sample_token(cand_tokens, scores)
            if next_tok is None:
                break
            generated.append(next_tok)
            core.advance_context([next_tok])
            last_tok = next_tok
        if not generated:
            print("assistant> (no prediction)")
            continue
        print("assistant> " + bpe.decode(generated))
        scored = core.scores_from_context(fallback_token=tokens[-1])
        if scored is not None:
            cand_tokens, scores = scored
            top_vals, top_idx = torch.topk(scores, k=min(5, scores.numel()))
            preview = ", ".join([f"{cand_tokens[int(i.item())]}({v.item():.2f})" for i, v in zip(top_idx, top_vals)])
            print("  candidates:", preview)


if __name__ == "__main__":
    main()
