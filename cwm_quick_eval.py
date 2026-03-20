from __future__ import annotations

from typing import List

import torch

from bpe_tokenizer import BPETokenizer
from cwm_core import CWMCore


PROMPTS = [
    "안녕, 네 이름은 뭐야?",
    "내 이름은 리리야. 기억해 줘.",
    "오늘 기분이 어때?",
    "나는 지금 공부하고 있어.",
    "우리가 방금 무슨 이야기를 했지?",
    "짧게 자기소개를 해줘.",
]


def _sample_token(
    tokens: List[str],
    scores: torch.Tensor,
    temperature: float,
    recent: List[str],
    repeat_penalty: float,
    repeat_window: int,
) -> str | None:
    if scores.numel() == 0:
        return None
    # 반복 패널티: 최근 생성 토큰이면 score 차감
    scores = scores.clone()
    recent_set = recent[-repeat_window:]
    for i, tok in enumerate(tokens):
        if tok in recent_set:
            scores[i] -= repeat_penalty
    probs = torch.softmax(scores / max(temperature, 1e-4), dim=0)
    idx = int(torch.multinomial(probs, 1).item())
    return tokens[idx]


def quick_eval(core: CWMCore, bpe: BPETokenizer, max_gen: int = 12) -> None:
    spec = core.spec
    print("== quick_eval ==")
    for prompt in PROMPTS:
        tokens = bpe.encode(prompt)
        if not tokens:
            continue
        core.advance_context(tokens)
        last_tok = tokens[-1]
        generated: List[str] = []
        for _ in range(max_gen):
            scored = core.scores_from_context(fallback_token=last_tok)
            if scored is None:
                break
            cand_tokens, scores = scored
            next_tok = _sample_token(
                cand_tokens,
                scores,
                temperature=spec.gen_temperature,
                recent=generated,
                repeat_penalty=spec.gen_repeat_penalty,
                repeat_window=spec.gen_repeat_window,
            )
            if next_tok is None:
                break
            generated.append(next_tok)
            core.advance_context([next_tok])
            last_tok = next_tok
        text = bpe.decode(generated) if generated else "(no prediction)"
        print(f"you> {prompt}")
        print(f"assistant> {text}")