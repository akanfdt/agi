"""
Minimal trainer for CWMCore using plain text files.
Uses BPE tokenizer (cwm_tokens.json) for subword training.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List
import os

from bpe_tokenizer import BPETokenizer
from cwm_core import CWMCore, CWMSpec
from cwm_quick_eval import quick_eval
from cwm_textio import parse_dialogue_line


VOCAB_PATH = "cwm_tokens.json"
MODEL_PATH = "cwm_model.pt"
USE_EXISTING_VOCAB = True
SAVE_EVERY_STEPS = 10000
RESUME_FROM_MODEL = True
TRAIN_BATCH_SIZE = 32
TOKEN_LOG_EVERY = 1000
STEP_LOG_EVERY_TOKENS = TOKEN_LOG_EVERY * 100

SCRIPT_PATH = "script.txt"
NAMUWIKI_PATH = "namuwiki_pretrain.txt"
TRAIN_SCRIPT = True
TRAIN_NAMUWIKI = False

FILE_TOKEN_LIMITS = {
    SCRIPT_PATH: None,
    NAMUWIKI_PATH: None,
}

VOCAB_TOKEN_LIMITS = {
    SCRIPT_PATH: None,
    NAMUWIKI_PATH: None,
}


def extract_content_text(line: str) -> str:
    return parse_dialogue_line(line).content


def iter_content_lines(path: str) -> Iterable[str]:
    encodings = ["utf-8-sig", "cp949", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, encoding=enc) as f:
                for line in f:
                    content = extract_content_text(line)
                    if not content:
                        continue
                    yield content
            return
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err


def iter_tokens_from_file(path: str, bpe: BPETokenizer) -> Iterable[str]:
    for content in iter_content_lines(path):
        for tok in bpe.encode(content):
            yield tok


def load_vocab() -> List[str]:
    bpe = BPETokenizer.load(VOCAB_PATH)
    return bpe.vocab


def format_metrics(core: CWMCore) -> str:
    metrics = core.get_metrics()
    return (
        f"pred={metrics['prediction_loss']:.4f} "
        f"pred_ema={metrics['prediction_loss_ema']:.4f} "
        f"struct={metrics['structure_loss']:.4f} "
        f"struct_ema={metrics['structure_loss_ema']:.4f} "
        f"imit_ema={metrics['imitation_loss_ema']:.4f} "
        f"align={metrics['alignment_score']:.3f} "
        f"err_ema={metrics['ema_error']:.4f} "
        f"ctx_err={metrics['last_context_error']:.4f} "
        f"repeat={metrics['repeat_ema']:.3f} "
        f"gravity={int(metrics['gravity_edges'])} "
        f"orbits={int(metrics['orbits'])} "
        f"imit_ratio={metrics['imitation_ratio']:.2f}"
    )


def main() -> None:
    paths = []
    if TRAIN_NAMUWIKI:
        paths.append(NAMUWIKI_PATH)
    if TRAIN_SCRIPT:
        paths.append(SCRIPT_PATH)
    spec = CWMSpec()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] building/loading vocab...")
    vocab = load_vocab()
    bpe = BPETokenizer.load(VOCAB_PATH)

    if RESUME_FROM_MODEL and os.path.exists(MODEL_PATH):
        try:
            core = CWMCore.load(MODEL_PATH)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] loaded model: {MODEL_PATH} | "
                f"anchors={len(core.anchors)} | step={core.step}"
            )
        except Exception as exc:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] failed to load {MODEL_PATH}: {exc}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] starting fresh model")
            core = CWMCore(vocab, spec=spec)
    else:
        core = CWMCore(vocab, spec=spec)

    max_steps = 500000
    steps = 0
    last_step_log = -1
    for path in paths:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] loading: {path}")
        core.reset_line_memory()
        token_limit = FILE_TOKEN_LIMITS.get(path, None)
        token_count = 0
        batch_pairs: List[tuple[str, str]] = []
        done = False
        for content in iter_content_lines(path):
            if done:
                break
            tokens = bpe.encode(content)
            if not tokens:
                continue

            prev = None
            for tok in tokens:
                token_count += 1
                if token_count % TOKEN_LOG_EVERY == 0:
                    limit_str = f"/{token_limit}" if token_limit is not None else ""
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] tokens: {token_count}{limit_str} | steps={steps}")
                if token_limit is not None and token_count > token_limit:
                    done = True
                    break
                if prev is None:
                    prev = tok
                    continue
                batch_pairs.append((prev, tok))
                prev = tok
                if len(batch_pairs) >= TRAIN_BATCH_SIZE:
                    core.step_update_batch(batch_pairs, fast=True)
                    steps += len(batch_pairs)
                    batch_pairs = []
                    if steps % SAVE_EVERY_STEPS == 0:
                        core.save(MODEL_PATH)
                    if token_count % STEP_LOG_EVERY_TOKENS == 0 and steps != last_step_log:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] steps={steps} | "
                            f"anchors={len(core.anchors)} | {format_metrics(core)}"
                        )
                        last_step_log = steps
                    if steps >= max_steps:
                        done = True
                        break
        if batch_pairs and steps < max_steps:
            core.step_update_batch(batch_pairs, fast=True)
            steps += len(batch_pairs)
        if steps >= max_steps:
            break
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] done: {path} | "
            f"tokens={token_count} | steps={steps} | {format_metrics(core)}"
        )

        if path == SCRIPT_PATH:
            try:
                # Dialogue pairs are derived structurally from consecutive utterances.
                # No punctuation-based "question" rule is used.
                prev_tokens: List[str] | None = None
                for content in iter_content_lines(path):
                    cur_tokens = bpe.encode(content)
                    if not cur_tokens:
                        continue
                    core.observe_sentence(cur_tokens)
                    if prev_tokens:
                        core.reinforce_dialogue(prev_tokens, cur_tokens)
                        core.update_qa_memory(prev_tokens, cur_tokens)
                        core.train_imitation_pair(prev_tokens, cur_tokens)
                    prev_tokens = cur_tokens
            except Exception as exc:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] dialogue reinforcement skipped: {exc}")
 
    core.save(MODEL_PATH)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] saved: {MODEL_PATH} | "
        f"anchors={len(core.anchors)} | steps={steps} | {format_metrics(core)}"
    )
    quick_eval(core, bpe)


if __name__ == "__main__":
    main()
