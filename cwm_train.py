"""
Trainer for CWMCore — unified per-sentence loop.

변경 사항 (v1 → v2):
  - base token-pair 학습과 observe_sentence / reinforce_dialogue를
    문장 단위 단일 루프로 통합
  - 배치(batch_pairs)를 문장 경계에서 반드시 flush → 문장 간 토큰 쌍 생성 차단
  - 파일 2회 읽기 제거 (dialogue reinforcement가 인라인으로 처리됨)
  - is_dialogue_file 플래그로 대화 파일 / 일반 텍스트 파일 구분
  - train_imitation_pair는 효과 확인 전까지 비활성 유지
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Tuple
import os

from bpe_tokenizer import BPETokenizer
from cwm_core import CWMCore, CWMSpec
from cwm_quick_eval import quick_eval
from cwm_textio import parse_dialogue_line


# ---------------------------------------------------------------------------
# 경로 / 플래그 설정
# ---------------------------------------------------------------------------

VOCAB_PATH = "cwm_tokens.json"
MODEL_PATH = "cwm_model.pt"
RESUME_FROM_MODEL = False

SAVE_EVERY_STEPS = 10000
TRAIN_BATCH_SIZE = 32          # 문장 내 페어가 이 수에 도달하면 즉시 flush
                               # (문장 경계에서도 항상 flush하므로 상한으로만 작동)
TOKEN_LOG_EVERY = 1000
STEP_LOG_EVERY_TOKENS = 5000

SCRIPT_PATH = "script.txt"
NAMUWIKI_PATH = "namuwiki_pretrain.txt"

# (파일경로, 대화파일여부, 토큰상한)
# is_dialogue=True  → reinforce_dialogue + update_qa_memory 활성
# is_dialogue=False → observe_sentence + 토큰 쌍 학습만
TRAIN_FILES: List[Tuple[str, bool, Optional[int]]] = [
    (NAMUWIKI_PATH, False, None),
    (SCRIPT_PATH,   True,  None),
    (SCRIPT_PATH,   True,  None),
    (SCRIPT_PATH,   True,  None),
]

MAX_STEPS = 5_000_000


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def extract_content_text(line: str) -> str:
    return parse_dialogue_line(line).content


def iter_content_lines(path: str) -> Iterable[str]:
    """인코딩 자동 감지 후 content 텍스트만 yield."""
    encodings = ["utf-8-sig", "cp949", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, encoding=enc) as f:
                for line in f:
                    content = extract_content_text(line)
                    if content:
                        yield content
            return
        except UnicodeDecodeError as e:
            last_err = e
    if last_err:
        raise last_err


def load_vocab_and_bpe() -> Tuple[List[str], BPETokenizer]:
    bpe = BPETokenizer.load(VOCAB_PATH)
    return bpe.vocab, bpe


def format_metrics(core: CWMCore) -> str:
    m = core.get_metrics()
    return (
        f"pred={m['prediction_loss']:.4f} "
        f"pred_ema={m['prediction_loss_ema']:.4f} "
        f"struct={m['structure_loss']:.4f} "
        f"struct_ema={m['structure_loss_ema']:.4f} "
        f"imit_ema={m['imitation_loss_ema']:.4f} "
        f"align={m['alignment_score']:.3f} "
        f"err_ema={m['ema_error']:.4f} "
        f"ctx_err={m['last_context_error']:.4f} "
        f"repeat={m['repeat_ema']:.3f} "
        f"gravity={int(m['gravity_edges'])} "
        f"orbits={int(m['orbits'])} "
        f"imit_ratio={m['imitation_ratio']:.2f}"
    )


# ---------------------------------------------------------------------------
# 핵심: 통합 문장 단위 학습 루프
# ---------------------------------------------------------------------------

def train_file(
    core: CWMCore,
    bpe: BPETokenizer,
    path: str,
    is_dialogue: bool,
    token_limit: Optional[int],
) -> None:
    """
    파일 하나를 문장 단위로 순회하며 학습.
    step 관리는 core.step만 기준으로 사용 — 외부 카운터 없음.

    각 문장에서 수행하는 작업 (순서 중요):
      1. observe_sentence(tokens)
         → 요약 메모리에 이 문장을 쌓음.
            summary_prior가 즉시 이 문장의 맥락을 반영하기 시작.
      2. step_update_batch(pairs_within_sentence)
         → 이 문장 내 토큰 쌍만으로 앵커 이동 + gravity + orbit 학습.
            문장 경계를 넘는 페어는 생성되지 않음.
      3. (is_dialogue이고 prev_tokens 있을 때)
         reinforce_dialogue(prev_tokens, cur_tokens)
         → 이전 문장 → 현재 문장 방향의 gravity 강화.
         update_qa_memory(prev_tokens, cur_tokens)
         → QA 패턴 기억.
    """
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"train: {path} | dialogue={is_dialogue} | start_step={core.step}"
    )
    core.reset_line_memory()

    token_count = 0
    last_logged_step = -1
    prev_tokens: Optional[List[str]] = None
    done = False

    for content in iter_content_lines(path):
        if done:
            break

        tokens = bpe.encode(content)
        if not tokens:
            continue

        # ----- 토큰 카운트 / 상한 체크 -----
        token_count += len(tokens)
        if token_count % TOKEN_LOG_EVERY < len(tokens):
            limit_str = f"/{token_limit}" if token_limit is not None else ""
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"tokens: {token_count}{limit_str} | core.step={core.step}"
            )
        if token_limit is not None and token_count > token_limit:
            done = True

        # ----- 1. 요약 메모리에 문장 등록 -----
        core.observe_sentence(tokens)

        # ----- 2. 문장 내 토큰 쌍 학습 -----
        # TRAIN_BATCH_SIZE 초과 시 중간 flush, 문장 끝에서 반드시 flush.
        pairs: List[Tuple[str, str]] = list(zip(tokens[:-1], tokens[1:]))
        for batch_start in range(0, max(1, len(pairs)), TRAIN_BATCH_SIZE):
            batch = pairs[batch_start : batch_start + TRAIN_BATCH_SIZE]
            if not batch:
                continue
            core.step_update_batch(batch, fast=True)
            # core.step은 step_update_batch 내부에서 이미 증가함

            if core.step % SAVE_EVERY_STEPS == 0:
                core.save(MODEL_PATH)

            if token_count % STEP_LOG_EVERY_TOKENS < len(tokens) and core.step != last_logged_step:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] core.step={core.step} | "
                    f"anchors={len(core.anchors)} | {format_metrics(core)}"
                )
                last_logged_step = core.step

            if core.step >= MAX_STEPS:
                done = True
                break

        # ----- 3. 대화 구조 강화 (대화 파일일 때만) -----
        if is_dialogue and prev_tokens is not None and not done:
            core.reinforce_dialogue(prev_tokens, tokens)
            core.update_qa_memory(prev_tokens, tokens)
            # core.train_imitation_pair(prev_tokens, tokens)
            # ^ imitation: 통합 루프 안정화 확인 후 재검토

        # ----- 4. 문장 단위 context 등록 -----
        # fast=True 배치에서 context가 안 쌓이므로 여기서 문장 전체를 등록.
        # scores_from_context / pred_ema가 실제 문맥을 반영하려면 필수.
        if not done:
            core.advance_context(tokens)

        prev_tokens = tokens

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] done: {path} | "
        f"tokens={token_count} | core.step={core.step} | {format_metrics(core)}"
    )


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] loading vocab / bpe...")
    vocab, bpe = load_vocab_and_bpe()

    spec = CWMSpec()

    if RESUME_FROM_MODEL and os.path.exists(MODEL_PATH):
        try:
            core = CWMCore.load(MODEL_PATH)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] loaded: {MODEL_PATH} | "
                f"anchors={len(core.anchors)} | step={core.step}"
            )
        except Exception as exc:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] load failed ({exc}) → fresh model")
            core = CWMCore(vocab, spec=spec)
    else:
        core = CWMCore(vocab, spec=spec)

    for path, is_dialogue, token_limit in TRAIN_FILES:
        if core.step >= MAX_STEPS:
            break
        train_file(core, bpe, path, is_dialogue, token_limit)

    core.save(MODEL_PATH)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] saved: {MODEL_PATH} | "
        f"anchors={len(core.anchors)} | core.step={core.step} | {format_metrics(core)}"
    )
    quick_eval(core, bpe)


if __name__ == "__main__":
    main()