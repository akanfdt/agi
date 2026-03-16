# CWM Pretraining
# 나무위키로 한국어 기초 학습 후 대화 스크립트로 파인튜닝
# 순서: pretrain.py → anchor.py (파인튜닝)

import re
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

NAMUWIKI_PATH = "./namuwiki_pretrain.txt"
SCRIPT_PATH   = "./script.txt"
VOCAB_PATH    = "./cwm_tokens.json"
PRETRAIN_PATH = "./cwm_pretrained.pt"

# 나무위키에서 사용할 문장 수
NAMUWIKI_SAMPLES = 500000   # 10만 개로 시작
MIN_FREQ         = 10        # 최소 5번 이상 등장한 단어만

DIM        = 128
HIDDEN     = 256
NUM_LAYERS = 2
SEQ_LEN    = 10
EPOCHS     = 3       # Pretraining은 적은 에폭으로 (데이터 많아서)
LR         = 0.001
BATCH_SIZE = 256     # Pretraining은 큰 배치

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. 토크나이저 ─────────────────────────────────────────────
def tokenize(line):
    pattern = re.compile(r'[가-힣a-zA-Z0-9]+|[^가-힣a-zA-Z0-9\s]')
    tokens  = pattern.findall(line)
    return [t for t in tokens if re.match(r'^[가-힣]+$', t)]

# ── 2. 나무위키 + 스크립트 통합 어휘집 구축 ──────────────────
def build_unified_vocab():
    """
    나무위키 어휘 + 스크립트 어휘 통합
    나무위키: 한국어 기초 어휘
    스크립트: 대화체 어휘 (반드시 포함)
    """
    freq = defaultdict(int)

    # 1. 나무위키에서 어휘 수집
    print(f"   나무위키 어휘 수집 중... ({NAMUWIKI_SAMPLES}개)")
    with open(NAMUWIKI_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NAMUWIKI_SAMPLES:
                break
            for tok in tokenize(line.strip()):
                freq[tok] += 1

    # 2. 스크립트 어휘는 전부 포함 (빈도 무관)
    print(f"   스크립트 어휘 수집 중...")
    with open(SCRIPT_PATH, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            for tok in tokenize(line):
                freq[tok] += MIN_FREQ + 1  # 강제로 포함

    # 빈도 필터
    vocab = {"<PAD>": 0, "<EOS>": 1}
    for tok, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        if cnt >= MIN_FREQ:
            vocab[tok] = len(vocab)

    print(f"   통합 어휘집: {len(vocab)}개")
    return vocab

# ── 3. GRU 모델 ───────────────────────────────────────────────
class CWMGRU(nn.Module):
    def __init__(self, vocab_size, dim=128, hidden=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.identity   = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru        = nn.GRU(dim, hidden, num_layers,
                                 batch_first=True, dropout=0.2)
        self.output     = nn.Linear(hidden, vocab_size)
        self.dropout    = nn.Dropout(0.2)
        nn.init.uniform_(self.identity.weight, -0.05, 0.05)

    def forward(self, x, hidden=None):
        emb      = self.dropout(self.identity(x))
        out, hid = self.gru(emb, hidden)
        logits   = self.output(out[:, -1, :])
        return logits, hid

    def get_identity(self, token_ids):
        return F.normalize(self.identity(token_ids), dim=-1)

# ── 4. 시퀀스 배치 생성기 ─────────────────────────────────────
def sequence_generator(path, vocab, seq_len, max_samples,
                        batch_size, is_script=False):
    """
    메모리 효율적인 배치 생성
    전체를 메모리에 올리지 않고 스트리밍으로 처리
    """
    pad_id = vocab["<PAD>"]
    eos_id = vocab["<EOS>"]
    batch_inputs  = []
    batch_targets = []
    count = 0

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            if is_script and ":" in line:
                line = line.split(":", 1)[1].strip()

            tokens = tokenize(line)
            ids = [vocab[t] for t in tokens if t in vocab]
            ids.append(eos_id)

            if len(ids) < 2:
                continue

            for i in range(len(ids) - 1):
                start  = max(0, i - seq_len + 1)
                inp    = ids[start:i+1]
                target = ids[i+1]

                pad_len = seq_len - len(inp)
                padded  = [pad_id] * pad_len + inp

                batch_inputs.append(padded)
                batch_targets.append(target)

                if len(batch_inputs) >= batch_size:
                    yield (torch.tensor(batch_inputs, dtype=torch.long),
                           torch.tensor(batch_targets, dtype=torch.long))
                    batch_inputs  = []
                    batch_targets = []

            count += 1

    if batch_inputs:
        yield (torch.tensor(batch_inputs, dtype=torch.long),
               torch.tensor(batch_targets, dtype=torch.long))

# ── 5. Pretraining ────────────────────────────────────────────
def pretrain(model, vocab, epochs=3, lr=0.001):
    """나무위키로 기초 한국어 학습"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss  = 0
        total_batch = 0
        model.train()

        for inputs, targets in sequence_generator(
                NAMUWIKI_PATH, vocab, SEQ_LEN,
                NAMUWIKI_SAMPLES, BATCH_SIZE):

            inputs  = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits, _ = model(inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_batch += 1

            if total_batch % 500 == 0:
                avg = total_loss / total_batch
                print(f"   Epoch {epoch+1} | Batch {total_batch} | Loss: {avg:.4f}")

        avg = total_loss / max(total_batch, 1)
        print(f"   Epoch {epoch+1}/{epochs} 완료 | 평균 Loss: {avg:.4f}")

# ── 6. 파인튜닝 (스크립트로) ──────────────────────────────────
def finetune(model, vocab, epochs=20, lr=0.0005):
    """대화 스크립트로 파인튜닝"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss  = 0
        total_batch = 0
        model.train()

        for inputs, targets in sequence_generator(
                SCRIPT_PATH, vocab, SEQ_LEN,
                999999, 256, is_script=True):

            inputs  = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits, _ = model(inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_batch += 1

        avg = total_loss / max(total_batch, 1)
        if (epoch + 1) % 5 == 0:
            print(f"   파인튜닝 Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")

# ── 7. 저장 ───────────────────────────────────────────────────
def save(model, vocab, path):
    torch.save({
        "identity":     model.identity.weight.detach().cpu(),
        "gru_state":    {k: v.cpu() for k, v in model.gru.state_dict().items()},
        "output_state": {k: v.cpu() for k, v in model.output.state_dict().items()},
        "vocab":        vocab,
        "dim":          DIM,
        "hidden":       HIDDEN,
        "num_layers":   NUM_LAYERS,
        "seq_len":      SEQ_LEN,
    }, path)
    # cwm_anchors.pt 에도 동일하게 저장 (generator.py가 읽을 수 있도록)
    torch.save({
        "identity":     model.identity.weight.detach().cpu(),
        "gru_state":    {k: v.cpu() for k, v in model.gru.state_dict().items()},
        "output_state": {k: v.cpu() for k, v in model.output.state_dict().items()},
        "vocab":        vocab,
        "dim":          DIM,
        "hidden":       HIDDEN,
        "num_layers":   NUM_LAYERS,
        "seq_len":      SEQ_LEN,
    }, "./cwm_anchors.pt")
    print(f"   저장: {path}")
    print(f"   저장: ./cwm_anchors.pt")

# ── 8. 검증 ───────────────────────────────────────────────────
def verify(model, vocab):
    id2tok = {v: k for k, v in vocab.items()}
    model.eval()

    test_contexts = [
        ["나는", "음악"],
        ["오늘", "기분이"],
        ["어떤", "음악"],
        ["만나서"],
    ]

    print("\n   GRU 다음 토큰 예측:")
    pad_id = vocab["<PAD>"]

    with torch.no_grad():
        for ctx_toks in test_contexts:
            ctx_ids = [vocab[t] for t in ctx_toks if t in vocab]
            if not ctx_ids:
                continue
            padded = [pad_id] * (SEQ_LEN - len(ctx_ids)) + ctx_ids
            x = torch.tensor([padded], dtype=torch.long, device=DEVICE)
            logits, _ = model(x)
            logits = logits.squeeze(0)

            top5 = logits.topk(5).indices.tolist()
            preds = [id2tok.get(i, "?") for i in top5
                     if id2tok.get(i, "?") not in ["<PAD>", "<EOS>"]][:3]

            ctx_str  = " ".join(ctx_toks)
            pred_str = ", ".join(preds)
            print(f"   '{ctx_str}' 다음 → {pred_str}")

# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("CWM Pretraining")
    print("나무위키 → 기초 학습 → 스크립트 파인튜닝")
    print("=" * 50)
    print(f"\n디바이스: {DEVICE}")

    # 1. 어휘집
    print("\n1. 통합 어휘집 구축")
    vocab = build_unified_vocab()

    # 어휘집 저장 (tokenizer 결과 교체)
    vocab_data = {
        "vocabulary": [{"token": tok, "freq": 999}
                       for tok in vocab if tok not in ["<PAD>", "<EOS>"]]
    }
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"   어휘집 저장: {VOCAB_PATH}")

    # 2. 모델
    print("\n2. 모델 초기화")
    model = CWMGRU(len(vocab), DIM, HIDDEN, NUM_LAYERS).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"   파라미터: {total:,}개")
    print(f"   어휘집: {len(vocab)}개")

    # 3. Pretraining (나무위키)
    print(f"\n3. Pretraining 시작 ({NAMUWIKI_SAMPLES}개 문장, {EPOCHS} epochs)")
    print(f"   (시간이 걸릴 수 있어 - RTX 3080 기준 20~40분)")
    pretrain(model, vocab, epochs=EPOCHS, lr=LR)

    # 4. 파인튜닝 (스크립트)
    print("\n4. 스크립트 파인튜닝 시작")
    finetune(model, vocab, epochs=20, lr=0.0005)

    # 5. 검증
    print("\n5. 검증")
    verify(model, vocab)

    # 6. 저장
    print("\n6. 저장")
    model = model.cpu()
    save(model, vocab, PRETRAIN_PATH)

    print("\n완료! 바로 대화 테스트: python generator.py")

if __name__ == "__main__":
    main()