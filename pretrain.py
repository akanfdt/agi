# CWM Pretraining v2
# 기존 가중치 이어받기 + 어휘집 고정
# 절대 새로 초기화하지 않음

import re
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

NAMUWIKI_PATH = "./namuwiki_pretrain.txt"
SCRIPT_PATH   = "./script.txt"
VOCAB_PATH    = "./cwm_tokens.json"
ANCHOR_PATH   = "./cwm_anchors.pt"

NAMUWIKI_SAMPLES = 500000
MIN_FREQ         = 5

DIM        = 128
HIDDEN     = 256
NUM_LAYERS = 2
SEQ_LEN    = 10
EPOCHS     = 3
LR         = 0.001
BATCH_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 토크나이저 ────────────────────────────────────────────────
def tokenize(line):
    pattern = re.compile(r'[가-힣a-zA-Z0-9]+|[^가-힣a-zA-Z0-9\s]')
    tokens  = pattern.findall(line)
    return [t for t in tokens if re.match(r'^[가-힣]+$', t)]

# ── 어휘집 ────────────────────────────────────────────────────
def load_or_build_vocab():
    """
    어휘집이 있으면 로드 (기존 유지)
    없으면 새로 구축
    핵심: 한 번 만든 어휘집은 절대 삭제하지 않음
    새 단어는 추가만 함
    """
    existing_vocab = {}

    # 기존 어휘집 로드
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for item in data["vocabulary"]:
            existing_vocab[item["token"]] = item["freq"]
        print(f"   기존 어휘집 로드: {len(existing_vocab)}개")

    # 기존 앵커에서 어휘집 로드 (우선순위)
    if os.path.exists(ANCHOR_PATH):
        saved = torch.load(ANCHOR_PATH, map_location="cpu", weights_only=False)
        if "vocab" in saved:
            existing_vocab_ids = saved["vocab"]
            print(f"   기존 앵커 어휘집: {len(existing_vocab_ids)}개 (이것 사용)")
            return existing_vocab_ids

    if existing_vocab:
        # vocab을 {token: id} 형태로 변환
        vocab = {"<PAD>": 0, "<EOS>": 1}
        for tok in existing_vocab:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        print(f"   기존 어휘집 사용: {len(vocab)}개")
        return vocab

    # 새로 구축
    print("   새 어휘집 구축 중...")
    freq = defaultdict(int)

    with open(NAMUWIKI_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NAMUWIKI_SAMPLES:
                break
            for tok in tokenize(line.strip()):
                freq[tok] += 1

    with open(SCRIPT_PATH, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            for tok in tokenize(line):
                freq[tok] += MIN_FREQ + 1

    vocab = {"<PAD>": 0, "<EOS>": 1}
    for tok, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        if cnt >= MIN_FREQ:
            vocab[tok] = len(vocab)

    print(f"   새 어휘집: {len(vocab)}개")
    return vocab

def save_vocab(vocab):
    data = {
        "vocabulary": [{"token": tok, "freq": 999}
                       for tok in vocab if tok not in ["<PAD>", "<EOS>"]]
    }
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── GRU 모델 ──────────────────────────────────────────────────
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

# ── 모델 로드 또는 생성 ───────────────────────────────────────
def load_or_create_model(vocab):
    """
    기존 가중치 있으면 이어받기
    없으면 새로 생성
    """
    model = CWMGRU(len(vocab), DIM, HIDDEN, NUM_LAYERS)

    if os.path.exists(ANCHOR_PATH):
        print("   기존 가중치 발견 → 이어받기 시도")
        saved = torch.load(ANCHOR_PATH, map_location="cpu", weights_only=False)

        saved_vocab_size = saved["identity"].shape[0]
        curr_vocab_size  = len(vocab)

        if saved_vocab_size == curr_vocab_size:
            # 어휘집 크기 같으면 그대로 로드
            model.identity.weight.data.copy_(saved["identity"])
            model.gru.load_state_dict(saved["gru_state"])
            model.output.load_state_dict(saved["output_state"])
            print(f"   이어받기 완료 (어휘집 {curr_vocab_size}개)")

        elif saved_vocab_size < curr_vocab_size:
            # 어휘집이 늘었으면 기존 가중치 복사 + 새 단어는 랜덤 초기화
            model.identity.weight.data[:saved_vocab_size].copy_(saved["identity"])
            model.gru.load_state_dict(saved["gru_state"])
            # output 레이어도 부분 복사
            model.output.weight.data[:saved_vocab_size].copy_(
                saved["output_state"]["weight"][:saved_vocab_size])
            model.output.bias.data[:saved_vocab_size].copy_(
                saved["output_state"]["bias"][:saved_vocab_size])
            print(f"   어휘집 확장: {saved_vocab_size} → {curr_vocab_size}")
            print(f"   기존 가중치 보존 + 새 단어 랜덤 초기화")
        else:
            print(f"   어휘집 크기 불일치 ({saved_vocab_size} → {curr_vocab_size})")
            print(f"   새로 시작 (기존과 다른 어휘집)")
    else:
        print("   기존 가중치 없음 → 새로 시작")

    return model.to(DEVICE)

# ── 시퀀스 배치 생성기 ────────────────────────────────────────
def sequence_generator(path, vocab, seq_len, max_samples,
                        batch_size, is_script=False):
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

# ── 저장 ─────────────────────────────────────────────────────
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
    print(f"   저장: {path}")

# ── Pretraining ───────────────────────────────────────────────
def pretrain(model, vocab, epochs=3, lr=0.001):
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

        # epoch마다 중간 저장
        model.cpu()
        save(model, vocab, ANCHOR_PATH)
        model.to(DEVICE)
        print(f"   중간 저장 완료")

# ── 파인튜닝 ─────────────────────────────────────────────────
def finetune(model, vocab, epochs=20, lr=0.0005):
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

    # 파인튜닝 후 저장
    model.cpu()
    save(model, vocab, ANCHOR_PATH)
    model.to(DEVICE)

# ── 검증 ─────────────────────────────────────────────────────
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
            top5  = logits.squeeze(0).topk(5).indices.tolist()
            preds = [id2tok.get(i, "?") for i in top5
                     if id2tok.get(i, "?") not in ["<PAD>", "<EOS>"]][:3]
            print(f"   '{' '.join(ctx_toks)}' 다음 → {', '.join(preds)}")

# ── 메인 ─────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("CWM Pretraining v2")
    print("기존 가중치 이어받기 + 어휘집 고정")
    print("=" * 50)
    print(f"\n디바이스: {DEVICE}")

    print("\n1. 어휘집 로드/구축")
    vocab = load_or_build_vocab()
    save_vocab(vocab)
    print(f"   어휘집 확정: {len(vocab)}개")

    print("\n2. 모델 로드/생성")
    model = load_or_create_model(vocab)
    total = sum(p.numel() for p in model.parameters())
    print(f"   파라미터: {total:,}개")

    print(f"\n3. Pretraining ({NAMUWIKI_SAMPLES}개 문장, {EPOCHS} epochs)")
    pretrain(model, vocab, epochs=EPOCHS, lr=LR)

    print("\n4. 스크립트 파인튜닝")
    finetune(model, vocab, epochs=20, lr=0.0005)

    print("\n5. 검증")
    verify(model, vocab)

    print("\n완료! python generator2.py 로 대화 가능")

if __name__ == "__main__":
    main()