# CWM 앵커 시스템 v4
# Identity Vector + GRU 기반 문맥 언어 모델

import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

SCRIPT_PATH  = "./script.txt"
TOKENS_PATH  = "./cwm_tokens.json"
ANCHOR_PATH  = "./cwm_anchors.pt"
DIM          = 128    # 벡터 차원 증가
HIDDEN       = 256    # GRU 히든 크기
NUM_LAYERS   = 2      # GRU 레이어 수
SEQ_LEN      = 8      # 학습 시퀀스 길이
EPOCHS       = 2000
LR           = 0.001
BATCH_SIZE   = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. 데이터 로드 ────────────────────────────────────────────
def load_data():
    with open(TOKENS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    vocab = {}
    for item in data["vocabulary"]:
        tok, freq = item["token"], item["freq"]
        if freq >= 2:
            vocab[tok] = len(vocab)
    # 특수 토큰 추가
    vocab["<PAD>"] = len(vocab)
    vocab["<EOS>"] = len(vocab)
    print(f"   어휘집 크기: {len(vocab)}개")
    return vocab

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

def pre_tokenize(line):
    pattern = re.compile(r'[가-힣a-zA-Z0-9]+|[^가-힣a-zA-Z0-9\s]')
    tokens  = pattern.findall(line)
    return [t for t in tokens if re.match(r'^[가-힣a-zA-Z0-9]+$', t)]

# ── 2. 학습 데이터 생성 ───────────────────────────────────────
def build_sequences(lines, vocab, seq_len=8):
    """
    GRU 학습용 시퀀스 생성
    [t1, t2, t3, ...] → 다음 토큰 예측
    문맥을 seq_len 만큼 기억하며 학습
    """
    sequences = []
    eos_id    = vocab["<EOS>"]

    for line in lines:
        tokens = pre_tokenize(line)
        ids = [vocab[t] for t in tokens
               if t in vocab and re.match(r'^[가-힣a-zA-Z0-9]+$', t)]
        ids.append(eos_id)  # 문장 끝 표시

        if len(ids) < 2:
            continue

        # 슬라이딩 윈도우로 시퀀스 생성
        for i in range(len(ids) - 1):
            start  = max(0, i - seq_len + 1)
            input_ = ids[start:i+1]
            target = ids[i+1]
            sequences.append((input_, target))

    return sequences

def collate_sequences(batch, pad_id, seq_len):
    """배치 패딩"""
    inputs, targets = zip(*batch)
    # 왼쪽 패딩 (최근 문맥이 오른쪽에 오도록)
    padded = []
    for inp in inputs:
        pad_len = seq_len - len(inp)
        padded.append([pad_id] * pad_len + list(inp))
    return (torch.tensor(padded, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))

# ── 3. CWM 앵커 모델 v4 ───────────────────────────────────────
class AnchorModelV4(nn.Module):
    """
    Identity Embedding + GRU 언어 모델

    Identity Embedding: 각 토큰의 의미 위치 (CWM 곡선 위의 점)
    GRU:               문맥을 기억하며 다음 토큰 예측

    GRU가 하는 일:
    "나는 음악을 좋아..." 라는 문맥을 기억하고
    다음에 올 자연스러운 단어를 예측
    """
    def __init__(self, vocab_size, dim=128, hidden=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim        = dim

        # Identity Embedding (CWM 곡선 위의 위치)
        self.identity = nn.Embedding(vocab_size, dim, padding_idx=vocab_size-2)

        # GRU 언어 모델 (문맥 기억)
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        # 출력 레이어
        self.output = nn.Linear(hidden, vocab_size)
        self.dropout = nn.Dropout(0.3)

        # 초기화
        nn.init.uniform_(self.identity.weight, -0.1, 0.1)

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) 토큰 인덱스
        반환: (batch, vocab_size) 다음 토큰 확률
        """
        # 토큰 → 임베딩
        emb = self.dropout(self.identity(x))  # (batch, seq, dim)

        # GRU로 문맥 처리
        out, hidden = self.gru(emb, hidden)   # (batch, seq, hidden)

        # 마지막 타임스텝의 출력으로 다음 토큰 예측
        last_out = out[:, -1, :]              # (batch, hidden)
        logits   = self.output(last_out)      # (batch, vocab_size)

        return logits, hidden

    def get_identity(self, token_ids):
        """Identity Vector 반환 (유사도 계산용)"""
        return F.normalize(self.identity(token_ids), dim=-1)

    def predict_next(self, context_ids, temperature=0.8, top_k=10):
        """
        문맥 → 다음 토큰 예측
        context_ids: 지금까지의 토큰 ID 리스트
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)
            logits, _ = self.forward(x)
            logits = logits.squeeze(0)

            # Top-K 샘플링
            top_k_logits, top_k_ids = logits.topk(top_k)
            probs = F.softmax(top_k_logits / temperature, dim=-1)
            idx   = torch.multinomial(probs, 1).item()
            return top_k_ids[idx].item()

# ── 4. Identity 학습 (맥락 유사도) ───────────────────────────
def train_identity(model, lines, vocab, epochs=100, lr=0.01):
    """
    CWM Identity Vector 학습
    같은 맥락 토큰 → 가까운 벡터
    """
    optimizer = torch.optim.Adam([model.identity.weight], lr=lr)

    # 맥락 쌍 생성
    context_pairs = []
    for line in lines:
        tokens = pre_tokenize(line)
        ids = [vocab[t] for t in tokens
               if t in vocab and re.match(r'^[가-힣a-zA-Z0-9]+$', t)]
        for i, center in enumerate(ids):
            for j in range(max(0, i-3), min(len(ids), i+4)):
                if i != j:
                    context_pairs.append((center, ids[j]))

    pairs = torch.tensor(context_pairs, dtype=torch.long, device=DEVICE)
    vocab_size = model.vocab_size

    print(f"   맥락 쌍: {len(pairs)}개")

    for epoch in range(epochs):
        idx   = torch.randperm(len(pairs), device=DEVICE)[:BATCH_SIZE]
        batch = pairs[idx]

        center_vecs  = model.get_identity(batch[:, 0])
        context_vecs = model.get_identity(batch[:, 1])
        pos_sim  = F.cosine_similarity(center_vecs, context_vecs)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_sim * 5, torch.ones_like(pos_sim))

        neg_ids  = torch.randint(0, vocab_size,
                                 (BATCH_SIZE * 5,), device=DEVICE)
        neg_vecs = model.get_identity(neg_ids).view(BATCH_SIZE, 5, -1)
        neg_sim  = F.cosine_similarity(
            center_vecs.unsqueeze(1).expand_as(neg_vecs), neg_vecs, dim=-1)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_sim * 5, torch.zeros_like(neg_sim))

        loss = pos_loss + neg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"   Identity Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")

# ── 5. GRU 학습 (문맥 언어 모델) ─────────────────────────────
def train_gru(model, sequences, vocab, epochs=400, lr=0.001):
    """
    GRU 언어 모델 학습
    시퀀스 → 다음 토큰 예측
    """
    optimizer = torch.optim.Adam(
        list(model.gru.parameters()) + list(model.output.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5)

    pad_id = vocab["<PAD>"]
    print(f"   시퀀스: {len(sequences)}개")

    for epoch in range(epochs):
        # 랜덤 배치
        batch_idx = torch.randperm(len(sequences))[:BATCH_SIZE].tolist()
        batch     = [sequences[i] for i in batch_idx]
        inputs, targets = collate_sequences(batch, pad_id, SEQ_LEN)
        inputs  = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, _ = model(inputs)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            scheduler.step()
            print(f"   GRU Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

# ── 6. 검증 ───────────────────────────────────────────────────
def verify(model, vocab):
    id2tok = {v: k for k, v in vocab.items()}
    model.eval()

    # Identity 유사도
    print("\n   Identity 유사도:")
    with torch.no_grad():
        all_vecs = model.get_identity(torch.arange(len(vocab), device=DEVICE))
        for tok in ["좋아", "나는", "음악", "친구"]:
            if tok not in vocab:
                continue
            tid     = vocab[tok]
            vec     = model.get_identity(torch.tensor([tid], device=DEVICE))
            sims    = F.cosine_similarity(vec, all_vecs)
            top5    = sims.topk(6).indices.tolist()
            similar = [(id2tok[i], sims[i].item())
                       for i in top5 if id2tok[i] != tok][:3]
            sim_str = ", ".join([f"{t}({s:.2f})" for t, s in similar])
            print(f"   '{tok}' → {sim_str}")

    # GRU 다음 토큰 예측
    print("\n   GRU 다음 토큰 예측:")
    test_contexts = [
        ["나는", "음악"],
        ["오늘", "기분이"],
        ["나도", "좋아"],
        ["어떤", "음악"],
    ]
    for ctx_toks in test_contexts:
        ctx_ids = [vocab[t] for t in ctx_toks if t in vocab]
        if not ctx_ids:
            continue
        # 패딩
        pad_id  = vocab["<PAD>"]
        padded  = [pad_id] * (SEQ_LEN - len(ctx_ids)) + ctx_ids

        preds = []
        for _ in range(5):
            next_id  = model.predict_next(padded, temperature=0.8)
            next_tok = id2tok.get(next_id, "?")
            if next_tok not in ["<PAD>", "<EOS>"]:
                preds.append(next_tok)

        ctx_str  = " ".join(ctx_toks)
        pred_str = ", ".join(set(preds))
        print(f"   '{ctx_str}' 다음 → {pred_str}")

# ── 7. 저장 ───────────────────────────────────────────────────
def save(model, vocab, path):
    torch.save({
        "identity":   model.identity.weight.detach().cpu(),
        "gru_state":  {k: v.cpu() for k, v in model.gru.state_dict().items()},
        "output_state": {k: v.cpu() for k, v in model.output.state_dict().items()},
        "vocab":      vocab,
        "dim":        DIM,
        "hidden":     HIDDEN,
        "num_layers": NUM_LAYERS,
        "seq_len":    SEQ_LEN,
    }, path)
    print(f"   저장: {path}")

# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("CWM 앵커 시스템 v4")
    print("Identity + GRU 문맥 언어 모델")
    print("=" * 50)
    print(f"\n디바이스: {DEVICE}")

    print("\n1. 데이터 로드")
    vocab = load_data()
    lines = load_script(SCRIPT_PATH)
    print(f"   대사: {len(lines)}개")

    print("\n2. 시퀀스 생성")
    sequences = build_sequences(lines, vocab, SEQ_LEN)
    print(f"   시퀀스: {len(sequences)}개")

    print("\n3. 모델 초기화")
    model = AnchorModelV4(len(vocab), DIM, HIDDEN, NUM_LAYERS).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"   파라미터: {total:,}개")

    print("\n4. Identity Vector 학습")
    train_identity(model, lines, vocab, epochs=100, lr=0.01)

    print("\n5. GRU 언어 모델 학습")
    train_gru(model, sequences, vocab, epochs=EPOCHS, lr=LR)

    print("\n6. 검증")
    verify(model, vocab)

    print("\n7. 저장")
    model = model.cpu()
    save(model, vocab, ANCHOR_PATH)

    print("\n완료! 다음 단계: python generator.py")

if __name__ == "__main__":
    main()