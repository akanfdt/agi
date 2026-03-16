# CWM 문장 생성기 v2
# GRU 기반 문맥 언어 모델로 자연스러운 문장 생성

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

ANCHOR_PATH = "./cwm_anchors.pt"
SCRIPT_PATH = "./script.txt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN  = 15
TEMP     = 0.7
TOP_K    = 10

# ── 1. 모델 정의 (anchor.py와 동일) ──────────────────────────
class AnchorModelV4(nn.Module):
    def __init__(self, vocab_size, dim=128, hidden=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim        = dim
        self.identity   = nn.Embedding(vocab_size, dim, padding_idx=vocab_size-2)
        self.gru        = nn.GRU(dim, hidden, num_layers,
                                 batch_first=True, dropout=0.3)
        self.output     = nn.Linear(hidden, vocab_size)
        self.dropout    = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        emb      = self.dropout(self.identity(x))
        out, hid = self.gru(emb, hidden)
        logits   = self.output(out[:, -1, :])
        return logits, hid

    def get_identity(self, token_ids):
        return F.normalize(self.identity(token_ids), dim=-1)

    def predict_next(self, context_ids, temperature=0.7, top_k=10):
        self.eval()
        with torch.no_grad():
            x = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)
            logits, _ = self.forward(x)
            logits     = logits.squeeze(0)
            top_logits, top_ids = logits.topk(top_k)
            probs = F.softmax(top_logits / temperature, dim=-1)
            idx   = torch.multinomial(probs, 1).item()
            return top_ids[idx].item()

# ── 2. 앵커 로드 ──────────────────────────────────────────────
def load_model():
    data     = torch.load(ANCHOR_PATH, map_location="cpu", weights_only=False)
    vocab    = data["vocab"]
    id2tok   = {v: k for k, v in vocab.items()}
    dim      = data.get("dim", 128)
    hidden   = data.get("hidden", 256)
    n_layers = data.get("num_layers", 2)
    seq_len  = data.get("seq_len", 8)

    model = AnchorModelV4(len(vocab), dim, hidden, n_layers)
    model.identity.weight.data.copy_(data["identity"])
    model.gru.load_state_dict(data["gru_state"])
    model.output.load_state_dict(data["output_state"])
    model = model.to(DEVICE)
    model.eval()

    print(f"   어휘집: {len(vocab)}개  dim: {dim}  hidden: {hidden}")
    return model, vocab, id2tok, seq_len

# ── 3. 토큰화 ─────────────────────────────────────────────────
def tokenize(text):
    pattern = re.compile(r'[가-힣a-zA-Z0-9]+')
    return pattern.findall(text)

# ── 4. 문장 생성 ─────────────────────────────────────────────
def generate(model, vocab, id2tok, seed_tokens, seq_len,
             max_len=15, temp=0.7):
    """
    GRU로 자연스러운 문장 생성
    seed_tokens: 입력 문장의 토큰들 (문맥 시작점)
    """
    pad_id = vocab.get("<PAD>", 0)
    eos_id = vocab.get("<EOS>", 1)

    # 시드 토큰 → ID
    seed_ids = [vocab[t] for t in seed_tokens if t in vocab]
    if not seed_ids:
        return []

    # 응답 시작 토큰 선택
    # 시드 문맥으로 첫 토큰 예측
    context = [pad_id] * (seq_len - len(seed_ids)) + seed_ids
    context = context[-seq_len:]

    generated = []
    current_context = list(context)

    # 응답 시작 단어 후보
    start_bonus = {
        "나는": 0.5, "나도": 0.4, "응": 0.3, "맞아": 0.3,
        "어": 0.2, "음": 0.2, "진짜": 0.2, "헤헤": 0.1,
        "사실": 0.2, "좋아": 0.3, "재밌": 0.2
    }

    # 첫 토큰은 보너스 적용해서 선택
    with torch.no_grad():
        x = torch.tensor([current_context], dtype=torch.long, device=DEVICE)
        logits, _ = model(x)
        logits = logits.squeeze(0).cpu()

        # 응답 시작 보너스
        for tok, bonus in start_bonus.items():
            if tok in vocab:
                logits[vocab[tok]] += bonus

        # 한글 토큰만
        mask = torch.full((len(vocab),), -1e9)
        for tid, tok in id2tok.items():
            if re.match(r'^[가-힣]+$', tok) and tok not in ["<PAD>", "<EOS>"]:
                mask[tid] = 0
        logits = logits + mask

        probs    = F.softmax(logits / temp, dim=-1)
        first_id = torch.multinomial(probs, 1).item()

    generated.append(first_id)
    current_context = (current_context + [first_id])[-seq_len:]

    # 이후 토큰 생성
    for step in range(max_len - 1):
        with torch.no_grad():
            x = torch.tensor([current_context], dtype=torch.long, device=DEVICE)
            logits, _ = model(x)
            logits = logits.squeeze(0).cpu()

        # 반복 패널티
        for i, prev_id in enumerate(reversed(generated)):
            penalty = 4.0 / (i + 1)
            logits[prev_id] -= penalty

        # EOS 처리
        if len(generated) >= 5:
            logits[eos_id] += 1.0

        # 한글만
        mask = torch.full((len(vocab),), -1e9)
        for tid, tok in id2tok.items():
            if re.match(r'^[가-힣]+$', tok) and tok not in ["<PAD>"]:
                mask[tid] = 0
        logits = logits + mask

        probs   = F.softmax(logits / temp, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id == eos_id:
            break

        generated.append(next_id)
        current_context = (current_context + [next_id])[-seq_len:]

        # 자연스러운 종료 (6~12 토큰 사이)
        if len(generated) >= 6:
            tok = id2tok.get(next_id, "")
            if tok in ["야", "어", "해", "지", "다", "응", "네",
                       "겠어", "있어", "같아", "좋아", "맞아"]:
                break

    return generated

def ids_to_text(token_ids, id2tok):
    stop_tokens = {"멜", "멜은", "멜이", "멜이야", "<PAD>", "<EOS>"}
    tokens = [id2tok.get(tid, "") for tid in token_ids
              if id2tok.get(tid, "") not in stop_tokens]
    tokens = [t for t in tokens if len(t) >= 1]
    return " ".join(tokens)

# ── 5. 전체 응답 파이프라인 ───────────────────────────────────
def respond(user_input, model, vocab, id2tok, seq_len, n_tries=3):
    tokens = tokenize(user_input)
    if not tokens:
        return "응?"

    # 여러 번 생성해서 가장 자연스러운 것 선택
    candidates = []
    for _ in range(n_tries):
        ids  = generate(model, vocab, id2tok, tokens, seq_len, MAX_LEN, TEMP)
        text = ids_to_text(ids, id2tok)
        if len(text) >= 4:
            candidates.append(text)

    if not candidates:
        return "음..."

    # 중간 길이의 응답 선택 (너무 짧거나 너무 길지 않게)
    candidates.sort(key=lambda x: abs(len(x) - 15))
    return candidates[0]

# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("CWM 문장 생성기 v2")
    print("GRU 기반 문맥 언어 모델")
    print("=" * 50)
    print(f"\n디바이스: {DEVICE}")

    print("\n모델 로드")
    model, vocab, id2tok, seq_len = load_model()

    # 테스트
    print("\n" + "=" * 50)
    print("생성 테스트")
    print("=" * 50)

    test_inputs = [
        "어떤 음악 좋아해",
        "너는 어떤 성격이야",
        "취미가 뭐야",
        "오늘 기분이 어때",
        "나도 그거 좋아해",
    ]

    for text in test_inputs:
        print(f"\n입력: '{text}'")
        for i in range(3):
            response = respond(text, model, vocab, id2tok, seq_len)
            print(f"  생성 {i+1}: {response}")

    # 대화 모드
    print("\n" + "=" * 50)
    print("대화 모드 (종료: q)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n나: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() == "q":
            break
        response = respond(user_input, model, vocab, id2tok, seq_len)
        print(f"릴리아: {response}")

if __name__ == "__main__":
    main()