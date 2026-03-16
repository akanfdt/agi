import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── 설정 및 경로 ──────────────────────────────────────────────
ANCHOR_PATH = "./cwm_anchors.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LR_ONLINE   = 1e-4  # 학습률을 높여 교정 효과 강화 (설계 10장 반영)
MAX_LEN     = 15

# ── 1. 앵커 및 모델 구조 ─────────────────────────────────────
class AnchorPoint:
    def __init__(self, token_id, token_text, vector):
        self.id = token_id
        self.text = token_text
        self.vector = vector

class AnchorModelV4(nn.Module):
    def __init__(self, vocab_size, dim=128, hidden=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.identity   = nn.Embedding(vocab_size, dim, padding_idx=vocab_size-2)
        self.gru        = nn.GRU(dim, hidden, num_layers, batch_first=True)
        self.output     = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.identity(x)
        out, hid = self.gru(emb, hidden)
        logits = self.output(out[:, -1, :])
        return logits, hid

# ── 2. 유틸리티 함수 (에러 해결 핵심) ──────────────────────────
def tokenize(text):
    pattern = re.compile(r'[가-힣a-zA-Z0-9]+')
    return pattern.findall(text)

def ids_to_text(token_ids, id2tok):
    stop_tokens = {"<PAD>", "<EOS>"}
    tokens = [id2tok.get(tid, "") for tid in token_ids if id2tok.get(tid, "") not in stop_tokens]
    return " ".join([t for t in tokens if len(t) >= 1])

def load_all():
    """모델과 데이터를 로드하고 변수들을 반환합니다."""
    # weights_only=False는 사용자 정의 클래스(AnchorPoint 등) 로드를 위해 필요할 수 있음
    data = torch.load(ANCHOR_PATH, map_location=DEVICE, weights_only=False)
    vocab = data["vocab"]
    id2tok = {v: k for k, v in vocab.items()}
    
    model = AnchorModelV4(len(vocab), data['dim'], data['hidden'], data['num_layers'])
    model.identity.weight.data.copy_(data["identity"])
    model.gru.load_state_dict(data["gru_state"])
    model.output.load_state_dict(data["output_state"])
    model = model.to(DEVICE)

    anchor_points = {}
    normalized_weights = F.normalize(model.identity.weight.data, dim=-1)
    for tok, idx in vocab.items():
        anchor_points[idx] = AnchorPoint(idx, tok, normalized_weights[idx])
        
    return model, vocab, id2tok, anchor_points

# ── 3. CWM 실시간 진화 엔진 (저장 로직 포함) ───────────────────
class CWMLiveEngine:
    def __init__(self, model, vocab, id2tok, anchor_points):
        self.model = model
        self.vocab = vocab
        self.id2tok = id2tok
        self.anchors = anchor_points
        self.optimizer = optim.Adam(model.parameters(), lr=LR_ONLINE)

    def save_weights(self):
        """현재 가중치(곡선)를 파일로 영구 저장합니다."""
        data = torch.load(ANCHOR_PATH, map_location=DEVICE, weights_only=False)
        data["identity"] = self.model.identity.weight.data.cpu()
        data["gru_state"] = self.model.gru.state_dict()
        data["output_state"] = self.model.output.state_dict()
        torch.save(data, ANCHOR_PATH)
        print(f"\n[CWM 시스템] 곡선의 궤적이 {ANCHOR_PATH}에 영구 기록되었습니다.")

    def train_step(self, input_ids, target_ids):
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(input_ids)
        loss = F.cross_entropy(logits, target_ids.view(-1))
        loss.backward()
        self.optimizer.step()
        
        # 가중치 변화 동기화
        with torch.no_grad():
            weights = F.normalize(self.model.identity.weight.data, dim=-1)
            for idx in self.anchors:
                self.anchors[idx].vector = weights[idx]
        return loss.item()

    def generate(self, input_ids, max_len=MAX_LEN):
        self.model.eval()
        generated = input_ids
        context_path = [self.anchors[idx] for idx in input_ids[0].tolist() if idx in self.anchors]

        for _ in range(max_len):
            with torch.no_grad():
                logits, _ = self.model(generated)
                if context_path:
                    # 곡선의 겹침(Filter) 적용 - 인력 강화
                    path_vec = torch.mean(torch.stack([a.vector for a in context_path[-5:]]), dim=0)
                    context_sim = F.cosine_similarity(self.model.identity.weight, path_vec.unsqueeze(0))
                    logits += context_sim * 1.0 
                
                probs = F.softmax(logits / 0.7, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_tok], dim=1)
                if next_tok.item() in self.anchors:
                    context_path.append(self.anchors[next_tok.item()])
                if next_tok.item() == 0: break
        return generated

# ── 4. 메인 실행 루프 (모든 변수 선언 완료) ────────────────────
def main():
    print("="*50 + "\nCWM v0.4 진화 엔진 가동 중\n" + "="*50)
    
    # [중요] 여기서 모든 변수(model, vocab 등)가 정의됩니다.
    model, vocab, id2tok, anchor_points = load_all()
    engine = CWMLiveEngine(model, vocab, id2tok, anchor_points)
    
    prev_input_tensor = None

    while True:
        try:
            user_input = input("\n나: ").strip()
        except EOFError: break
        
        if user_input.lower() == 'q':
            engine.save_weights() # 종료 시 파일에 기록
            break
        
        tokens = tokenize(user_input)
        ids = [vocab[t] for t in tokens if t in vocab]
        if not ids: continue
        
        curr_tensor = torch.tensor([ids], device=DEVICE)

        # 실시간 학습: 이전 입력과 현재 입력의 인과관계를 곡선에 반영
        if prev_input_tensor is not None:
            loss = engine.train_step(prev_input_tensor, curr_tensor[:, 0])
            print(f"(곡선 교정 중... Loss: {loss:.4f})")

        # 결과 생성
        output_ids = engine.generate(curr_tensor)
        response_ids = output_ids[0][len(ids):].tolist()
        print(f"릴리아: {ids_to_text(response_ids, id2tok)}")
        
        prev_input_tensor = curr_tensor

if __name__ == "__main__":
    main()