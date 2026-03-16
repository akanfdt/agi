# CWM 국소 탐색 (Local Search)
# 입력 문장 → 앵커 위치 → 주변 탐색 → 다음 단어 예측

import re
import json
import torch
import torch.nn.functional as F

TOKENS_PATH = "./cwm_tokens.json"
ANCHOR_PATH = "./cwm_anchors.pt"
SCRIPT_PATH = "./script.txt"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K       = 5    # 탐색할 주변 단어 수
SPREAD      = 3    # 간접 경로 단계 수
DECAY       = 0.5  # 거리마다 신호 감쇠 (절반)

# ── 1. 앵커 로드 ──────────────────────────────────────────────
def load_anchors():
    data = torch.load(ANCHOR_PATH, map_location=DEVICE, weights_only=True)
    embeddings = data["embeddings"].to(DEVICE)  # (vocab_size, dim)
    vocab      = data["vocab"]                  # {token: id}
    id2tok     = {v: k for k, v in vocab.items()}
    print(f"   어휘집: {len(vocab)}개 토큰")
    print(f"   벡터 차원: {embeddings.shape[1]}D")
    return embeddings, vocab, id2tok

# ── 2. 문장 토큰화 ────────────────────────────────────────────
def tokenize(text):
    pattern = re.compile(r'[가-힣a-zA-Z0-9]+|[^가-힣a-zA-Z0-9\s]')
    tokens  = pattern.findall(text)
    # 한글/영문만 (문장부호 제외)
    return [t for t in tokens if re.match(r'^[가-힣a-zA-Z0-9]+$', t)]

# ── 3. 앵커 포인트 찾기 ───────────────────────────────────────
def find_anchors(tokens, vocab, embeddings):
    """
    입력 토큰들의 앵커 위치 찾기
    어휘집에 없는 토큰은 가장 가까운 벡터로 근사
    """
    anchor_ids   = []
    anchor_vecs  = []
    anchor_toks  = []

    all_vecs = embeddings  # (vocab_size, dim)

    for tok in tokens:
        if tok in vocab:
            # 어휘집에 있으면 직접 찾기
            tok_id  = vocab[tok]
            tok_vec = embeddings[tok_id]
            anchor_ids.append(tok_id)
            anchor_vecs.append(tok_vec)
            anchor_toks.append(tok)
        else:
            # 어휘집에 없으면 글자 단위로 가장 가까운 토큰 찾기
            # 첫 두 글자로 검색
            prefix = tok[:2] if len(tok) >= 2 else tok
            candidates = [(k, v) for k, v in vocab.items()
                          if k.startswith(prefix)]
            if candidates:
                best_tok, best_id = candidates[0]
                anchor_ids.append(best_id)
                anchor_vecs.append(embeddings[best_id])
                anchor_toks.append(f"{tok}→{best_tok}")

    return anchor_ids, anchor_vecs, anchor_toks

# ── 4. 국소 탐색 ─────────────────────────────────────────────
def local_search(anchor_ids, anchor_vecs, embeddings, id2tok,
                 top_k=5, spread=3, decay=0.5):
    """
    CWM 핵심 탐색 알고리즘:
    1. 앵커 포인트에서 시작
    2. 유사한 벡터들로 퍼져나감
    3. 거리마다 신호 감쇠
    4. 활성화된 영역 = 이 문장의 의미 공간
    """
    if not anchor_vecs:
        return {}

    # 앵커 벡터들의 평균 = 문장의 중심 벡터
    center_vec = torch.stack(anchor_vecs).mean(dim=0, keepdim=True)
    center_vec = F.normalize(center_vec, dim=-1)

    # 활성화 점수 (토큰 ID → 점수)
    activation = {}

    # 앵커 자체는 강도 1.0
    for aid in anchor_ids:
        activation[aid] = 1.0

    # 간접 경로로 퍼져나감
    current_ids      = set(anchor_ids)
    current_strength = 1.0

    for step in range(spread):
        current_strength *= decay  # 거리마다 절반

        if current_strength < 0.01:
            break

        # 현재 활성화된 벡터들의 평균
        current_vecs = embeddings[list(current_ids)]
        current_center = current_vecs.mean(dim=0, keepdim=True)
        current_center = F.normalize(current_center, dim=-1)

        # 가장 유사한 top_k 토큰 찾기
        sims = F.cosine_similarity(current_center, embeddings)
        top_ids = sims.topk(top_k + len(current_ids)).indices.tolist()

        new_ids = set()
        for tid in top_ids:
            if tid not in anchor_ids:  # 앵커 자체 제외
                score = sims[tid].item() * current_strength
                if tid not in activation or activation[tid] < score:
                    activation[tid] = score
                new_ids.add(tid)
                if len(new_ids) >= top_k:
                    break

        current_ids = new_ids

    # 문장 중심 벡터와 최종 유사도로 점수 보정
    all_sims = F.cosine_similarity(center_vec, embeddings).detach().cpu()
    for tid in activation:
        activation[tid] *= (all_sims[tid].item() + 1) / 2

    return activation

# ── 5. 다음 단어 예측 ─────────────────────────────────────────
def predict_next(activation, id2tok, anchor_ids, top_n=5):
    """
    활성화된 영역에서 가장 점수 높은 토큰 선택
    앵커(입력 단어) 자체는 제외
    """
    candidates = [
        (tid, score) for tid, score in activation.items()
        if tid not in anchor_ids
    ]
    candidates.sort(key=lambda x: -x[1])

    return [(id2tok[tid], score) for tid, score in candidates[:top_n]
            if tid in id2tok]

# ── 6. 스크립트에서 비슷한 문장 찾기 ─────────────────────────
def find_similar_context(input_text, lines, vocab, embeddings, top_n=3):
    """
    입력과 비슷한 맥락의 대화를 스크립트에서 찾기
    → 더 자연스러운 다음 문장 예측에 활용
    """
    input_tokens = tokenize(input_text)
    input_ids    = [vocab[t] for t in input_tokens if t in vocab]

    if not input_ids:
        return []

    # 입력 문장 벡터
    input_vecs  = embeddings[input_ids]
    input_center = F.normalize(input_vecs.mean(dim=0, keepdim=True), dim=-1)

    # 스크립트 각 대사와 유사도 계산
    scored_lines = []
    for i, line in enumerate(lines):
        line_tokens = tokenize(line)
        line_ids    = [vocab[t] for t in line_tokens if t in vocab]
        if not line_ids:
            continue
        line_vecs   = embeddings[line_ids]
        line_center = F.normalize(line_vecs.mean(dim=0, keepdim=True), dim=-1)
        sim = F.cosine_similarity(input_center, line_center).item()
        scored_lines.append((sim, i, line))

    scored_lines.sort(key=lambda x: -x[0])
    return scored_lines[:top_n]

def load_script_with_pairs(path):
    """대사 쌍으로 로드 (질문 → 답변)"""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                speaker, text = line.split(":", 1)
                lines.append((speaker.strip(), text.strip()))
    return lines

# ── 7. 대화 흐름 예측 ─────────────────────────────────────────
def predict_response(input_text, script_pairs, vocab, embeddings, id2tok):
    """
    입력 문장과 가장 비슷한 스크립트 문장을 찾고
    그 다음 대사를 후보 응답으로 반환
    """
    input_tokens = tokenize(input_text)
    input_ids    = [vocab[t] for t in input_tokens if t in vocab]

    if not input_ids:
        return []

    input_vecs   = embeddings[input_ids]
    input_center = F.normalize(input_vecs.mean(dim=0, keepdim=True), dim=-1)

    candidates = []
    for i, (speaker, text) in enumerate(script_pairs):
        line_tokens = tokenize(text)
        line_ids    = [vocab[t] for t in line_tokens if t in vocab]
        if not line_ids:
            continue
        line_vecs   = embeddings[line_ids]
        line_center = F.normalize(line_vecs.mean(dim=0, keepdim=True), dim=-1)
        sim = F.cosine_similarity(input_center, line_center).item()

        # 다음 대사가 릴리아 대사면 후보로
        if i + 1 < len(script_pairs):
            next_speaker, next_text = script_pairs[i + 1]
            if "릴리아" in next_speaker:
                candidates.append((sim, text, next_text))

    candidates.sort(key=lambda x: -x[0])
    return candidates[:3]

# ── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("CWM 국소 탐색")
    print("입력 → 앵커 → 탐색 → 예측")
    print("=" * 50)
    print(f"\n디바이스: {DEVICE}")

    # 앵커 로드
    print("\n앵커 로드")
    embeddings, vocab, id2tok = load_anchors()

    # 스크립트 로드
    script_pairs = load_script_with_pairs(SCRIPT_PATH)
    print(f"   스크립트: {len(script_pairs)}개 대사")

    print("\n" + "=" * 50)
    print("테스트 시작")
    print("=" * 50)

    test_inputs = [
        "나 오늘 기분이 좋아",
        "어떤 음악 좋아해",
        "너는 어떤 성격이야",
        "취미가 뭐야",
        "나도 그거 좋아해",
    ]

    for text in test_inputs:
        print(f"\n입력: '{text}'")
        print("-" * 40)

        # 토큰화
        tokens = tokenize(text)
        print(f"토큰: {tokens}")

        # 앵커 찾기
        anchor_ids, anchor_vecs, anchor_toks = find_anchors(
            tokens, vocab, embeddings)
        print(f"앵커: {anchor_toks}")

        # 국소 탐색
        activation = local_search(
            anchor_ids, anchor_vecs, embeddings, id2tok,
            top_k=TOP_K, spread=SPREAD, decay=DECAY
        )

        # 다음 단어 예측
        predictions = predict_next(activation, id2tok, anchor_ids, top_n=5)
        print(f"활성화 단어:")
        for tok, score in predictions:
            bar = "█" * max(1, int(score * 30))
            print(f"  {tok:10s} {bar} ({score:.3f})")

        # 스크립트에서 비슷한 응답 찾기
        responses = predict_response(
            text, script_pairs, vocab, embeddings, id2tok)
        if responses:
            print(f"비슷한 상황의 릴리아 응답:")
            for sim, matched, response in responses[:2]:
                print(f"  [{sim:.3f}] Q: {matched}")
                print(f"          A: {response}")

    # 대화 모드
    print("\n" + "=" * 50)
    print("대화 모드 (종료: q)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n나: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() == 'q':
            break

        tokens = tokenize(user_input)
        if not tokens:
            print("릴리아: (인식된 단어가 없어)")
            continue

        anchor_ids, anchor_vecs, anchor_toks = find_anchors(
            tokens, vocab, embeddings)

        activation = local_search(
            anchor_ids, anchor_vecs, embeddings, id2tok,
            top_k=TOP_K, spread=SPREAD, decay=DECAY
        )

        responses = predict_response(
            user_input, script_pairs, vocab, embeddings, id2tok)

        if responses:
            best_sim, matched, response = responses[0]
            print(f"릴리아: {response}  [{best_sim:.2f}]")
        else:
            predictions = predict_next(activation, id2tok, anchor_ids, top_n=3)
            if predictions:
                words = " ".join([t for t, _ in predictions])
                print(f"릴리아: (관련 단어: {words})")
            else:
                print("릴리아: (모르겠어...)")

if __name__ == "__main__":
    main()