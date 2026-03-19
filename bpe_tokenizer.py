from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple


WORD_PATTERN = re.compile(r"[\uac00-\ud7a3A-Za-z0-9]+|[^\uac00-\ud7a3A-Za-z0-9\s]")
WORD_ONLY = re.compile(r"^[\uac00-\ud7a3A-Za-z0-9]+$")
WORD_START = "▁"


def _get_pairs(tokens: List[str]) -> List[Tuple[str, str]]:
    if len(tokens) < 2:
        return []
    return list(zip(tokens[:-1], tokens[1:]))


def _merge_pair(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


@dataclass
class BPETokenizer:
    vocab: List[str]
    merges: List[Tuple[str, str]]
    ranks: Dict[Tuple[str, str], int]
    unk_token: str = "<unk>"
    vocab_set: set[str] | None = None

    @staticmethod
    def load(path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(item.split(" ")) for item in data.get("merges", [])]
        vocab = data.get("vocab", [])
        ranks = {pair: i for i, pair in enumerate(merges)}
        tok = BPETokenizer(
            vocab=vocab,
            merges=merges,
            ranks=ranks,
            unk_token=data.get("unk_token", "<unk>"),
        )
        tok.vocab_set = set(vocab)
        return tok

    def encode(self, text: str) -> List[str]:
        tokens: List[str] = []
        for raw in WORD_PATTERN.findall(text):
            if not raw.strip():
                continue
            if WORD_ONLY.match(raw):
                tokens.extend(self._encode_word(raw))
            else:
                tokens.append(raw)
        return tokens

    def _encode_word(self, word: str) -> List[str]:
        if self.vocab_set is not None:
            whole = WORD_START + word
            if whole in self.vocab_set:
                return [whole]
        tokens = [WORD_START] + list(word)
        if not self.ranks:
            return tokens
        while True:
            pairs = _get_pairs(tokens)
            if not pairs:
                break
            best_pair = None
            best_rank = None
            for pair in pairs:
                rank = self.ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            tokens = _merge_pair(tokens, best_pair)
        return tokens

    def decode(self, tokens: List[str]) -> str:
        out: List[str] = []
        for tok in tokens:
            if tok == self.unk_token:
                if out:
                    out.append(" ")
                out.append("<?>")
                continue
            if tok.startswith(WORD_START):
                if out:
                    out.append(" ")
                out.append(tok[len(WORD_START) :])
            else:
                out.append(tok)
        text = "".join(out)
        text = re.sub(r"\s+([,.;!?])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text


def train_bpe(
    corpus_lines: Iterable[str],
    vocab_size: int,
    max_word_types: int = 200000,
    min_freq: int = 2,
    log_every_merges: int = 500,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    word_counts: Dict[str, int] = {}
    punct_counts: Dict[str, int] = {}

    for i, line in enumerate(corpus_lines, start=1):
        for raw in WORD_PATTERN.findall(line):
            if not raw.strip():
                continue
            if WORD_ONLY.match(raw):
                word_counts[raw] = word_counts.get(raw, 0) + 1
            else:
                punct_counts[raw] = punct_counts.get(raw, 0) + 1
        if len(word_counts) > max_word_types * 2:
            items = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_word_types]
            word_counts = dict(items)
        if i % 200000 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] lines={i} | word_types={len(word_counts)}")

    word_counts = {word: count for word, count in word_counts.items() if count >= min_freq}

    vocab: Dict[Tuple[str, ...], int] = {}
    symbols: List[str] = ["<unk>", WORD_START]
    symbol_set = set(symbols)

    for word, freq in word_counts.items():
        chars = (WORD_START,) + tuple(word)
        vocab[chars] = freq
        for ch in chars:
            if ch not in symbol_set:
                symbol_set.add(ch)
                symbols.append(ch)

    for punct in punct_counts:
        if punct not in symbol_set:
            symbol_set.add(punct)
            symbols.append(punct)

    merges: List[Tuple[str, str]] = []
    target_merges = max(0, vocab_size - len(symbols))

    for merge_i in range(target_merges):
        pair_counts: Dict[Tuple[str, str], int] = {}
        for word, freq in vocab.items():
            if len(word) < 2:
                continue
            for pair in zip(word[:-1], word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda item: item[1])[0]
        merges.append(best_pair)
        merged_symbol = best_pair[0] + best_pair[1]
        if merged_symbol not in symbol_set:
            symbol_set.add(merged_symbol)
            symbols.append(merged_symbol)
        next_vocab: Dict[Tuple[str, ...], int] = {}
        for word, freq in vocab.items():
            merged = tuple(_merge_pair(list(word), best_pair))
            next_vocab[merged] = next_vocab.get(merged, 0) + freq
        vocab = next_vocab
        if log_every_merges > 0 and (merge_i + 1) % log_every_merges == 0:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] merges={merge_i + 1}/{target_merges} | symbols={len(symbols)}"
            )

    return symbols, merges


def save_word_counts(path: str, word_counts: Dict[str, int], punct_counts: Dict[str, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"word_counts": word_counts, "punct_counts": punct_counts}, f, ensure_ascii=False)


def load_word_counts(path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    word_counts = {k: int(v) for k, v in data.get("word_counts", {}).items()}
    punct_counts = {k: int(v) for k, v in data.get("punct_counts", {}).items()}
    return word_counts, punct_counts
