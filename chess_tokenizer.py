"""Character-level tokenizer for chess movetext in SAN, e.g. ";1.e4 e5 2.Nf3 Nc6 3.Bb5 a6".

Every character that can appear in a SAN game gets one id. Id 0 is padding.
";" marks the start of a game.
"""
PAD = 0
_CHARS = " #+-.0123456789;=BKNOQRabcdefghx"
STOI = {c: i + 1 for i, c in enumerate(_CHARS)}   # 0 is reserved for PAD
ITOS = {i: c for c, i in STOI.items()}
# 33 real symbols; round up to 64 so the output layer has a GPU-friendly shape (extra ids are unused)
VOCAB_SIZE = 64


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(ITOS.get(int(i), "") for i in ids)
