"""Lossless byte tokens plus 256 frequent byte pairs learned from train.txt."""
import os

import numpy as np


CHUNK_SIZE = 1 << 20


class BytePairTokenizer:
    vocab_size = 513
    bos = 512

    def __init__(self, pairs):
        self.lookup = np.full(65536, -1, dtype=np.int16)
        self.lookup[pairs] = np.arange(256, 512, dtype=np.int16)
        self.pieces = [bytes([i]) for i in range(256)]
        self.pieces += [int(pair).to_bytes(2, "big") for pair in pairs] + [b""]

    def encode(self, text: bytes) -> np.ndarray:
        chunks = []
        # Independent chunks bound temporary memory; a boundary may miss one merge.
        for start in range(0, len(text), CHUNK_SIZE):
            values = np.frombuffer(text[start : start + CHUNK_SIZE], dtype=np.uint8).astype(np.uint16)
            pair_ids = self.lookup[(values[:-1] << 8) | values[1:]]
            positions = np.arange(len(pair_ids))
            candidates = pair_ids >= 0
            previous_gap = np.maximum.accumulate(np.where(candidates, -1, positions))
            selected = candidates & ((positions - previous_gap) % 2 == 1)
            values[:-1][selected] = pair_ids[selected]
            keep = np.ones(len(values), dtype=bool)
            keep[1:][selected] = False
            chunks.append(values[keep])
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)

    def decode(self, ids) -> bytes:
        return b"".join(self.pieces[int(i)] for i in ids)


def train_tokenizer(text: bytes, out_dir):
    counts = np.zeros(65536, dtype=np.int64)
    for start in range(0, len(text) - 1, CHUNK_SIZE):
        values = np.frombuffer(text[start : start + CHUNK_SIZE + 1], dtype=np.uint8).astype(np.uint16)
        counts += np.bincount((values[:-1] << 8) | values[1:], minlength=65536)
    pairs = np.argsort(-counts, kind="stable")[:256].astype(np.uint16)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pairs.npy"), pairs)
    return BytePairTokenizer(pairs)


def load_tokenizer(out_dir):
    return BytePairTokenizer(np.load(os.path.join(out_dir, "pairs.npy"), allow_pickle=False))
