"""Lossless byte-level BPE tokenizer trained on data/train.txt.

Vocab layout (default vocab_size=1024):
  0..255                : raw bytes
  256..vocab_size-2     : learned BPE merges
  vocab_size-1          : BOS token
"""
import array
from collections import Counter, defaultdict
import os
import re

import numpy as np

# Lossless pre-tokenization: `|.` with DOTALL guarantees every byte is matched.
_CHUNK_RE = re.compile(rb" ?[A-Za-z]+| ?[0-9]+|\s+|.", re.DOTALL)


class BPETokenizer:
    def __init__(self, merges: np.ndarray):
        self.merges = np.asarray(merges, dtype=np.uint16).reshape(-1, 2)
        self.vocab_size = 256 + len(self.merges) + 1
        self.bos = self.vocab_size - 1
        self._ranks = {(int(a), int(b)): i for i, (a, b) in enumerate(self.merges)}
        vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            vocab.append(vocab[int(a)] + vocab[int(b)])
        self._vocab = vocab
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @classmethod
    def train(cls, text: bytes, vocab_size: int = 1024, sample_bytes: int = 4_000_000):
        num_merges = vocab_size - 257
        sample = text[:sample_bytes]
        counts = Counter(_CHUNK_RE.findall(sample))

        words = [list(w) for w in counts.keys()]
        freqs = list(counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for j in range(len(w) - 1):
                p = (w[j], w[j + 1])
                pair_counts[p] += f
                pair_words[p].add(wi)

        merges = []
        for m in range(num_merges):
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: kv[1])[0]
            if pair_counts[best] < 2:
                break
            a, b = best
            new_id = 256 + m
            merges.append((a, b))

            affected = list(pair_words.pop(best, ()))
            pair_counts.pop(best, None)
            for wi in affected:
                w = words[wi]
                f = freqs[wi]
                if len(w) < 2:
                    continue
                # Subtract old pairs for this word
                for j in range(len(w) - 1):
                    p = (w[j], w[j + 1])
                    if p != best:
                        pair_counts[p] -= f
                        if pair_counts[p] <= 0:
                            pair_counts.pop(p, None)
                # Merge (a, b) -> new_id in-place
                nw = []
                j = 0
                n = len(w)
                while j < n:
                    if j + 1 < n and w[j] == a and w[j + 1] == b:
                        nw.append(new_id)
                        j += 2
                    else:
                        nw.append(w[j])
                        j += 1
                words[wi] = nw
                # Add new pairs for this word
                for j in range(len(nw) - 1):
                    p = (nw[j], nw[j + 1])
                    pair_counts[p] += f
                    pair_words[p].add(wi)

        return cls(np.asarray(merges, dtype=np.uint16))

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "bpe_merges.npy"), self.merges)

    def _encode_chunk(self, chunk: bytes) -> tuple[int, ...]:
        ids = list(chunk)
        ranks = self._ranks
        while len(ids) >= 2:
            best_rank = None
            best_pair = None
            for j in range(len(ids) - 1):
                p = (ids[j], ids[j + 1])
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = p
            if best_pair is None:
                break
            a, b = best_pair
            new_id = 256 + best_rank
            out = []
            j = 0
            n = len(ids)
            while j < n:
                if j + 1 < n and ids[j] == a and ids[j + 1] == b:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            ids = out
        return tuple(ids)

    def encode(self, text: bytes) -> np.ndarray:
        cache = self._cache
        enc = self._encode_chunk
        out = array.array("H")
        n = len(text)
        step = 8_000_000
        pos = 0
        while pos < n:
            end = min(pos + step, n)
            if end < n:
                nl = text.find(b"\n", end)
                if nl != -1 and nl - end < 65536:
                    end = nl + 1
            for chunk in _CHUNK_RE.findall(text[pos:end]):
                toks = cache.get(chunk)
                if toks is None:
                    toks = enc(chunk)
                    if len(cache) < 250_000:
                        cache[chunk] = toks
                out.extend(toks)
            pos = end
        return np.frombuffer(out, dtype=np.uint16).copy()

    def decode(self, ids) -> bytes:
        vocab = self._vocab
        bos = self.bos
        return b"".join(vocab[int(i)] for i in ids if int(i) < bos)


def load_tokenizer(out_dir):
    path = os.path.join(out_dir, "bpe_merges.npy")
    if os.path.exists(path):
        return BPETokenizer(np.load(path))
    return BPETokenizer(np.empty((0, 2), dtype=np.uint16))
