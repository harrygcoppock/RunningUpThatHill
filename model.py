"""A vanilla GPT-2 style transformer. Adapted from nanoGPT (github.com/karpathy/nanoGPT, MIT)."""
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 257
    block_size: int = 256  # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    ngram_buckets: int = 8192  # rows of the hashed bigram/trigram embedding table
    ngram_dim: int = 64  # its width, projected up to n_embd


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Fused attention (FlashAttention / memory-efficient kernel): softmax(QK^T/sqrt(d))V is computed
        # tile by tile in SRAM, so the (T, T) attention matrix is never written to HBM. is_causal=True
        # applies the same no-peeking mask as before.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NgramHashEmbedding(nn.Module):
    """Hashed n-gram input embedding (the "BigramHash" trick from OpenAI's parameter-golf challenge), here for
    bigrams and trigrams. A byte on its own says little: the model has to spend attention and layers just to
    find out which 2-3 bytes came before it. This looks the previous one and two tokens up directly: each
    (prev, cur) and (prev2, prev, cur) is hashed into a shared table of ngram_buckets rows, and the sum is
    projected up and added to the token embedding. It only reads tokens at or before position t, inside the
    same row, so it stays causal. Positions with no history in the window use their own row."""

    def __init__(self, cfg):
        super().__init__()
        self.buckets = cfg.ngram_buckets
        self.table = nn.Embedding(cfg.ngram_buckets, cfg.ngram_dim)
        self.proj = nn.Linear(cfg.ngram_dim, cfg.n_embd, bias=False)
        nn.init.zeros_(self.proj.weight)  # starts as a no-op, so training begins exactly like the plain model

    def forward(self, idx):
        cur = idx.long()
        prev = F.pad(cur, (1, 0), value=-1)[:, :-1]
        prev2 = F.pad(cur, (2, 0), value=-1)[:, :-2]
        n = self.buckets - 1  # the last row is "no history"
        h2 = torch.where(prev < 0, n, ((36313 * cur) ^ (27191 * prev)) % n)
        h3 = torch.where(prev2 < 0, n, ((36313 * cur) ^ (27191 * prev) ^ (51497 * prev2) ^ 1) % n)
        return self.proj(self.table(h2) + self.table(h3))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.ngram_emb = NgramHashEmbedding(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx):
        """idx: (B, T) token ids -> logits: (B, T, vocab_size)"""
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos) + self.ngram_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def save_model(model, out_dir):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, os.path.join(out_dir, "model.pt"))


def load_model(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model
