"""A vanilla GPT-2 style transformer. Adapted from nanoGPT (github.com/karpathy/nanoGPT, MIT)."""
import math
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
    mtp: bool = False  # add the training-only multi-token prediction module


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))  # no peeking at the future
        y = F.softmax(att, dim=-1) @ v
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


class MTP(nn.Module):
    """DeepSeek-V3 multi-token prediction module (depth 1). It combines the main model's final hidden
    state at position i with the embedding of token i+1, runs one more transformer block, and predicts
    token i+2 through the shared output head. It only adds training signal, and is not saved."""

    def __init__(self, cfg):
        super().__init__()
        self.norm_h = nn.LayerNorm(cfg.n_embd)
        self.norm_e = nn.LayerNorm(cfg.n_embd)
        self.proj = nn.Linear(2 * cfg.n_embd, cfg.n_embd, bias=False)
        self.block = Block(cfg)

    def forward(self, h, next_emb):
        return self.block(self.proj(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1)))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.mtp = MTP(cfg) if cfg.mtp else None

    def forward(self, idx, return_hidden=False):
        """idx: (B, T) token ids -> logits: (B, T, vocab_size), plus the final hidden states if asked"""
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        return (logits, x) if return_hidden else logits

    def mtp_logits(self, h, idx):
        """Logits for token i+2 at each position i < T-1, from hidden states h of model(idx)."""
        h2 = self.mtp(h[:, :-1], self.tok_emb(idx[:, 1:]))
        return self.head(self.ln_f(h2))


def save_model(model, out_dir):
    # The MTP module is only for training: drop it, so it costs nothing at eval time or towards the size limit.
    config = {**asdict(model.cfg), "mtp": False}
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("mtp.")}
    torch.save({"config": config, "state_dict": state}, os.path.join(out_dir, "model.pt"))


def load_model(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model
