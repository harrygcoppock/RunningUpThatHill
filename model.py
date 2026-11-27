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
    n_expert: int = 8  # routed experts per layer
    expert_top_k: int = 2  # routed experts each token uses
    expert_hidden: int = 128  # hidden width of each routed expert
    shared_hidden: int = 256  # hidden width of the always-on shared expert


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


class FFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc = nn.Linear(d, h)
        self.proj = nn.Linear(h, d)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class MoE(nn.Module):
    """DeepSeek-V3 style mixture of experts: one shared expert every token uses, plus top-k of n_expert
    small routed experts chosen by sigmoid affinity. Load balancing has no auxiliary loss: a per-expert
    bias is added to the scores only for choosing experts. After each training forward it is nudged
    up for under-used experts and down for over-used ones."""

    def __init__(self, cfg, bias_update_speed=1e-3):
        super().__init__()
        self.top_k, self.speed = cfg.expert_top_k, bias_update_speed
        self.shared = FFN(cfg.n_embd, cfg.shared_hidden)
        self.experts = nn.ModuleList([FFN(cfg.n_embd, cfg.expert_hidden) for _ in range(cfg.n_expert)])
        self.router = nn.Linear(cfg.n_embd, cfg.n_expert, bias=False)
        self.register_buffer("route_bias", torch.zeros(cfg.n_expert))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])  # (N, C): routing is per token, so batch rows never mix
        scores = torch.sigmoid(self.router(x).float())  # (N, n_expert) affinities
        idx = (scores + self.route_bias).topk(self.top_k, dim=-1).indices  # bias picks, it does not weight
        gate = scores.gather(1, idx)
        gate = (gate / gate.sum(-1, keepdim=True)).to(x.dtype)  # normalised over the chosen experts
        out = self.shared(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out = out.index_add(0, tok, expert(x[tok]) * gate[tok, slot, None])
        if self.training:
            with torch.no_grad():
                load = torch.bincount(idx.flatten(), minlength=len(self.experts)).float()
                self.route_bias += self.speed * torch.sign(load.mean() - load)
        return out.view(shape)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MoE(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


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

    def forward(self, idx):
        """idx: (B, T) token ids -> logits: (B, T, vocab_size)"""
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
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
