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
        # In-document copy-pointer head (CPLM, modded-nanogpt PR #379)
        ptr_dim = 64
        self.ptr_q = nn.Linear(cfg.n_embd, ptr_dim, bias=False)
        self.ptr_k = nn.Linear(cfg.n_embd, ptr_dim, bias=False)
        self.ptr_gate = nn.Linear(cfg.n_embd, 1)
        self.ptr_sink = nn.Linear(cfg.n_embd, 1)
        self.ptr_scale = nn.Parameter(torch.tensor(ptr_dim ** -0.5))
        strict_tril = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool), diagonal=-1)
        self.register_buffer("strict_tril", strict_tril, persistent=False)

    def forward(self, idx):
        """idx: (B, T) token ids -> log-probabilities: (B, T, vocab_size)"""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        logits = self.head(h)
        if T <= 1:
            return logits

        pq = self.ptr_q(h).float()
        pk = self.ptr_k(h[:, :-1]).float()
        pq = pq * torch.rsqrt(pq.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        pk = pk * torch.rsqrt(pk.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        scores = torch.bmm(pq, pk.transpose(1, 2)) * self.ptr_scale.float()
        scores = scores.masked_fill(~self.strict_tril[:T, : T - 1], float("-inf"))
        sink = self.ptr_sink(h).float()
        weights = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)
        w_copy, w_sink = weights[:, :, :-1], weights[:, :, -1:]

        p_copy = torch.zeros(B, T, logits.size(-1), device=logits.device, dtype=torch.float32)
        targets = idx[:, 1:].unsqueeze(1).expand(B, T, T - 1)
        p_copy.scatter_add_(2, targets, w_copy)

        alpha = torch.sigmoid(self.ptr_gate(h).float())
        p_lm = F.softmax(logits.float(), dim=-1)
        p = p_lm * (1.0 - alpha * (1.0 - w_sink)) + alpha * p_copy
        return torch.log(p.clamp_min(1e-7))


def save_model(model, out_dir):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, os.path.join(out_dir, "model.pt"))


def load_model(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model
