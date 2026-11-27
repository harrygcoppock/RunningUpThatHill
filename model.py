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
        # GPT-2 initialisation (Radford et al. 2019, as in nanoGPT) instead of PyTorch's defaults, which start
        # the embeddings at N(0, 1), ~50x larger than everything they are added to. All weights get N(0, 0.02),
        # biases 0, and the two projections that write into the residual stream get 0.02 / sqrt(2 * n_layer),
        # so the residual stream's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if p.dim() == 1:
                if not name.endswith(("ln1.weight", "ln2.weight", "ln_f.weight")):
                    nn.init.zeros_(p)
            else:
                std = 0.02 / math.sqrt(2 * cfg.n_layer) if name.endswith("proj.weight") else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

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
