"""Train a language model for a fixed time budget, then save it to out/.

Usage: python train.py [key=value ...]    e.g. python train.py n_layer=4 n_embd=256
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, save_model
from tokenizer import train_tokenizer


class Muon(torch.optim.Optimizer):
    """Single-device Muon; https://github.com/KellerJordan/Muon."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                momentum = state["momentum_buffer"]
                momentum.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp(momentum, group["momentum"])
                dtype = torch.float32 if p.device.type == "mps" else torch.bfloat16
                x = update.to(dtype)
                if p.size(0) > p.size(1):
                    x = x.T
                x = x / (x.norm() + 1e-7)
                for _ in range(5):
                    a = x @ x.T
                    b = -4.7750 * a + 2.0315 * a @ a
                    x = 3.4445 * x + b @ x
                if p.size(0) > p.size(1):
                    x = x.T
                x *= max(1, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(x, alpha=-group["lr"])


# ---- config (override any of these from the command line with key=value) ----
time_budget = 300.0  # seconds of training
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
batch_size = 64
learning_rate = 6e-4
muon_learning_rate = 0.02
seed = 1337
out_dir = "out"
for arg in sys.argv[1:]:
    key, val = arg.split("=")
    globals()[key] = type(globals()[key])(val)
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs(out_dir, exist_ok=True)

with open("data/train.txt", "rb") as f:
    text = f.read()
tok = train_tokenizer(text, out_dir)
data = tok.encode(text)
del text


def get_batch():
    """A batch of random windows: y is x shifted one token to the left."""
    ix = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


model = GPT(GPTConfig(tok.vocab_size, block_size, n_layer, n_head, n_embd)).to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}")
muon_params = [p for p in model.blocks.parameters() if p.ndim == 2]
muon_ids = {id(p) for p in muon_params}
optimizer = torch.optim.AdamW([p for p in model.parameters() if id(p) not in muon_ids], lr=learning_rate)
muon_optimizer = Muon(muon_params, lr=muon_learning_rate)

step, t0 = 0, time.time()
while time.time() - t0 < time_budget:
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    optimizer.zero_grad()
    muon_optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    muon_optimizer.step()
    step += 1
    if step % 100 == 0:
        elapsed = time.time() - t0
        print(f"step {step} | loss {loss.item():.4f} | {elapsed:.0f}s | {step * batch_size * block_size / elapsed:,.0f} tok/s")

print(f"done: {step} steps, {step * batch_size * block_size:,} tokens in {time.time() - t0:.0f}s")
save_model(model, out_dir)
