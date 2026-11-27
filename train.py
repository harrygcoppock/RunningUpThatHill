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
from tokenizer import load_tokenizer

# ---- config (override any of these from the command line with key=value) ----
time_budget = 300.0  # seconds of training
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
batch_size = 64
learning_rate = 6e-4
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

tok = load_tokenizer(out_dir)
with open("data/train.txt", "rb") as f:
    data = tok.encode(f.read())


def get_batch():
    """A batch of random windows: y is x shifted one token to the left."""
    ix = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

# Don't know if these actually help, but they sometimes do
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

raw = GPT(GPTConfig(tok.vocab_size, block_size, n_layer, n_head, n_embd)).to(device)
model = torch.compile(raw)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

step, t0 = 0, time.time()
while time.time() - t0 < time_budget:
    x, y = get_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    step += 1
    if step % 100 == 0:
        elapsed = time.time() - t0
        print(f"step {step} | loss {loss.item():.4f} | {elapsed:.0f}s | {step * batch_size * block_size / elapsed:,.0f} tok/s")

print(f"done: {step} steps, {step * batch_size * block_size:,} tokens in {time.time() - t0:.0f}s")
save_model(raw, out_dir)
