# Running up that hill

Train the best language model you can in **5 minutes on one NVIDIA L4**.
Score: **bits per byte (bpb)** on a held-out test set. Lower is better.

The starting point is a deliberately vanilla GPT (adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)):
fp32, manual attention, learned position embeddings, AdamW at a constant learning rate, byte-level tokenizer.
Every trick from the lectures is still on the table.

## Quickstart

```bash
pip install -r requirements.txt
python prepare.py        # download data (once)
python train.py          # 5 min of training -> out/
python eval.py           # -> val bpb
```

Scoring runs on **PyTorch 2.8.0** (Python 3.12, CUDA 12.9), the AWS
`pytorch-training:2.8.0-gpu-py312-cu129` image. For local results that match, use the same version:
`pip install torch==2.8.0`.

Override settings from the command line, e.g. `python train.py n_layer=6 n_embd=384 time_budget=60`.

The baseline (4 layers × 256, 3.4M parameters, 13.4 MB) scores **2.121 bpb** on the L4.

## Files

| File | What it is | Can you change it? |
|---|---|---|
| `model.py` | The transformer, plus `save_model` / `load_model` | Yes |
| `tokenizer.py` | Byte-level tokenizer | Yes |
| `train.py` | Training loop with a 5-minute budget | Yes |
| `eval.py` | The referee | **No** |
| `prepare.py` | Downloads the data | **No** |

## Rules

1. **One change per PR**, and you explain your diff to the class.
2. We rerun your `train.py` on the reference L4 and score it with `eval.py`. You do not submit weights.
3. Training reads only `data/train.txt`. No pretrained weights, and no other data.
4. Any tokenizer is allowed, as long as it is lossless. Its files go in `out/` and count towards the size limit.
5. **Size limit: 16 MB** (16,000,000 bytes) for `model.py` + `tokenizer.py` + everything in `out/`.
6. A new record must beat the current one by **at least 0.02 bpb** (about 3× the run-to-run noise). Closer results are rerun with a second seed.
7. Only packages already in the scoring image (PyTorch 2.8.0) are available, and there is no internet access.
8. Keep `train.py`'s `step … | loss … | …s` log line, so your training curve can be plotted.

## How to submit

[Fork this repo](https://github.com/harrygcoppock/RunningUpThatHill/fork), make your change on a branch of your fork, and open a PR
back to `main` here. It must change only `model.py`, `tokenizer.py` and/or `train.py`. Fill in the one-line change in the PR template.
Base your branch on the latest `main`, so your change builds on the current record.

What happens next:

1. **Check.** A bot checks that your PR only touches those three files. If it touches anything else, it is rejected and not run.
2. **Approval.** Harry looks at your diff and, if it's fine, approves the run by adding the `score` label.
3. **Run.** Your three files are copied into a sandbox on an NVIDIA L4: no internet, and no access to the held-out test set.
   Your `train.py` trains from scratch for 5 minutes, then the official `eval.py` scores the model and asks it to continue
   *"Deep Learning is all about"*.
4. **Results.** The score, a training-curve plot and your model's rollout are posted as a comment on your PR and added to the
   leaderboard site. The `score` label is then removed: if you push another commit, it needs approving again.
5. **Merge.** If your change beats the current record by at least 0.02 bpb, you explain your diff to the class and Harry merges it.
   It becomes the new record, and everyone builds on it from there.

A run takes about 7 minutes.
