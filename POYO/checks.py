from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sae import TopKSAE
from utils import get_trials, load_model, make_batches, run_decoder

CKPT = Path.home() / "repos/torch_brain/examples/poyo/outputs/2026-10-05/14-33-20/checkpoints/best.pt"
DATA_ROOT = Path.home() / "brainsets/processed"
DEVICE = torch.device("cuda")
SAE_PATH = "sae.pt"

OUT_DT = 0.01
OUT_TS = np.arange(0, 1.0, OUT_DT, dtype=np.float32)
PRE = 0.25  # window starts 0.25 s before reach onset (get_trials default)

N_BIG = 4  # features with the largest output push
N_SMALL = 10  # most frequent of the remaining features

# ----------------------------- model + SAE

model, ds = load_model(CKPT, DATA_ROOT, DEVICE)
SESSIONS = ds.recording_ids

ckpt = torch.load(SAE_PATH)
sae = TopKSAE(model.readout.in_features, ckpt["n_latents"], ckpt["k"]).to(DEVICE)
sae.load_state_dict(ckpt["sae"])

# ----------------------------- data: SAE latents on reach-aligned trials

trials, targets = get_trials(ds, n_per=10)
batches = make_batches(model, ds, trials, OUT_TS, DEVICE)
dec, _ = run_decoder(model, batches)

with torch.no_grad():
    z = sae.encode(dec).cpu().numpy()  # (n_trials, T, n_latents)
    out_dir = (model.readout.weight @ sae.dec.weight).T.cpu().numpy()  # (n_latents, 2) velocity push

sess_idx = np.array([SESSIONS.index(rid) for rid, _, _ in trials])
push = np.linalg.norm(out_dir, axis=1)
freq = (z > 0).mean(axis=(0, 1))
live = np.flatnonzero(freq > 0)

big = live[np.argsort(push[live])[::-1][:N_BIG]]
rest = np.setdiff1d(live, big)
small = rest[np.argsort(freq[rest])[::-1][:N_SMALL]]
features = np.concatenate([big, small])
print(f"{len(trials)} trials, {len(live)} live latents")

# ----------------------------- colors + labels

TARGET_COLORS = plt.cm.hsv(np.linspace(0, 1, 8, endpoint=False))  # 0 = red, matches the viz
t = OUT_TS - PRE


def label(j):
    kind = "big" if j in big else "small"
    ang = np.degrees(np.arctan2(out_dir[j, 1], out_dir[j, 0]))
    return f"{kind} {j}: {ang:.0f}°, push {push[j]:.2f}, freq {freq[j]:.0%}"

# ----------------------------- 1. timing: mean activation over the trial, per target

fig, axes = plt.subplots(2, 7, figsize=(24, 7), sharex=True)
for ax, j in zip(axes.flat, features):
    for tg in range(8):
        ax.plot(t, z[targets == tg, :, j].mean(axis=0), color=TARGET_COLORS[tg], lw=1.2)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(label(j), fontsize=8)
fig.supxlabel("time from reach onset (s)")
fig.supylabel("mean activation")
fig.suptitle("SAE features: activation over the reach, one line per target")
fig.tight_layout()

# ----------------------------- 2. session identity: mean activation per session

sess_mean = np.stack(
    [z[sess_idx == s].mean(axis=(0, 1)) for s in range(len(SESSIONS))], axis=1
)  # (n_latents, 16)
is_c = np.array([r.startswith("c_") for r in SESSIONS])
c_frac = sess_mean[:, is_c].sum(axis=1) / (sess_mean.sum(axis=1) + 1e-9)  # 0.5 = no monkey preference

rows = sess_mean[features] / (sess_mean[features].max(axis=1, keepdims=True) + 1e-9)

fig2, ax = plt.subplots(figsize=(11, 6))
ax.imshow(rows, aspect="auto", cmap="viridis")
ax.axvline(is_c.sum() - 0.5, color="white", lw=2)  # sessions are sorted: C first, then M
ax.set_yticks(range(len(features)), [f"{label(j)}   C share {c_frac[j]:.0%}" for j in features], fontsize=8)
ax.set_xticks(range(len(SESSIONS)), [r.split("_center")[0] for r in SESSIONS], rotation=90, fontsize=8)
ax.set_title("Mean activation per session (each row scaled to its max); white line = monkey C | M")
fig2.tight_layout()

# monkey-specific features among all live latents
c_only = live[c_frac[live] > 0.8]
m_only = live[c_frac[live] < 0.2]
print(f"monkey-specific latents: {len(c_only)} mostly C, {len(m_only)} mostly M (of {len(live)} live)")
print("  C:", c_only.tolist())
print("  M:", m_only.tolist())

plt.show()