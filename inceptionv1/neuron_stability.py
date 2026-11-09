import json
from pathlib import Path

import numpy as np
import torch

# ---- paths ----
BUNDLE_PATH = Path("replay_bundle.pt")
GROUPS_PATH = Path("class_groups_dogsplit.json")  # from make_groups.py
OUT_PATH = Path("neuron_stability.json")

# ---- config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["inception", "convnext"]
START_EPOCH = 2.0  # skip formation
END_EPOCH = 40.0  # skip the end, where the decaying lr freezes everything
GAP = 1.0  # step stability compares epoch e with e + GAP
N_RANDOM = 2000  # random top-k sets for the "has a theme" baseline
FORMED_PCT = 95
STRENGTH_MULT = 1.0  # raise (e.g. 1.5-2) for a stricter "has a theme at START_EPOCH" filter
N_SHOW = 8
SHOW_EPOCHS = [2, 12, 50]
HIST_BINS = np.linspace(-0.2, 1.0, 13)
SEED = 0

# ---- load ----
bundle = torch.load(BUNDLE_PATH)
groups = json.loads(GROUPS_PATH.read_text())
n_groups = len(groups["groups"])
class_names = bundle["class_names"]
epochs = bundle["epochs"].numpy()
K = bundle["k"]
probe_labels = bundle["probe_labels"].long().to(DEVICE)
n_probe = len(probe_labels)
img_group = torch.tensor(groups["class_to_group"], device=DEVICE)[probe_labels]
top_idx = {m: bundle["top_idx"][m].long().to(DEVICE) for m in MODELS}  # (S, C, K)
base = torch.bincount(img_group, minlength=n_groups).float() / n_probe  # group base rates
gen = torch.Generator(device=DEVICE).manual_seed(SEED)


def snap_at(e):
    return int(np.abs(epochs - e).argmin())


def shares_of(idx):  # (..., K) probe indices -> (..., G) group shares
    g = img_group[idx]
    return torch.zeros(*g.shape[:-1], n_groups, device=DEVICE).scatter_add_(
        -1, g, torch.ones_like(g, dtype=torch.float)) / K


def centered_cos(a, b):  # cosine after removing the base rate: 0 = unrelated, 1 = same theme
    a, b = a - base, b - base
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


def top_classes(m, n, s, k=3):
    vals, cls = torch.bincount(probe_labels[top_idx[m][s, n]], minlength=len(class_names)).topk(k)
    return ", ".join(f"{class_names[c]} {v}" for c, v in zip(cls.tolist(), vals.tolist()) if v > 0)


# ---- "has a theme" threshold: theme strength of random top-k sets ----
rand_idx = torch.rand(N_RANDOM, n_probe, generator=gen, device=DEVICE).topk(K, dim=1).indices
strength_thresh = torch.quantile((shares_of(rand_idx) - base).norm(dim=-1), FORMED_PCT / 100).item() * STRENGTH_MULT

# windows sampled once per epoch so the dense early snapshots don't dominate
win = [snap_at(e) for e in np.arange(START_EPOCH, END_EPOCH - GAP + 1e-6, 1.0)]
win_next = [snap_at(epochs[s] + GAP) for s in win]
s_start, s_end = snap_at(START_EPOCH), len(epochs) - 1
print(f"window epochs {epochs[win[0]]:.1f}-{epochs[win_next[-1]]:.1f}, {len(win)} steps of {GAP:g} epoch; "
      f"theme-strength threshold {strength_thresh:.3f}")

# ---- per-neuron scores ----
results = {}
for m in MODELS:
    shares = shares_of(top_idx[m])  # (S, C, G)
    formed = (shares[s_start] - base).norm(dim=-1) > strength_thresh
    step = torch.stack([centered_cos(shares[a], shares[b]) for a, b in zip(win, win_next)]).mean(0)
    start_end = centered_cos(shares[s_start], shares[s_end])
    ids = formed.nonzero().squeeze(1)
    results[m] = {"formed": formed, "step": step, "start_end": start_end,
                  "rank": ids[step[ids].argsort(descending=True)]}

# ---- report ----
print(f"\n{'':<10} {'formed':>12} {'step stab (median, IQR)':>28} {'start->end (median, IQR)':>28}")
for m in MODELS:
    r = results[m]
    f = r["formed"]
    q = lambda x: [torch.quantile(x[f], p).item() for p in (0.5, 0.25, 0.75)]
    st, se = q(r["step"]), q(r["start_end"])
    print(f"{m:<10} {f.sum().item():>5}/{len(f):<6} {st[0]:>12.2f} ({st[1]:.2f}-{st[2]:.2f}) "
          f"{se[0]:>17.2f} ({se[1]:.2f}-{se[2]:.2f})")

print("\nhistogram of step stability (formed neurons, % of each model)")
print(f"{'bin':>13} {'inception':>10} {'convnext':>10}")
counts = {m: np.histogram(results[m]["step"][results[m]["formed"]].cpu().numpy(), bins=HIST_BINS)[0] for m in MODELS}
for i in range(len(HIST_BINS) - 1):
    row = [counts[m][i] / counts[m].sum() for m in MODELS]
    print(f"{HIST_BINS[i]:>5.1f} - {HIST_BINS[i + 1]:<5.1f} {row[0]:>10.0%} {row[1]:>10.0%}")

for m in MODELS:
    r = results[m]
    for title, sel in [("most stable", r["rank"][:N_SHOW]), ("least stable", r["rank"].flip(0)[:N_SHOW])]:
        print(f"\n{m}: {title}")
        for n in sel.tolist():
            print(f"  neuron {n:4d} | step {r['step'][n]:.2f} | start->end {r['start_end'][n]:.2f}")
            for e in SHOW_EPOCHS:
                print(f"      ep {e:>2}: {top_classes(m, n, snap_at(e))}")

# ---- save ----
OUT_PATH.write_text(json.dumps({
    "groups_file": str(GROUPS_PATH),
    "window": [START_EPOCH, END_EPOCH, GAP],
    "strength_thresh": strength_thresh,
    "models": {m: {
        "step": [round(v, 4) for v in results[m]["step"].tolist()],
        "start_end": [round(v, 4) for v in results[m]["start_end"].tolist()],
        "formed": results[m]["formed"].tolist(),
        "rank_by_step": results[m]["rank"].tolist(),  # formed neurons, most -> least stable
    } for m in MODELS},
}))
print(f"\nsaved {OUT_PATH}")