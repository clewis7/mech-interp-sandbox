from pathlib import Path

import torch
from torch_brain.batching import collate
from torch_brain.datasets import DatasetIndex

from poyo_data.wrapper import PoyoDatasetWrapper
from sae import TopKSAE
from utils import load_model

CKPT = Path.home() / "repos/torch_brain/examples/poyo/outputs/2026-10-05/14-33-20/checkpoints/best.pt"
DATA_ROOT = Path.home() / "brainsets/processed"
DEVICE = torch.device("cuda")
SEED = 0
OUT_PATH = "sae.pt"

# ------------- SAE parameters
N_LATENTS = 256  # 4x the 128-d decoder state
K = 6  # active latents per timepoint

# ------------- training parameters
BATCH = 4096
LR = 1e-3
N_EPOCHS = 20
WINDOW_BATCH = 64  # 1 s windows per POYO forward pass when collecting

torch.manual_seed(SEED)

# ----------------------------- model

model, ds = load_model(CKPT, DATA_ROOT, DEVICE)
wrapped = PoyoDatasetWrapper(ds, model.tokenize)

# capture the decoder state = the input to the readout
captured = {}
model.readout.register_forward_pre_hook(lambda m, args: captured.update(dec=args[0].detach()))

# ----------------------------- collect decoder states

def window_indices(split):
    """1 s windows tiling every interval of `split` that is at least 1 s long."""
    L = model.sequence_length
    indices = []
    for rid, iv in wrapped.get_sampling_intervals(split).items():
        for start, end in zip(iv.start, iv.end):
            t = start
            while t + L <= end:
                indices.append(DatasetIndex(rid, float(t), float(t + L)))
                t += L
    return indices

@torch.no_grad()
def collect(split):
    """Decoder states and target velocities at every cursor timestamp in `split`.

    Returns acts (n, 128) and targets (n, 2, normalized like training), on GPU.
    """
    indices = window_indices(split)

    acts, targets = [], []
    for i in range(0, len(indices), WINDOW_BATCH):
        X, Y = collate([wrapped[j] for j in indices[i:i + WINDOW_BATCH]])
        X = {k: v.to(DEVICE) for k, v in X.items()}
        mask = Y["output_mask"].to(DEVICE)

        model(**X, output_timestamps=Y["timestamps"].to(DEVICE))
        acts.append(captured["dec"][mask])
        targets.append(Y["values"].to(DEVICE)[mask])

    return torch.cat(acts), torch.cat(targets)


train_acts, _ = collect("train")
val_acts, val_targets = collect("valid")
print(f"train {tuple(train_acts.shape)}, valid {tuple(val_acts.shape)}")

# ----------------------------- SAE

sae = TopKSAE(train_acts.shape[1], N_LATENTS, K).to(DEVICE)
sae.b_pre.data = train_acts.mean(dim=0)  # start centered on the data
optim = torch.optim.Adam(sae.parameters(), lr=LR)


def r2(pred, target):
    """R² pooled over samples and dims."""
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum()
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def evaluate():
    x_hat, z = sae(val_acts)
    recon = r2(x_hat, val_acts).item()
    dead = (z.amax(dim=0) == 0).float().mean().item()
    return recon, dead


# ----------------------------- train

n = len(train_acts)
for epoch in range(N_EPOCHS):
    perm = torch.randperm(n, device=DEVICE)
    total = 0.0
    for i in range(0, n, BATCH):
        x = train_acts[perm[i:i + BATCH]]
        x_hat, _ = sae(x)
        loss = ((x_hat - x) ** 2).sum(dim=-1).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        sae.normalize_decoder()
        total += loss.item() * len(x)

    recon, dead = evaluate()
    print(f"epoch {epoch:2d}   train mse {total / n:.4f}   val recon R² {recon:.3f}   dead {dead:.1%}")

# ----------------------------- final check: does the model still decode through the SAE?

with torch.no_grad():
    x_hat, z = sae(val_acts)
    intact = r2(model.readout(val_acts), val_targets).item()
    through_sae = r2(model.readout(x_hat), val_targets).item()
    l0 = (z > 0).float().sum(dim=-1).mean().item()

print(f"decoding R² intact {intact:.3f}, through SAE {through_sae:.3f}")
print(f"active latents per timepoint {l0:.1f} (k = {K})")

torch.save(
    {"sae": sae.state_dict(), "n_latents": N_LATENTS, "k": K, "ckpt": str(CKPT)},
    OUT_PATH,
)
print(f"saved {OUT_PATH}")