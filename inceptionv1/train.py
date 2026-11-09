import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from model import (
    DATA_PATH,
    DEVICE,
    IMG_SIZE,
    MEAN,
    STD,
    acts,
    convnext_hook_module,
    inception_hook_modules,
    make_convnext,
    make_inception_v1,
    save_to,
)

# ---- config ----
SEED = 0
EPOCHS = 100  # full schedule length (lr schedule is built for this)
N_EPOCHS_RUN = 100  # set to 1 for a smoke test
BATCH = 256
LR = 2e-3
WD = 0.05
WARMUP_EPOCHS = 5
GRAD_CLIP = 5.0
N_CLASSES = 100

# regularization (identical for both models; ConvNeXt stochastic depth is set in model_test.make_convnext)
MAX_SHIFT = 0.25  # random translation, +-4px of 32px in normalized coords
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
ERASE_P = 0.25  # per-image probability of random erasing

# probes: dense early (recursive-generalization window), sparse later
PROBE_BATCH = 500
PROBE_EVERY_EARLY = 25
PROBE_EVERY_LATE = 390  # every 2 epochs
EARLY_EPOCHS = 3

# checkpoints: every epoch early, then every 5
CKPT_DENSE_EPOCHS = 12
CKPT_EVERY_LATE = 5

N_TRAIN_EVAL = 10_000  # clean (unaugmented) train subset, to watch the train/test gap
CKPT_DIR = Path("ckpts")
HIST_PATH = "probe_hist.pt"

torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
CKPT_DIR.mkdir(exist_ok=True)

# ---- data, all on GPU ----
data = torch.load(DATA_PATH)
train_imgs = data["train_imgs"].to(DEVICE)
train_fine = data["train_fine"].to(DEVICE)
test_imgs = data["test_imgs"].to(DEVICE)
test_fine = data["test_fine"].to(DEVICE)
probe_idx = data["probe_idx"].to(DEVICE)
probe_imgs = test_imgs[probe_idx]
probe_fine = test_fine[probe_idx]

n_train = len(train_imgs)
n_probe = len(probe_imgs)
steps_per_epoch = n_train // BATCH
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch
early_steps = EARLY_EPOCHS * steps_per_epoch
train_eval_idx = torch.randperm(n_train, device=DEVICE)[:N_TRAIN_EVAL]

mean = MEAN.to(DEVICE)
std = STD.to(DEVICE)
mix = v2.RandomChoice([v2.MixUp(num_classes=N_CLASSES, alpha=MIXUP_ALPHA), v2.CutMix(num_classes=N_CLASSES, alpha=CUTMIX_ALPHA)])


# ---- GPU preprocessing: normalize, random flip + shift, upsample to IMG_SIZE in one grid_sample ----
def preprocess(x_uint8, augment):
    x = (x_uint8.float() / 255 - mean) / std
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=DEVICE)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    if augment:
        theta[:, 0, 0] = torch.where(torch.rand(b, device=DEVICE) < 0.5, -1.0, 1.0)
        theta[:, :, 2] = (torch.rand(b, 2, device=DEVICE) * 2 - 1) * MAX_SHIFT
    grid = F.affine_grid(theta, (b, 3, IMG_SIZE, IMG_SIZE), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


# per-image random erasing (area 2-33%, aspect 0.3-3.3), filled with noise, vectorized on GPU
def random_erase(x):
    b, _, h, w = x.shape
    area = torch.empty(b, device=DEVICE).uniform_(0.02, 0.33) * h * w
    ratio = torch.exp(torch.empty(b, device=DEVICE).uniform_(math.log(0.3), math.log(3.3)))
    eh = (area * ratio).sqrt().clamp(max=h).long()
    ew = (area / ratio).sqrt().clamp(max=w).long()
    y0 = (torch.rand(b, device=DEVICE) * (h - eh + 1)).long()
    x0 = (torch.rand(b, device=DEVICE) * (w - ew + 1)).long()
    ys = torch.arange(h, device=DEVICE).view(1, h, 1)
    xs = torch.arange(w, device=DEVICE).view(1, 1, w)
    mask = (
        (ys >= y0.view(b, 1, 1)) & (ys < (y0 + eh).view(b, 1, 1))
        & (xs >= x0.view(b, 1, 1)) & (xs < (x0 + ew).view(b, 1, 1))
        & (torch.rand(b, device=DEVICE) < ERASE_P).view(b, 1, 1)
    )
    return torch.where(mask[:, None], torch.randn_like(x), x)


def lr_at(step):
    if step < warmup_steps:
        return LR * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return LR * 0.5 * (1 + math.cos(math.pi * t))


def should_probe(step):
    return step % (PROBE_EVERY_EARLY if step < early_steps else PROBE_EVERY_LATE) == 0


def should_ckpt(epoch):
    return epoch < CKPT_DENSE_EPOCHS or (epoch + 1) % CKPT_EVERY_LATE == 0 or epoch == N_EPOCHS_RUN - 1


# ---- models + optimizers (identical recipe: AdamW, no decay on 1-D params) ----
inception = make_inception_v1().to(DEVICE)
convnext = make_convnext().to(DEVICE)
models = {"inception": inception, "convnext": convnext}


def param_groups(model):
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]  # biases, norms, layer scales
    return [{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}]


opts = {name: torch.optim.AdamW(param_groups(m), lr=LR) for name, m in models.items()}

cnx_fc1 = convnext_hook_module(convnext)
cnx_channels_last = isinstance(cnx_fc1, nn.Linear)


# ---- probe: per-neuron spatial max / min on the fixed probe set ----
def layer_acts(name):
    if name == "inception":
        return torch.cat([acts[f"inc_{i}"] for i in range(4)], dim=1)  # (B, 1024, H, W)
    a = acts["cnx"]
    return a.permute(0, 3, 1, 2) if cnx_channels_last else a  # (B, C, H, W)


@torch.no_grad()
def run_probe():
    handles = [m.register_forward_hook(save_to(f"inc_{i}")) for i, m in enumerate(inception_hook_modules(inception))]
    handles.append(cnx_fc1.register_forward_hook(save_to("cnx")))
    result = {}
    for name, model in models.items():
        model.eval()
        maxs, mins, correct = [], [], 0
        for s in range(0, n_probe, PROBE_BATCH):
            x = preprocess(probe_imgs[s:s + PROBE_BATCH], augment=False)
            logits = model(x)
            a = layer_acts(name)
            maxs.append(a.amax(dim=(2, 3)))
            mins.append(a.amin(dim=(2, 3)))
            correct += (logits.argmax(1) == probe_fine[s:s + PROBE_BATCH]).sum()
        result[name] = {"max": torch.cat(maxs), "min": torch.cat(mins), "acc": correct.item() / n_probe}
    for h in handles:
        h.remove()
    acts.clear()
    return result


def n_dead(pos):  # neurons never positive on the probe set
    return (pos.amax(0) <= 0).sum().item()


@torch.no_grad()
def accuracy(model, imgs, labels):
    model.eval()
    correct = 0
    for s in range(0, len(imgs), PROBE_BATCH):
        logits = model(preprocess(imgs[s:s + PROBE_BATCH], augment=False))
        correct += (logits.argmax(1) == labels[s:s + PROBE_BATCH]).sum()
    return correct.item() / len(imgs)


def report_acc(prefix):
    parts = []
    for name, m in models.items():
        tr = accuracy(m, train_imgs[train_eval_idx], train_fine[train_eval_idx])
        te = accuracy(m, test_imgs, test_fine)
        parts.append(f"{name} train {tr:.3f} test {te:.3f}")
    print(f"{prefix} | " + " | ".join(parts))


# ---- history (GPU-resident fp16, one entry per probe) ----
hist_steps = []
hist = {name: {"max": [], "min": [], "acc": []} for name in models}

# ---- train ----
print(f"{steps_per_epoch} steps/epoch, running {N_EPOCHS_RUN} of {EPOCHS} epochs")
last_loss = {}
step = 0
t0 = time.time()
for epoch in range(N_EPOCHS_RUN):
    perm = torch.randperm(n_train, device=DEVICE)
    for i in range(steps_per_epoch):
        idx = perm[i * BATCH:(i + 1) * BATCH]
        x = random_erase(preprocess(train_imgs[idx], augment=True))
        x, y = mix(x, train_fine[idx])  # y: soft labels (B, N_CLASSES)
        lr = lr_at(step)

        for name, model in models.items():
            model.train()
            for g in opts[name].param_groups:
                g["lr"] = lr
            loss = F.cross_entropy(model(x), y, label_smoothing=LABEL_SMOOTH)
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opts[name].step()
            last_loss[name] = loss.detach()

        if should_probe(step):
            probe = run_probe()
            hist_steps.append(step)
            for name in models:
                hist[name]["max"].append(probe[name]["max"].half())
                hist[name]["min"].append(probe[name]["min"].half())
                hist[name]["acc"].append(probe[name]["acc"])
            if step % 100 == 0 or step >= early_steps:
                print(f"  probe @ {step}: acc inc {probe['inception']['acc']:.3f} cnx {probe['convnext']['acc']:.3f} | "
                      f"dead inc {n_dead(probe['inception']['max'])} cnx {n_dead(probe['convnext']['max'])}")

        if step % 100 == 0:
            torch.cuda.synchronize()
            li, lc = last_loss["inception"].item(), last_loss["convnext"].item()
            flag = "" if math.isfinite(li) and math.isfinite(lc) else "  <-- NON-FINITE LOSS"
            print(f"epoch {epoch} step {step} lr {lr:.2e} | loss inc {li:.3f} cnx {lc:.3f} | {time.time() - t0:.0f}s{flag}")

        step += 1

    if should_ckpt(epoch):
        torch.save({name: m.state_dict() for name, m in models.items()}, CKPT_DIR / f"epoch_{epoch:02d}.pt")
    report_acc(f"epoch {epoch} done")

# ---- summary ----
torch.cuda.synchronize()
print(f"\ndone: {step} steps in {time.time() - t0:.0f}s")
report_acc("final")
probe = run_probe()
for name in models:
    print(f"{name}: dead {n_dead(probe[name]['max'])} / {probe[name]['max'].shape[1]}")

ls = [p.abs().mean().item() for n, p in convnext.named_parameters() if "gamma" in n]
print(f"convnext layer-scale mean |gamma|: first {ls[0]:.4f} last {ls[-1]:.4f}")

# ---- save probe history for the viz ----
torch.save(
    {
        "steps": hist_steps,
        "steps_per_epoch": steps_per_epoch,
        "probe_idx": data["probe_idx"],
        **{f"{name}_{k}": torch.stack(hist[name][k]).cpu() for name in models for k in ("max", "min")},
        **{f"{name}_acc": hist[name]["acc"] for name in models},
    },
    HIST_PATH,
)
print(f"saved {HIST_PATH} ({len(hist_steps)} snapshots) and checkpoints in {CKPT_DIR}/")