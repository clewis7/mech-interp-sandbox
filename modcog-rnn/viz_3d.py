import mechiviz as mv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastplotlib as fpl
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from model import LeakyRNN
from projection import ProbeProjector, add_color_wheel

CKPT_PATH = "modcog_rnn.pt"
DEVICE = "cuda"

# ------------------------------------ Training Parameters
N_HIDDEN = 256
TAU_MS = 200
SIGMA_REC = 0.05      # recurrent noise (training only)
ACTIVATION = "relu"   # "relu" | "tanh" | "softplus"
L2_RATE = 1e-4     # penalty on hidden activity
L2_WEIGHT = 1e-4   # penalty on recurrent weights

BATCH = 256
LR = 1e-3
N_STEPS = 20_000
GRAD_CLIP = 1.0
RESP_WEIGHT = 5.0     # upweight response steps; fixation dominates the labels

VAL_FRAC = 0.1
EVAL_EVERY = 500
EVAL_BATCH = 2048
PROBE_PER_TASK = 8

IGNORE_INDEX = -100

# ------------------------------------ Demo parameters
STEPS_PER_FRAME = 5    # training steps between redraws
REFIT_EVERY = 150      # steps between direction-plane refits
SMOOTH = 0.5           # frame-to-frame smoothing of the projected positions
RING_RADIUS = 1.0
VIEW_LIM = 1.6

ACC_EVERY = 200        # steps between validation accuracy checks
ACC_BATCH = 2048
Z_SCALE = 2.0

# ----------------------------------- Data loading
BANK_PATH = "modcog_bank.pt"
bank = torch.load(BANK_PATH, map_location=DEVICE)

obs = bank["obs"]  # (N, T, 33)
labels = bank["labels"]  # (N, T), padding = -100
lengths = bank["lengths"]  # (N,)
task_id = bank["task_id"]  # (N,)
family = bank["family"]  # (N,)
extension = bank["extension"]  # (N,) 0 base, 1 int, 2 seq
task_names = bank["task_names"]
family_names = bank["family_names"]
extension_names = bank["extension_names"]

n_trials, t_max, ob_dim = obs.shape
n_tasks = len(task_names)
n_in = ob_dim + n_tasks
n_out = bank["n_ring"] + 1
alpha = bank["dt"] / TAU_MS
rule_eye = torch.eye(n_tasks, device=DEVICE)

print(f"{n_trials} trials, T={t_max}, n_in={n_in}, n_out={n_out}, alpha={alpha}")

# ------------------------------ Train/test/val split

perm = torch.randperm(n_trials, device=DEVICE)
n_val = int(VAL_FRAC * n_trials)
val_idx = perm[:n_val]
train_idx = perm[n_val:]

# fixed probe set: first PROBE_PER_TASK val trials of each task (used for trajectory viz later)
val_tasks = task_id[val_idx]
order = torch.argsort(val_tasks, stable=True)
sorted_idx = val_idx[order]
sorted_tasks = val_tasks[order]
counts = torch.bincount(sorted_tasks, minlength=n_tasks)
starts = torch.cumsum(counts, 0) - counts
rank = torch.arange(len(sorted_idx), device=DEVICE) - starts[sorted_tasks]
probe_idx = sorted_idx[rank < PROBE_PER_TASK]
probe_idx = probe_idx[torch.argsort(extension[probe_idx], stable=True)]

print(f"train {len(train_idx)}, val {len(val_idx)}, probe {len(probe_idx)}")


def make_batch(idx):
    """Gather trials and append the rule one-hot, all on GPU."""
    x = obs[idx]
    lens = lengths[idx]
    valid = torch.arange(x.shape[1], device=DEVICE)[None] < lens[:, None]
    rule = rule_eye[task_id[idx]][:, None] * valid[..., None]
    x = torch.cat([x * valid[..., None], rule], dim=-1)
    return x, labels[idx], lens, task_id[idx]


# ------------------------------------------ model

model = LeakyRNN(n_in, N_HIDDEN, n_out, alpha, SIGMA_REC, ACTIVATION).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
print(sum(p.numel() for p in model.parameters()), "params")


def compute_loss(logits, y, hidden, mask):
    """Weighted cross-entropy plus rate and recurrent-weight penalties."""
    ce = F.cross_entropy(
        logits.reshape(-1, n_out), y.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    ).view_as(y)
    w = torch.where(y > 0, RESP_WEIGHT, 1.0) * (y != IGNORE_INDEX)
    task_loss = (ce * w).sum() / w.sum()
    rate_loss = (hidden.pow(2) * mask[..., None]).sum() / (mask.sum() * hidden.shape[-1])
    weight_loss = model.w_rec.weight.pow(2).mean()
    return task_loss + L2_RATE * rate_loss + L2_WEIGHT * weight_loss

@torch.no_grad()
def quick_accuracy(n=ACC_BATCH):
    """Trial accuracy on a random validation sample: fixation held and last step correct."""
    was_training = model.training
    model.eval()
    idx = val_idx[torch.randint(len(val_idx), (n,), device=DEVICE)]
    x, y, lens, _ = make_batch(idx)
    logits, _ = model(x)
    pred = logits.argmax(-1)
    fix_ok = ~((pred != 0) & (y == 0)).any(dim=1)
    rows = torch.arange(n, device=DEVICE)
    resp_ok = pred[rows, lens - 1] == y[rows, lens - 1]
    model.train(was_training)
    return (fix_ok & resp_ok).float().mean().item()

# ----------------------------------------- projection

probe_x, probe_labels, probe_lens, _ = make_batch(probe_idx)
projector = ProbeProjector(
    model, probe_x, probe_labels, probe_lens, ridge=L2_WEIGHT,
    n_ring=bank["n_ring"], smooth=SMOOTH, z_mode="time", z_scale=Z_SCALE
)
projector.update(refit=True)

# get the trials from the probe set that matches the extension type (base, int, seq)
probe_ext = extension[probe_idx].cpu().numpy()
panel_sel = {name: np.flatnonzero(probe_ext == e) for e, name in enumerate(extension_names)}
init_positions = projector.line_data()
init_colors = projector.colors().cpu().numpy()


# ----------------------------------------- visualization

figure = fpl.Figure(shape=(1,3),
                    size=(1400, 550),
                    names=["base", "int", "seq"],
                    cameras="3d",
                    controller_types="orbit",
                    controller_ids="sync")


for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False
    s.toolbar = False


phi = np.linspace(0, 2 * np.pi, 200, dtype=np.float32)
ring = np.column_stack([RING_RADIUS * np.cos(phi), RING_RADIUS * np.sin(phi), np.zeros_like(phi)])

graphics = {}
for name in ["base", "int", "seq"]:
    s = figure[name]
    for z in (0.0, Z_SCALE):
        ring_z = np.column_stack([
            RING_RADIUS * np.cos(phi), RING_RADIUS * np.sin(phi), np.full_like(phi, z)
        ])
        s.add_line(ring_z, colors="w", thickness=2.0)
    for a in np.arange(0, 2 * np.pi, np.pi / 2):
        guide = np.array([[np.cos(a), np.sin(a), 0], [np.cos(a), np.sin(a), Z_SCALE]], dtype=np.float32)
        s.add_line(guide, colors="w", thickness=2.0)
    sel = panel_sel[name]
    graphics[name] = s.add_line_collection(
        data=[init_positions[p] for p in sel],
        colors=[init_colors[p] for p in sel],
        thickness=1.0,
    )
    s.auto_scale(maintain_aspect=True)


#add_color_wheel(figure["base"], VIEW_LIM)

TITLE_TEXT = "step 0"

figure.show(autoscale=False)

# ------------------------------------------ shared buffers (mechiviz)

buffers = {}
for name, collection in graphics.items():
    bufs = []
    for graphic in collection.graphics:
        buf = mv.TorchTensorBuffer(shape=graphic.data.value.shape)
        buf.buffer = graphic.data.buffer
        bufs.append(buf)
    buffers[name] = bufs

# ------------------------------------------- training loop

state = {"step": 0}


def train_step():
    batch_idx = train_idx[torch.randint(len(train_idx), (BATCH,), device=DEVICE)]
    x, y, lens, _ = make_batch(batch_idx)
    mask = torch.arange(x.shape[1], device=DEVICE)[None] < lens[:, None]

    logits, hidden = model(x)
    loss = compute_loss(logits, y, hidden, mask)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return loss.detach()


# ------------------------------------------- animation function

def animate():
    global TITLE_TEXT
    if state["step"] >= N_STEPS:
        return

    for _ in range(STEPS_PER_FRAME):
        loss = train_step()
        state["step"] += 1

    refit = state["step"] % REFIT_EVERY < STEPS_PER_FRAME
    positions = projector.update(refit=refit)

    for name, bufs in buffers.items():
        sel = panel_sel[name]
        for b, p in zip(bufs, sel):
            b.update(positions[p], synchronize=False)

    if state["step"] % ACC_EVERY < STEPS_PER_FRAME:
        state["acc"] = quick_accuracy()

    TITLE_TEXT = f"step {state['step']:,}   log10 loss {torch.log10(loss).item():.2f}   R² {projector.r2:.2f}   accuracy {state.get('acc', 0):.2f}"


figure.add_animations(animate)

# --------------------------------------------  gui
class TitleBar(ImguiWindow):
    def update(self):
        imgui.text(TITLE_TEXT)


# make GUI instance
gui = TitleBar()

# add it to the top edge of the figure
figure.add_imgui_window(
    gui,
    location="top",
    size=35,
    title=None,
    window_flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize,
)


if __name__ == "__main__":
    fpl.loop.run()

