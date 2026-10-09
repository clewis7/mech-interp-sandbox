from pathlib import Path

import mechiviz as mv
import fastplotlib as fpl
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui
import torch
import numpy as np

from transfer import (
    HeldOutDataset,
    extend_for_session,
    make_optimizer,
    make_window_pool,
    r2,
    train_step,
    true_velocity,
)
from utils import (
    fit_pca,
    get_trials,
    load_model,
    make_batches,
    run_decoder,
    to_paths,
)

CKPT = Path.home() / "repos/torch_brain/examples/poyo/outputs/2026-10-05/14-33-20/checkpoints/best.pt"
DATA_ROOT = Path.home() / "brainsets/processed"
DEVICE = torch.device("cuda")
NEW_RID = "j_20160405_center_out_reaching" #"c_20131220_center_out_reaching"  # or

OUT_DT = 0.01
OUT_TS = np.arange(0, 1.0, OUT_DT, dtype=np.float32)

# ------------- training parameters
LR = 3e-4  # lower than 1e-3 so convergence is slow enough to watch
N_STEPS = 2_000

# ------------- demo parameters
TITLE_TEXT = "step 0"

# ----------------------------- model

model, ds = load_model(CKPT, DATA_ROOT, DEVICE)
SESSIONS = ds.recording_ids

# reference (trained sessions): decoder-state PCA basis, built before extending vocab
ref_trials, ref_targets = get_trials(ds)
ref_batches = make_batches(model, ds, ref_trials, OUT_TS, DEVICE)
ref_dec, _ = run_decoder(model, ref_batches)
mu, pcs = fit_pca(ref_dec)

# held-out session: extend vocab (new rows random), freeze backbone
new_ds = HeldOutDataset(str(DATA_ROOT), [NEW_RID])
new_unit_idx, new_sess_idx = extend_for_session(model, new_ds)

# fixed eval trials for the new session (test split; transfer trains on train split)
eval_trials, eval_targets = get_trials(new_ds, n_per=10, split="test")
eval_batches = make_batches(model, new_ds, eval_trials, OUT_TS, DEVICE)
eval_true_vel = true_velocity(new_ds, eval_trials, OUT_TS, DEVICE)
print(f"{len(new_unit_idx)} new units, {len(eval_trials)} eval trials")

# training windows (train split), tokenized once and kept on GPU
pool = make_window_pool(model, new_ds, DEVICE)
optim = make_optimizer(model, lr=LR)

# starting values of the new rows, for reset
INIT_UNIT = model.unit_emb.weight[new_unit_idx].detach().clone()
INIT_SESS = model.session_emb.weight[new_sess_idx].detach().clone()

step = 0

# ----------------------------- data

# background: trained sessions, averaged per (session, target)
ref_rids = np.array([rid for rid, _, _ in ref_trials])
ref_proj = (ref_dec - mu) @ pcs
bg_traj, bg_targets = [], []
for rid in SESSIONS:
    for tg in range(8):
        m = torch.as_tensor((ref_rids == rid) & (ref_targets == tg), device=DEVICE)
        if m.any():
            bg_traj.append(ref_proj[m].mean(0).cpu().numpy())
            bg_targets.append(tg)
bg_targets = np.array(bg_targets)

# new session: current model (untrained new embeddings)
new_dec, new_vel = run_decoder(model, eval_batches)
NEW_TRAJ = (new_dec - mu) @ pcs  # (N, T, 3), GPU
NEW_PATHS = to_paths(new_vel, OUT_DT)  # (N, T, 2), GPU
TRUE_PATHS = to_paths(eval_true_vel, OUT_DT).cpu().numpy()  # (N, T, 2)

_reach = np.linalg.norm(TRUE_PATHS, axis=-1).max(axis=1)
_angles = np.arange(8) * np.pi / 4
TARGET_XY = np.median(_reach) * np.column_stack([np.cos(_angles), np.sin(_angles)])

# ----------------------------- colors
TARGET_COLORS = fpl.utils.make_colors(9, "hsv")[:8]

BG_COLORS = TARGET_COLORS[bg_targets].copy()
BG_COLORS[:, -1] = 0.15
NEW_TRAJ_COLORS = list(TARGET_COLORS[eval_targets])

# ----------------------------- figure

figure = fpl.Figure(shape=(1, 2),
                    size=(1400, 750),
                    names=["decoder trajectories", "reach output"],
                    controller_types=["orbit", "panzoom"])
figure.canvas.set_title("POYO Transfer Learning")

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False


sp = figure["decoder trajectories"]
sp.add_line_collection(bg_traj, colors=list(BG_COLORS), thickness=1.0, name="background")
sp.add_line_collection(list(NEW_TRAJ.cpu().numpy()), colors=NEW_TRAJ_COLORS, thickness=1.5, name="new")


sp = figure["reach output"]
sp.add_line_collection(list(TRUE_PATHS), colors="gray", alpha=0.3, thickness=1.0, name="true")
sp.add_line_collection(list(NEW_PATHS.cpu().numpy()), colors=NEW_TRAJ_COLORS, thickness=1.5, name="pred")
sp.add_scatter(TARGET_XY, colors=TARGET_COLORS, sizes=25, size_space="screen", name="targets")
sp.add_scatter(np.zeros((1, 2)), colors="gray", sizes=10, size_space="screen", name="center")

# ------------------------------- shared buffers

traj_buffers = []
path_buffers = []


def make_buffer(g):
    buf = mv.TorchTensorBuffer(shape=g.data.value.shape)
    buf.buffer = g.data.buffer
    return buf


def adopt_buffers():
    global traj_buffers, path_buffers
    traj_buffers = [make_buffer(g) for g in figure["decoder trajectories"]["new"].graphics]
    path_buffers = [make_buffer(g) for g in figure["reach output"]["pred"].graphics]


def pad_z(x):
    """(..., 2) -> (..., 3) with z = 0, to match the graphics' vertex buffers."""
    return torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)


@torch.no_grad()
def update_graphics():
    global TITLE_TEXT

    # decoder trajectories + reach output on the fixed eval trials
    dec, vel = run_decoder(model, eval_batches)
    traj = (dec - mu) @ pcs
    paths = pad_z(to_paths(vel, OUT_DT))
    for i, buf in enumerate(traj_buffers):
        buf.update(traj[i], synchronize=True)
    for i, buf in enumerate(path_buffers):
        buf.update(paths[i], synchronize=True)

    TITLE_TEXT = f"step {step:,}   eval R² {r2(vel, eval_true_vel).item():.3f}"


def reset():
    global step, optim
    with torch.no_grad():
        model.unit_emb.weight[new_unit_idx] = INIT_UNIT
        model.session_emb.weight[new_sess_idx] = INIT_SESS
    optim = make_optimizer(model, lr=LR)
    step = 0
    update_graphics()

# -------------------------------- training loop

adopt_buffers()
update_graphics()

def animate():
    global step

    if step >= N_STEPS:
        return

    X, Y = pool[torch.randint(len(pool), (1,)).item()]
    train_step(model, optim, X, Y)
    step += 1

    update_graphics()


figure.add_animations(animate)


# ------------- gui
class TitleBar(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        imgui.text(f"{NEW_RID.replace('_center_out_reaching', '')}   {TITLE_TEXT}")
        imgui.same_line()

        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - 100)
        if imgui.button("Reset"):
            reset()


# make GUI instance
gui = TitleBar()

# add it to the top edge of the figure
figure.add_imgui_window(
    gui,
    location="top",
    size=40,
    title=None,
    window_flags=imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_scrollbar,
)

figure.show()


if __name__ == "__main__":
    fpl.loop.run()