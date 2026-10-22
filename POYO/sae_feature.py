from pathlib import Path

import mechiviz as mv
import fastplotlib as fpl
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui
import torch
import numpy as np

from sae import TopKSAE
from utils import get_trials, load_model, make_batches, run_decoder, to_paths

CKPT = Path.home() / "repos/torch_brain/examples/poyo/outputs/2026-10-05/14-33-20/checkpoints/best.pt"
DATA_ROOT = Path.home() / "brainsets/processed"
DEVICE = torch.device("cuda")
SAE_PATH = "sae.pt"

OUT_DT = 0.01
OUT_TS = np.arange(0, 1.0, OUT_DT, dtype=np.float32)
PRE = 0.25  # window starts 0.25 s before reach onset (get_trials default)

# ------------- demo parameters
TITLE_TEXT = ""

# ----------------------------- model + SAE

model, ds = load_model(CKPT, DATA_ROOT, DEVICE)

ckpt = torch.load(SAE_PATH)
sae = TopKSAE(model.readout.in_features, ckpt["n_latents"], ckpt["k"]).to(DEVICE)
sae.load_state_dict(ckpt["sae"])

# ----------------------------- data: decoder states + SAE activations on reach-aligned trials

trials, targets = get_trials(ds)
batches = make_batches(model, ds, trials, OUT_TS, DEVICE)
dec, vel = run_decoder(model, batches)
PATHS = to_paths(vel, OUT_DT).cpu().numpy()  # (N, T, 2) predicted reach paths

with torch.no_grad():
    Z = sae.encode(dec)  # (N, T, n_latents), GPU
    OUT_DIR = (model.readout.weight @ sae.dec.weight).T.cpu().numpy()  # (n_latents, 2) velocity push

N_TRIALS, T = Z.shape[0], Z.shape[1]
_t = OUT_TS - PRE

# averaging matrix: (8, N) per target, rows sum to 1
TARGET_W = torch.zeros(8, N_TRIALS, device=DEVICE)
TARGET_W[torch.as_tensor(targets), torch.arange(N_TRIALS)] = 1.0
TARGET_W /= TARGET_W.sum(dim=1, keepdim=True).clamp_min(1)

_reach = np.median(np.linalg.norm(PATHS, axis=-1).max(axis=1))
_angles = np.arange(8) * np.pi / 4
TARGET_XY = _reach * np.column_stack([np.cos(_angles), np.sin(_angles)])

# ----------------------------- data: feature map (live features only)

freq = (Z > 0).float().mean(dim=(0, 1)).cpu().numpy()
LIVE = np.flatnonzero(freq > 0)

push = np.linalg.norm(OUT_DIR[LIVE], axis=1)
angle = np.arctan2(OUT_DIR[LIVE, 1], OUT_DIR[LIVE, 0])

# radius on a log scale so the small features spread out instead of piling at the center
_lp = np.log(push)
radius = 0.3 + 0.7 * (_lp - _lp.min()) / (_lp.max() - _lp.min())
FEAT_XY = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
FEAT_SIZES = 4 + 16 * np.sqrt(freq[LIVE] / freq[LIVE].max())

SELECTED = LIVE[np.argmax(push)]  # start on the biggest velocity feature
print(f"{N_TRIALS} trials, {len(LIVE)} live features")

# ----------------------------- colors
_PD_LUT = fpl.utils.make_colors(361, "hsv")[:360]
TARGET_COLORS = fpl.utils.make_colors(9, "hsv")[:8]
VIRIDIS = torch.as_tensor(fpl.utils.make_colors(256, "viridis"), device=DEVICE, dtype=torch.float32)


def angle_colors(angles, alpha=1.0):
    # 0 rad = red, same hue as target 0
    idx = ((np.mod(angles, 2 * np.pi) / (2 * np.pi)) * 360).astype(int) % 360
    c = _PD_LUT[idx].copy()
    c[:, -1] = alpha
    return c

# ----------------------------- figure

# (x0, x1, y0, y1) as fractions of the canvas
extents = [
    (0, 0.34, 0, 1),
    (0.34, 0.67, 0, 1),
    (0.67, 1, 0, 1),
]
figure = fpl.Figure(extents=extents,
                    size=(1800, 650),
                    names=["features", "activation over the reach", "where it fires on the reach"],
                    controller_types=["panzoom", "panzoom", "panzoom"])
figure.canvas.set_title("POYO SAE Feature Dashboard")

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False

figure["features"].tooltip.enabled = True


sp = figure["features"]
sp.add_scatter(FEAT_XY, colors=angle_colors(angle), sizes=FEAT_SIZES, size_space="screen", name="features")
sp["features"].tooltip_format = lambda pick_info: (
    f"feature {LIVE[pick_info['vertex_index']]}\n"
    f"direction {np.degrees(angle[pick_info['vertex_index']]):.0f} deg\n"
    f"push {push[pick_info['vertex_index']]:.3f}\n"
    f"fires {freq[LIVE[pick_info['vertex_index']]]:.0%}"
)
_sel = np.flatnonzero(LIVE == SELECTED)[0]
sp.add_scatter(FEAT_XY[_sel:_sel + 1], colors="white", sizes=28, size_space="screen", alpha=0.4, name="selected")
sp.camera.show_rect(-1.15, 1.15, -1.15, 1.15)


# activation over the reach: one line per target, x = time from reach onset, y = activation scaled to max
sp = figure["activation over the reach"]
sp.add_line(np.array([[0, 0], [0, 1.05]], dtype=np.float32), colors="gray", alpha=0.5, name="onset")
sp.add_line_collection(
    [np.column_stack([_t, np.zeros(T)]).astype(np.float32) for _ in range(8)],
    colors=list(TARGET_COLORS),
    thickness=2,
    name="lines",
)
sp.camera.maintain_aspect = False
sp.camera.show_rect(_t[0] - 0.05, _t[-1] + 0.05, -0.05, 1.1)


# where it fires on the reach: every trial's reach path, each vertex colored by activation
sp = figure["where it fires on the reach"]
paths = sp.add_line_collection(list(PATHS), colors="gray", thickness=1.5, name="paths")
_gray = np.tile(np.array([[0.5, 0.5, 0.5, 1.0]], dtype=np.float32), (T, 1))
for g in paths.graphics:
    g.colors = _gray  # per-vertex colors, so each line has a (T, 4) colors buffer
sp.add_scatter(TARGET_XY, colors=TARGET_COLORS, sizes=18, size_space="screen", alpha=0.6, name="targets")

# ------------------------------- shared buffers

time_buffers = []
path_color_buffers = []


def make_buffer(g):
    buf = mv.TorchTensorBuffer(shape=g.data.value.shape)
    buf.buffer = g.data.buffer
    return buf


def make_color_buffer(g):
    buf = mv.TorchTensorBuffer(shape=g.colors.value.shape)
    buf.buffer = g.colors.buffer
    return buf


def adopt_buffers():
    global time_buffers, path_color_buffers
    time_buffers = [make_buffer(g) for g in figure["activation over the reach"]["lines"].graphics]
    path_color_buffers = [make_color_buffer(g) for g in figure["where it fires on the reach"]["paths"].graphics]


T_AXIS = torch.as_tensor(_t, device=DEVICE)

adopt_buffers()


@torch.no_grad()
def update_graphics():
    global TITLE_TEXT

    a = Z[:, :, SELECTED]  # (N, T) activation of the selected feature

    # activation over the reach: mean over trials of each target
    tc = TARGET_W @ a  # (8, T)
    tc = tc / tc.max().clamp_min(1e-6)
    pts = torch.stack([T_AXIS.expand(8, -1), tc, torch.zeros_like(tc)], dim=-1)  # (8, T, 3)
    for i, buf in enumerate(time_buffers):
        buf.update(pts[i], synchronize=True)

    # where it fires: color every vertex of every reach path by activation
    idx = (a / a.max().clamp_min(1e-6) * 255).long()
    colors = VIRIDIS[idx]  # (N, T, 4)
    for i, buf in enumerate(path_color_buffers):
        buf.update(colors[i], synchronize=True)

    i = np.flatnonzero(LIVE == SELECTED)[0]
    TITLE_TEXT = (
        f"feature {SELECTED}   direction {np.degrees(angle[i]):.0f} deg   "
        f"push {push[i]:.3f}   fires {freq[SELECTED]:.0%}"
    )


update_graphics()

# -------------------------------- click to select a feature


@figure["features"]["features"].add_event_handler("click")
def click_feature(ev):
    global SELECTED
    i = ev.pick_info["vertex_index"]
    SELECTED = LIVE[i]
    figure["features"]["selected"].data[0, :2] = FEAT_XY[i]
    update_graphics()


# ------------- gui
class TitleBar(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        imgui.text(TITLE_TEXT)


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