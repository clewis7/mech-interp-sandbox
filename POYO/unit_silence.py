from pathlib import Path

import mechiviz as mv
import fastplotlib as fpl
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui
import torch
import numpy as np

from transfer import r2, true_velocity
from utils import (
    fit_pca,
    get_trials,
    load_model,
    make_batches,
    run_decoder,
    to_paths,
    unit_preferred_directions,
)

CKPT = Path.home() / "repos/torch_brain/examples/poyo/outputs/2026-10-05/14-33-20/checkpoints/best.pt"
DATA_ROOT = Path.home() / "brainsets/processed"
DEVICE = torch.device("cuda")
SEED = 0

OUT_DT = 0.01
OUT_TS = np.arange(0, 1.0, OUT_DT, dtype=np.float32)

# ------------- demo parameters
TITLE_TEXT = ""
WEDGE_CENTER = 90.0  # degrees, set by clicking a target or a unit
WEDGE_WIDTH = 60.0  # degrees, set by the slider
ACTIVE = False  # nothing silenced until a direction is clicked

# ----------------------------- model

model, ds = load_model(CKPT, DATA_ROOT, DEVICE)
SESSIONS = ds.recording_ids

# fixed trial set from the 16 trained sessions (valid split)
trials, targets = get_trials(ds)
batches = make_batches(model, ds, trials, OUT_TS, DEVICE)
true_vel = true_velocity(ds, trials, OUT_TS, DEVICE)

# intact model: decoder-state PCA basis, reference outputs
ref_dec, ref_vel = run_decoder(model, batches)
mu, pcs = fit_pca(ref_dec)
REF_R2 = r2(ref_vel, true_vel).item()
print(f"{len(trials)} trials, intact R² {REF_R2:.3f}")

# trained units: vocab token and preferred direction, in session order
unit_tokens, unit_pd = [], []
for rid in SESSIONS:
    unit_tokens.append(np.array(model.unit_emb.tokenizer(list(ds.get_recording(rid).units.id))))
    unit_pd.append(unit_preferred_directions(ds, rid))
unit_tokens = np.concatenate(unit_tokens)
unit_pd = np.concatenate(unit_pd)
N_UNITS = len(unit_tokens)

# ----------------------------- ablation mask

# True = keep; indexed by unit vocab token
UNIT_MASK = torch.ones(model.unit_emb.weight.shape[0], dtype=torch.bool, device=DEVICE)


def silence_units(module, args, kwargs):
    # drop every spike token (and start/end token) from silenced units
    kwargs["input_mask"] = kwargs["input_mask"] & UNIT_MASK[kwargs["input_unit_index"]]
    return args, kwargs


model.register_forward_pre_hook(silence_units, with_kwargs=True)


def wedge_selection():
    """Indices into the trained-unit list of the units to silence."""
    if not ACTIVE:
        return np.array([], dtype=int)
    diff = np.angle(np.exp(1j * (unit_pd - np.radians(WEDGE_CENTER))))  # wrapped to [-pi, pi]
    return np.flatnonzero(np.abs(diff) <= np.radians(WEDGE_WIDTH) / 2)

# ----------------------------- data

REF_TRAJ = ((ref_dec - mu) @ pcs).cpu().numpy()  # (N, T, 3)
REF_PATHS = to_paths(ref_vel, OUT_DT).cpu().numpy()  # (N, T, 2)

_reach = np.linalg.norm(REF_PATHS, axis=-1).max(axis=1)
_angles = np.arange(8) * np.pi / 4
TARGET_XY = np.median(_reach) * np.column_stack([np.cos(_angles), np.sin(_angles)])

# units on a ring at their preferred direction, small radial jitter so they don't stack
_r = 1.0 + np.random.default_rng(SEED).uniform(-0.08, 0.08, N_UNITS)
UNIT_XY = np.column_stack([_r * np.cos(unit_pd), _r * np.sin(unit_pd)])

# ----------------------------- colors
_PD_LUT = fpl.utils.make_colors(361, "hsv")[:360]
TARGET_COLORS = fpl.utils.make_colors(9, "hsv")[:8]


def angle_colors(angles, alpha=1.0):
    # 0 rad = red, same hue as target 0, so a wedge's color matches its reach spoke
    idx = ((np.mod(angles, 2 * np.pi) / (2 * np.pi)) * 360).astype(int) % 360
    c = _PD_LUT[idx].copy()
    c[:, -1] = alpha
    return c


TRIAL_COLORS = list(TARGET_COLORS[targets])

# ----------------------------- figure

# (x0, x1, y0, y1) as fractions of the canvas: units on the left, ref on top, ablated below
extents = [
    (0, 0.34, 0, 1),
    (0.34, 0.67, 0, 0.5),
    (0.67, 1, 0, 0.5),
    (0.34, 0.67, 0.5, 1),
    (0.67, 1, 0.5, 1),
]
figure = fpl.Figure(extents=extents,
                    size=(1500, 900),
                    names=["units", "ref traj", "ref output", "ablate traj", "ablate output"],
                    controller_types=["panzoom", "orbit", "panzoom", "orbit", "panzoom"])
figure.canvas.set_title("POYO Unit Ablation")

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False

# ref and ablated panels move together
figure["ablate traj"].controller = figure["ref traj"].controller
figure["ablate output"].controller = figure["ref output"].controller


sp = figure["units"]
sp.add_scatter(UNIT_XY, colors=angle_colors(unit_pd), sizes=5, size_space="screen", name="units")
sp.add_line(np.zeros((3, 2), dtype=np.float32), colors="white", thickness=2, name="wedge")
sp["wedge"].visible = False
sp.camera.show_rect(-1.3, 1.3, -1.3, 1.3)

for name in ["ref traj", "ablate traj"]:
    figure[name].add_line_collection(list(REF_TRAJ), colors=TRIAL_COLORS, thickness=1.5, name="traj")

for name in ["ref output", "ablate output"]:
    sp = figure[name]
    sp.add_line_collection(list(REF_PATHS), colors=TRIAL_COLORS, thickness=1.5, name="paths")
    sp.add_scatter(TARGET_XY, colors=TARGET_COLORS, sizes=25, size_space="screen", name="targets")
    sp.add_scatter(np.zeros((1, 2)), colors="gray", sizes=10, size_space="screen", name="center")

# ------------------------------- shared buffers

traj_buffers = []
path_buffers = []
pending_adopt = True
pending_update = True


def make_buffer(g):
    buf = mv.TorchTensorBuffer(shape=g.data.value.shape)
    buf.buffer = g.data.buffer
    return buf


def adopt_buffers():
    global traj_buffers, path_buffers, pending_adopt
    traj_buffers = [make_buffer(g) for g in figure["ablate traj"]["traj"].graphics]
    path_buffers = [make_buffer(g) for g in figure["ablate output"]["paths"].graphics]
    pending_adopt = False


def pad_z(x):
    """(..., 2) -> (..., 3) with z = 0, to match the graphics' vertex buffers."""
    return torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)


def update_wedge_line():
    # V shape from the center out along the two wedge edges
    a0 = np.radians(WEDGE_CENTER - WEDGE_WIDTH / 2)
    a1 = np.radians(WEDGE_CENTER + WEDGE_WIDTH / 2)
    pts = 1.25 * np.array([[np.cos(a0), np.sin(a0)], [0, 0], [np.cos(a1), np.sin(a1)]])
    figure["units"]["wedge"].data[:, :2] = pts
    figure["units"]["wedge"].visible = ACTIVE


@torch.no_grad()
def update_graphics():
    global TITLE_TEXT

    # set the mask from the wedge
    sel = wedge_selection()
    UNIT_MASK.fill_(True)
    UNIT_MASK[torch.as_tensor(unit_tokens[sel], device=DEVICE)] = False

    # dim silenced units on the ring, draw the wedge
    alphas = np.ones(N_UNITS, dtype=np.float32)
    alphas[sel] = 0.1
    figure["units"]["units"].colors[:, -1] = alphas
    update_wedge_line()

    # ablated decoder trajectories + reach output
    dec, vel = run_decoder(model, batches)
    traj = (dec - mu) @ pcs
    paths = pad_z(to_paths(vel, OUT_DT))
    for i, buf in enumerate(traj_buffers):
        buf.update(traj[i], synchronize=True)
    for i, buf in enumerate(path_buffers):
        buf.update(paths[i], synchronize=True)

    if not ACTIVE:
        TITLE_TEXT = f"click a target or a unit to silence that direction   R² {REF_R2:.3f}"
    else:
        TITLE_TEXT = (
            f"{len(sel)} / {N_UNITS} units silenced (wedge at {WEDGE_CENTER:.0f} deg)   "
            f"R² {r2(vel, true_vel).item():.3f} (intact {REF_R2:.3f})"
        )

# -------------------------------- click to pick a direction


def set_center(angle_deg):
    global WEDGE_CENTER, ACTIVE, pending_update
    WEDGE_CENTER = angle_deg
    ACTIVE = True
    pending_update = True


def click_target(ev):
    set_center(ev.pick_info["vertex_index"] * 45.0)


for name in ["ref output", "ablate output"]:
    figure[name]["targets"].add_event_handler(click_target, "click")


@figure["units"]["units"].add_event_handler("click")
def click_unit(ev):
    set_center(np.degrees(unit_pd[ev.pick_info["vertex_index"]]))

# -------------------------------- update


def animate():
    global pending_update

    if pending_adopt:
        adopt_buffers()

    if pending_update:
        update_graphics()
        pending_update = False


figure.add_animations(animate)


# ------------- gui
class TitleBar(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        global WEDGE_WIDTH, ACTIVE, pending_update
        imgui.text(TITLE_TEXT)
        imgui.same_line()

        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - 360)
        imgui.text("Wedge Width:")
        imgui.same_line()
        imgui.set_next_item_width(200)
        changed, WEDGE_WIDTH = imgui.slider_float("##wedge width", WEDGE_WIDTH, 10.0, 180.0, "%.0f deg")
        if changed and ACTIVE:
            pending_update = True

        imgui.same_line()
        if imgui.button("Reset"):
            ACTIVE = False
            pending_update = True


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