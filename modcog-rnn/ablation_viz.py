import mechiviz as mv

import copy
import os

import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib.ui import ImguiWindow
import torch
import wgpu

from model import LeakyRNN
from generate_data import generate_task_data
from utils import get_checkpointed_tasks, PCAProjector, trajectory_colors, ablation_metrics

# ------------------ setup

DEVICE = "cuda"
N_PROBE = 256
SEED = 0
TITLE_TEXT = ""

TRAJ_SCALE = 1.0
VIEW_LIM = 3.0
THICKNESS = 1.5

task = "go"
reference = alternate = None
probe_x = probe_y = lens = colors = None
w_orig = mask = w_live = None
vmax = ref_acc = None
ref_proj = alt_proj = None
lines = {}
weights = rs = w_tex = None
alt_buffers = []

def load_task(name):
    """Load a checkpoint and rebuild models, probe set and projections."""
    global task, reference, alternate, probe_x, probe_y, lens, colors
    global w_orig, mask, w_live, vmax, ref_proj, alt_proj, ref_acc, TITLE_TEXT

    ckpt = torch.load(f"checkpoints/{name}.ckpt", map_location=DEVICE, weights_only=False)
    task = ckpt["task"]

    reference = LeakyRNN(**ckpt["config"]).to(DEVICE)
    reference.load_state_dict(ckpt["model"])
    reference.eval().requires_grad_(False)
    alternate = copy.deepcopy(reference)

    data = generate_task_data(task, N_PROBE, device=DEVICE, seed=SEED)
    lens = data["lengths"]
    valid = torch.arange(data["obs"].shape[1], device=DEVICE)[None] < lens[:, None]
    probe_x = data["obs"] * valid[..., None]
    probe_y = data["labels"]
    colors = trajectory_colors(probe_y.cpu().numpy(), lens.cpu().numpy(), data["n_ring"])

    w_orig = reference.w_rec.weight.detach().clone()
    mask = torch.ones_like(w_orig)
    w_live = w_orig.clone()
    vmax = float(abs(w_orig).max())

    ref_proj = PCAProjector(reference, probe_x, lens, scale=TRAJ_SCALE)
    ref_proj.update(refit=True)

    alt_proj = PCAProjector(alternate, probe_x, lens, scale=TRAJ_SCALE)
    alt_proj.basis = ref_proj.basis
    alt_proj.center = ref_proj.center
    alt_proj.update(refit=False)

    m = ablation_metrics(alternate, probe_x, probe_y, lens,
                         ref_proj.positions, alt_proj.positions)
    ref_acc = m["acc"]

    TITLE_TEXT = (
        f"acc {ref_acc:.2f} : {m['acc']:.2f}   "
        f"ring err {m['ring_err']:.1f}   drift {m['drift']:.2f}   "
        f"cut {int((mask == 0).sum())}/{mask.numel()}"
    )

load_task(task)

# ------------------ viz

figure = fpl.Figure(shape=(1,3),
                    size=(1200, 450),
                    names=["reference model", "ablation model", "weights"],
                    cameras=["3d", "3d", "2d"],
                    controller_types=["orbit", "orbit", "panzoom"])
figure.canvas.set_title("Post-Hoc Weight Ablation Viz")

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False

figure["reference model"].controller = figure["ablation model"].controller

def build_graphics():
    """Recreate every graphic. Trial length and n_hidden change between tasks."""
    global weights, rs

    for name, proj in (("reference model", ref_proj), ("ablation model", alt_proj)):
        s = figure[name]
        if name in lines:
            s.remove_graphic(lines[name])
        pos = proj.positions.cpu().numpy()
        lines[name] = s.add_line_collection(
            data=[pos[p] for p in range(len(pos))],
            colors=[colors[p] for p in range(len(colors))],
            thickness=THICKNESS,
        )

    s = figure["weights"]
    if weights is not None:
        s.remove_graphic(weights)
    if rs is not None:
        s.remove_graphic(rs)
    weights = s.add_image(
        data=w_live.cpu().numpy(),
        cmap="bwr", vmin=-vmax, vmax=vmax,
        texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    rs = weights.add_rectangle_selector()

build_graphics()

# ------------------ shared buffers

def adopt_buffers():
    """Link the new graphics to shared GPU buffers; needs one render first."""
    global alt_buffers, w_tex

    alt_buffers = []
    for g in lines["ablation model"].graphics:
        buf = mv.TorchTensorBuffer(shape=g.data.value.shape)
        buf.buffer = g.data.buffer
        alt_buffers.append(buf)

    w_tex = mv.TorchTensorTexture(shape=w_live.shape)
    w_tex.texture = weights.data.buffer[0, 0]

adopt_buffers()

# ------------------ funcs

def apply_mask():
    global TITLE_TEXT
    torch.mul(w_orig, mask, out=w_live)
    alternate.w_rec.weight.copy_(w_live)

    alt_proj.update(refit=False)
    for p, buf in enumerate(alt_buffers):
        buf.update(alt_proj.positions[p], synchronize=True)

    w_tex.update(w_live)

    m = ablation_metrics(alternate, probe_x, probe_y, lens,
                         ref_proj.positions, alt_proj.positions)
    TITLE_TEXT = (
        f"acc {ref_acc:.2f} : {m['acc']:.2f}   "
        f"ring err {m['ring_err']:.1f}   drift {m['drift']:.2f}   "
        f"cut {int((mask == 0).sum())}/{mask.numel()}"
    )

# ------------------- UI

class TitleBar(ImguiWindow):
    def __init__(self):
        super().__init__()
        self._tasks = get_checkpointed_tasks()
        self._task_idx = self._tasks.index("go")
        self._ablation_idx = 0

    def update(self):
        global TITLE_TEXT
        imgui.text(TITLE_TEXT)
        imgui.same_line()
        imgui.text("Task:")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed, self._task_idx = imgui.combo("##task", self._task_idx, self._tasks)
        if changed:
            load_task(self._tasks[self._task_idx])
            build_graphics()
            adopt_buffers()
        imgui.same_line()
        if imgui.button("Ablate"):
            # get selection
            x_idxs, y_idxs = rs.get_selected_indices()
            mask[x_idxs[0]:x_idxs[-1] + 1, y_idxs[0]:y_idxs[-1] + 1] = 0
            apply_mask()

        imgui.same_line()
        if imgui.button("Reset"):
            mask.fill_(1)
            apply_mask()


# add it to the top edge of the figure
figure.add_imgui_window(
    TitleBar(),
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