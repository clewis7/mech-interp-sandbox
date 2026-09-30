import mechiviz as mv

import copy
import math

import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib.ui import ImguiWindow
import torch
import wgpu
import numpy as np

from model import LeakyRNN
from generate_data import generate_task_data
from utils import get_checkpointed_tasks, trajectory_colors, ablation_metrics

# ------------------ setup

DEVICE = "cuda"
N_PROBE = 256
SEED = 0
TITLE_TEXT = ""

VIEW_LIM = 0.8
THICKNESS = 1.5
RING_RADIUS = 1.0

task = "go"
reference = alternate = None
probe_x = probe_y = lens = colors = None
freeze_idx = ring_xy = None
w_orig = mask = w_live = None
vmax = ref_acc = None
ref_pos = alt_pos = None
lines = {}
weights = rs = w_tex = None
alt_buffers = []


@torch.no_grad()
def output_positions(model, out):
    """Population vector of the output distribution at every timestep.

    Sums the softmax probability of each ring direction times that direction's
    unit vector, so the path sits near the origin while the model is holding
    fixation and swings out toward the reported direction during the response.
    Length is confidence: a split distribution gives a short vector.

    Writes into `out` (P, T, 3) in place.
    """
    logits, _ = model(probe_x)
    p = logits.softmax(-1)[..., 1:]  # drop the fixation unit
    vec = p @ ring_xy  # (P, T, 2)
    vec = vec.gather(1, freeze_idx[..., None].expand(-1, -1, 2))
    out[..., :2] = vec
    return out


def load_task(name):
    """Load a checkpoint and rebuild models, probe set and output trajectories."""
    global task, reference, alternate, probe_x, probe_y, lens, colors
    global freeze_idx, ring_xy, w_orig, mask, w_live, vmax
    global ref_pos, alt_pos, ref_acc, TITLE_TEXT

    ckpt = torch.load(f"checkpoints/{name}.ckpt", map_location=DEVICE, weights_only=False)
    task = ckpt["task"]

    reference = LeakyRNN(**ckpt["config"]).to(DEVICE)
    reference.load_state_dict(ckpt["model"])
    reference.eval().requires_grad_(False)
    alternate = copy.deepcopy(reference)

    data = generate_task_data(task, N_PROBE, device=DEVICE, seed=SEED)
    lens = data["lengths"]
    n_ring = data["n_ring"]
    t_max = data["obs"].shape[1]
    steps = torch.arange(t_max, device=DEVICE)
    valid = steps[None] < lens[:, None]

    probe_x = data["obs"] * valid[..., None]
    probe_y = data["labels"]
    colors = trajectory_colors(probe_y.cpu().numpy(), lens.cpu().numpy(), n_ring)

    # steps past the end of a trial repeat its last valid point
    freeze_idx = torch.minimum(steps[None], (lens - 1)[:, None])

    theta = torch.arange(n_ring, device=DEVICE) * 2 * torch.pi / n_ring
    ring_xy = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # (n_ring, 2)

    # ablation target: the linear readout, (n_out, n_hidden)
    w_orig = reference.readout.weight.detach().clone()
    mask = torch.ones_like(w_orig)
    w_live = w_orig.clone()
    vmax = float(abs(w_orig).max())

    ref_pos = torch.zeros(N_PROBE, t_max, 3, device=DEVICE)
    alt_pos = torch.zeros(N_PROBE, t_max, 3, device=DEVICE)
    output_positions(reference, ref_pos)
    output_positions(alternate, alt_pos)

    m = ablation_metrics(alternate, probe_x, probe_y, lens, ref_pos, alt_pos)
    ref_acc = m["acc"]

    TITLE_TEXT = (
        f"acc {ref_acc:.2f} : {m['acc']:.2f}   "
        f"cut {int((mask == 0).sum())}/{mask.numel()}"
    )


load_task(task)

# ------------------ viz

figure = fpl.Figure(shape=(1,3),
                    size=(1200, 450),
                    names=["reference model", "ablation model", "weights"])
figure.canvas.set_title("Post-Hoc Weight Ablation Viz")

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False
    s.controller.enabled = False

phi = np.linspace(0, 2 * np.pi, 200, dtype=np.float32)
unit_circle = np.column_stack(
    [RING_RADIUS * np.cos(phi), RING_RADIUS * np.sin(phi), np.zeros_like(phi)]
)


def build_graphics():
    """Recreate every graphic. Trial length and n_hidden change between tasks."""
    global weights, rs

    for name, pos in (("reference model", ref_pos), ("ablation model", alt_pos)):
        s = figure[name]
        if name in lines:
            s.remove_graphic(lines[name])
            s.remove_graphic(rings[name])

        rings[name] = s.add_line(unit_circle, colors=(0.35, 0.35, 0.35, 1.0), thickness=1.0)

        p = pos.cpu().numpy()
        lines[name] = s.add_line_collection(
            data=[p[i] for i in range(len(p))],
            colors=[colors[i] for i in range(len(colors))],
            thickness=THICKNESS,
        )
        s.camera.maintain_aspect = True
        s.camera.show_rect(-VIEW_LIM, VIEW_LIM, -VIEW_LIM, VIEW_LIM)

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
    rs = weights.add_linear_selector(axis="y", thickness=1.5)
    s.camera.maintain_aspect = False  # (n_out, n_hidden) is wide and short
    s.auto_scale()


rings = {}
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
    alternate.readout.weight.copy_(w_live)

    output_positions(alternate, alt_pos)
    for p, buf in enumerate(alt_buffers):
        buf.update(alt_pos[p], synchronize=True)

    w_tex.update(w_live)

    m = ablation_metrics(alternate, probe_x, probe_y, lens, ref_pos, alt_pos)
    TITLE_TEXT = (
        f"acc {ref_acc:.2f} : {m['acc']:.2f}   "
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
            row =  math.ceil(rs.selection)
            mask[row, :] = 0
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