import mechiviz as mv

import copy

import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib.ui import ImguiWindow
import torch
import wgpu

from model import LeakyRNN
from generate_data import generate_task_data
from utils import get_checkpointed_tasks, PCAProjector, trajectory_colors

# ------------------ setup

DEVICE = "cuda"
CKPT = "checkpoints/go.ckpt"
N_PROBE = 256
SEED = 0

TRAJ_SCALE = 1.0
VIEW_LIM = 3.0
THICKNESS = 1.5

# ------------------ initial models

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
task = ckpt["task"]

reference = LeakyRNN(**ckpt["config"]).to(DEVICE)
reference.load_state_dict(ckpt["model"])
reference.eval().requires_grad_(False)

alternate = copy.deepcopy(reference)

# ------------------ data

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

# ------------------ PCA

ref_proj = PCAProjector(reference, probe_x, lens, scale=TRAJ_SCALE)
ref_proj.update(refit=True)

alt_proj = PCAProjector(alternate, probe_x, lens, scale=TRAJ_SCALE)
alt_proj.basis = ref_proj.basis
alt_proj.center = ref_proj.center
alt_proj.update(refit=False)

print(f"{task}: probe {N_PROBE} x {ref_proj.n_steps}, "
      f"var explained {ref_proj.var_explained.sum():.2f}")

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

lines = {}
for name, proj in (("reference model", ref_proj), ("ablation model", alt_proj)):
    s = figure[name]

    pos = proj.positions.cpu().numpy()
    lines[name] = s.add_line_collection(
        data=[pos[p] for p in range(len(pos))],
        colors=[colors[p] for p in range(len(colors))],
        thickness=THICKNESS,
    )

figure["reference model"].controller = figure["ablation model"].controller

weights = figure["weights"].add_image(
        data=w_live.cpu().numpy(),
        cmap="bwr", vmin=-vmax, vmax=vmax,
        texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
rs = weights.add_rectangle_selector()

# ------------------ shared buffers

w_tex = mv.TorchTensorTexture(shape=w_live.shape)
w_tex.texture = weights.data.buffer[0, 0]

alt_buffers = [
    mv.TorchTensorBuffer(shape=g.data.value.shape) for g in lines["ablation model"].graphics
]
for buf, g in zip(alt_buffers, lines["ablation model"].graphics):
    buf.buffer = g.data.buffer

# ------------------ funcs

@torch.no_grad()
def probe_accuracy(model):
    """Trial accuracy on the probe set: fixation held throughout AND last step correct."""
    logits, _ = model(probe_x)
    pred = logits.argmax(-1)
    fix_ok = ~((pred != 0) & (probe_y == 0)).any(dim=1)
    rows = torch.arange(len(probe_y), device=DEVICE)
    resp_ok = pred[rows, lens - 1] == probe_y[rows, lens - 1]
    return (fix_ok & resp_ok).float().mean().item()

ref_acc = probe_accuracy(reference)

TITLE_TEXT = (
        f"ref {ref_acc:.3f}   ablated {ref_acc:.3f}")


def apply_mask():
    global TITLE_TEXT
    torch.mul(w_orig, mask, out=w_live)
    alternate.w_rec.weight.copy_(w_live)

    alt_proj.update(refit=False)
    for p, buf in enumerate(alt_buffers):
        buf.update(alt_proj.positions[p], synchronize=True)

    w_tex.update(w_live)

    alt_acc = probe_accuracy(alternate)
    TITLE_TEXT = (
        f"ref {ref_acc:.3f}   ablated {alt_acc:.3f}")

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