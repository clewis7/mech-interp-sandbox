"""
Steering by hand: one kick per click.

No feedback law here. You are the controller. Click a class and the sample
takes one fixed step toward that centroid in the species plane, then
integrates forward a little. Click again and it takes another. Click dog five
times and you have walked it toward dog in five discrete pushes, watching the
model answer back between each one.

The path is built as you fly it, so it is path dependent. Dragging the time
slider forward integrates on with no kick; clicking a class kicks first and
then integrates. Dragging back just scrubs what you have already flown.

Two things to watch:

  the staircase   each kick moves the dot by `strength`, and the integration
                  between kicks moves it back. Early on the model undoes most
                  of a push; late on it cannot. That gap is the whole story.

  the trail       where you have been. If it loops back on itself you are
                  fighting the flow rather than steering it.

    python steer_manual.py
"""

import itertools
import time

import mechiviz as mv        # must come before fastplotlib / pygfx / wgpu
import fastplotlib as fpl
import numpy as np
import torch
import wgpu
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from model import load_model, to_images

DEV = torch.device("cuda")
model, meta = load_model("rf_afhq64_cloud.pt", DEV)

N = 1
T = 100
DT = 1 / T
SHAPE = (meta["channels"], meta["size"], meta["size"])
seeds = itertools.count(1)

CLASSES = list(meta.get("classes", ["cat", "dog", "wild"]))
COLOR = {
    "cat": (0.22, 0.49, 0.87, 1.0),
    "dog": (0.95, 0.52, 0.12, 1.0),
    "wild": (0.30, 0.65, 0.33, 1.0),
}
C_CLASS = np.array([COLOR[c] for c in CLASSES], dtype=np.float32)
C_REF = (0.88, 0.90, 0.94, 1.0)
C_ALT = (0.95, 0.20, 0.25, 1.0)
C_TRAIL = (0.95, 0.20, 0.25, 0.45)

# up here rather than with the GUI: reset_path() reads P["t"] and runs at
# module level, before any window exists
P = {"t": 0, "strength": 1.0, "step": 5}
STRENGTH_MAX = 4.0

# ---------------------------------------------------------------- species plane

SUB, KPCA = 4000, 400


def fit_plane():
    blob = torch.load(f"afhq{meta['size']}_uint8.pt", weights_only=False)
    data, labels = blob["data"], blob["labels"]

    idx = torch.randperm(data.shape[0])[:SUB]
    X = data[idx].to(DEV).float().div_(127.5).sub_(1.0).flatten(1)
    y = labels[idx].to(DEV)
    nc = len(CLASSES)

    mean = X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X - mean, q=KPCA, niter=4)
    Z = (X - mean) @ V

    mu = Z.mean(0)
    Sw = torch.zeros(KPCA, KPCA, device=DEV)
    Sb = torch.zeros(KPCA, KPCA, device=DEV)
    for i in range(nc):
        Zi = Z[y == i]
        d = Zi - Zi.mean(0)
        Sw += d.T @ d
        m = (Zi.mean(0) - mu)[:, None]
        Sb += Zi.shape[0] * (m @ m.T)

    Sw += 1e-3 * Sw.diagonal().mean() * torch.eye(KPCA, device=DEV)
    ev, evec = torch.linalg.eig(torch.linalg.solve(Sw, Sb))
    W = evec.real[:, ev.real.argsort(descending=True)[:2]]

    Q, _ = torch.linalg.qr(V @ W)
    proj = (X - mean) @ Q
    cm = torch.stack([proj[y == i].mean(0) for i in range(nc)])

    ci, di, wi = (CLASSES.index(c) for c in ("cat", "dog", "wild"))
    if cm[di, 0] < cm[ci, 0]:
        Q[:, 0] = -Q[:, 0]
    if cm[wi, 1] < cm[[ci, di], 1].mean():
        Q[:, 1] = -Q[:, 1]

    proj = (X - mean) @ Q
    cm = torch.stack([proj[y == i].mean(0) for i in range(nc)])
    return Q.contiguous(), mean, cm, proj, y


print("fitting the species plane...")
_t0 = time.perf_counter()
Q, MEAN, CM, PDATA, YDATA = fit_plane()
print(f"done in {time.perf_counter() - _t0:.1f}s")
for i, c in enumerate(CLASSES):
    print(f"  {c:5s} centroid ({CM[i, 0]:+6.2f}, {CM[i, 1]:+6.2f})")
SPREAD = float(PDATA.std(0).mean())
print(f"  within-class spread ~{SPREAD:.2f}")

# A fixed unit direction per class, pointing out from the middle of the
# triangle of centroids. Deliberately NOT "toward the centroid from here":
# a centroid is a statistic of the data, while the basin boundary is a
# property of the flow, and there is no reason the two would coincide. Often
# you have to push well past a centroid to cross. Aiming AT one caps you
# there; aiming ALONG a direction does not.
MID = CM.mean(0, keepdim=True)
DIRS = CM - MID
DIRS = DIRS / DIRS.norm(dim=1, keepdim=True)
AT_CENTROID = ((CM - MID) * DIRS).sum(1)        # how far out each centroid is
for i, c in enumerate(CLASSES):
    print(f"  {c:5s} direction ({DIRS[i, 0]:+.2f}, {DIRS[i, 1]:+.2f})"
          f"   centroid at {AT_CENTROID[i]:+.2f} along it")


def embed(x):
    return (x.flatten(1) - MEAN) @ Q


# ---------------------------------------------------------------- trajectories


class Branch:
    """One trajectory. p[i] is where step i is HEADED, in the plane."""

    def __init__(self):
        self.x = torch.empty(T + 1, N, *SHAPE, device=DEV)
        self.p = torch.zeros(T + 1, 2, device=DEV)
        self.frontier = 0

    def seed(self, z):
        self.x[0] = z
        self.p[:] = 0.0
        self.frontier = 0

    @torch.no_grad()
    def project(self, i):
        if i >= T:
            self.p[i] = embed(self.x[i])[0]
            return
        ti = i * DT
        v = model(self.x[i], torch.full((N,), ti, device=DEV))
        self.p[i] = embed(self.x[i] + (1 - ti) * v)[0]

    @torch.no_grad()
    def advance(self, k):
        while self.frontier < k:
            i = self.frontier
            ti = i * DT
            v = model(self.x[i], torch.full((N,), ti, device=DEV))
            self.p[i] = embed(self.x[i] + (1 - ti) * v)[0]
            self.x[i + 1] = self.x[i] + v * DT
            self.frontier += 1
        self.project(self.frontier)


ref, alt = Branch(), Branch()
z0 = torch.zeros(N, *SHAPE, device=DEV)
kicks = []                      # (step index, class index), for the readout


def reseed():
    global z0
    torch.manual_seed(next(seeds))
    z0 = torch.randn(N, *SHAPE, device=DEV)
    ref.seed(z0.clone())
    reset_path()


def reset_path():
    kicks.clear()
    alt.seed(z0.clone())
    alt.advance(0)
    ref.advance(0)
    P["t"] = 0


@torch.no_grad()
def kick_toward(ci):
    """One fixed step ALONG the class direction, then integrate forward.

    The direction does not depend on where you currently are, so clicking
    repeatedly marches in a straight line and keeps going past the centroid.
    """
    i = alt.frontier
    if i >= T:
        return
    d = (Q @ (DIRS[ci] * P["strength"])).view(*SHAPE)
    alt.x[i] = alt.x[i] + d
    alt.project(i)
    kicks.append((i, ci))
    advance_both(min(i + P["step"], T))


def advance_both(k):
    alt.advance(k)
    ref.advance(k)
    P["t"] = alt.frontier


torch.manual_seed(0)
reseed()

# ---------------------------------------------------------------- figure

extents = [
    (0, 0.28, 0, 0.5),
    (0, 0.28, 0.5, 1.0),
    (0.28, 1.0, 0, 1.0),
]

figure = fpl.Figure(
    extents=extents,
    size=(1200, 760),
    cameras=["2d"] * 3,
    controller_types=["panzoom"] * 3,
    names=["reference", "steered", "species plane"],
)
figure.canvas.set_title("Steering by hand")

for s in figure:
    s.tooltip.enabled = False

for nm in ("reference", "steered"):
    figure[nm].axes.visible = False
    figure[nm].camera.local.scale_y = -1
    figure[nm].controller.enabled = False

img0 = to_images(ref.x[0])[0].cpu().numpy()
usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
img_ref = figure["reference"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)
img_alt = figure["steered"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)

zeros = torch.zeros(PDATA.shape[0], 1, device=DEV)
cloud = C_CLASS[YDATA.cpu().numpy()].copy()
cloud[:, 3] = 0.30
figure["species plane"].add_scatter(
    torch.cat([PDATA, zeros], 1).cpu().numpy(), colors=cloud, sizes=3
)
figure["species plane"].add_scatter(
    torch.cat([CM, torch.zeros(len(CLASSES), 1, device=DEV)], 1).cpu().numpy(),
    colors=C_CLASS,
    sizes=26,
)

# where you have flown it. The staircase of kicks against the flow pulling
# back is the thing worth seeing, so this one keeps its path.
trail = torch.zeros(T + 1, 3, device=DEV)
line_trail = figure["species plane"].add_line(
    trail.cpu().numpy(), colors=C_TRAIL, thickness=2
)
heads = figure["species plane"].add_scatter(
    np.zeros((2, 3), dtype=np.float32),
    colors=np.array([C_REF, C_ALT], dtype=np.float32),
    sizes=22,
)

figure["species plane"].camera.maintain_aspect = False
figure["species plane"].auto_scale()
figure["species plane"].axes.visible = False
figure["species plane"].controller.enabled = False

_x0, _x1 = float(PDATA[:, 0].min()), float(PDATA[:, 0].max())
_y0, _y1 = float(PDATA[:, 1].min()), float(PDATA[:, 1].max())
_w, _h = _x1 - _x0, _y1 - _y0
figure["species plane"].add_text(
    "LDA 1", offset=((_x0 + _x1) / 2, _y0 - 0.10 * _h, 0),
    font_size=16, face_color="white", screen_space=True,
)
figure["species plane"].add_text(
    "LDA 2", offset=(_x0 - 0.10 * _w, (_y0 + _y1) / 2, 0),
    font_size=16, face_color="white", screen_space=True,
)

# ---------------------------------------------------------------- buffers


def bind_texture(graphic, shape):
    tt = mv.TorchTensorTexture(shape)
    tt.texture = graphic.data.buffer[0, 0]
    return tt


def bind_points(graphic):
    tb = mv.TorchTensorBuffer(graphic.data.value.shape)
    tb.buffer = graphic.data.buffer
    return tb


tex_ref = bind_texture(img_ref, img0.shape)
tex_alt = bind_texture(img_alt, img0.shape)
buf_trail = bind_points(line_trail)
buf_heads = bind_points(heads)

head_xy = torch.zeros(2, 3, device=DEV)

# ---------------------------------------------------------------- GUI

def draw():
    k = P["t"]

    trail[: k + 1, :2] = alt.p[: k + 1]
    trail[k + 1:, :2] = alt.p[k]
    buf_trail.update(trail)

    head_xy[0, :2] = ref.p[k]
    head_xy[1, :2] = alt.p[k]
    buf_heads.update(head_xy)

    tex_ref.update(to_images(ref.x[k])[0])
    tex_alt.update(to_images(alt.x[k])[0])


class Params(ImguiWindow):
    def update(self):
        imgui.text("kick toward")
        for i, c in enumerate(CLASSES):
            if i:
                imgui.same_line()
            if imgui.button(c):
                kick_toward(i)
                draw()

        imgui.separator()
        imgui.set_next_item_width(150)
        _, P["strength"] = imgui.slider_float(
            "strength", P["strength"], 0.0, STRENGTH_MAX, "%.2f"
        )
        imgui.set_next_item_width(150)
        _, P["step"] = imgui.slider_int("steps per kick", P["step"], 1, 20)

        imgui.separator()
        if imgui.button("reset path"):
            reset_path()
            draw()


class TimeBar(ImguiWindow):
    def update(self):
        if imgui.button("Reseed"):
            reseed()
            draw()

        imgui.same_line()
        imgui.set_next_item_width(760)
        changed, want = imgui.slider_int("##t", P["t"], 0, T, "")
        imgui.same_line()
        imgui.text(f"t = {P['t'] * DT:.2f}")

        # Forward extends the path, integrating on with no kick; backward
        # just scrubs what has already been flown.
        if changed:
            if want > alt.frontier:
                alt.advance(want)
                ref.advance(want)
            P["t"] = want
            draw()


flags = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scroll_with_mouse
)

figure.add_imgui_window(TimeBar(), location="bottom", size=40, title=None,
                        window_flags=flags)
figure.add_imgui_window(Params(), location="right", size=240, title="steering")
figure.imgui_windows["bottom"]._draw_resize_handle = lambda: None

draw()
figure.show()

for i in range(3):
    reseed()

if __name__ == "__main__":
    fpl.loop.run()