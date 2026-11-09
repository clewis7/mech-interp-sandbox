import itertools
import time

import mechiviz as mv
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

# ---------------------------------------------------------------- species plane

SUB, KPCA = 4000, 400


def fit_plane():
    """-> Q (D, 2) orthonormal, mean (1, D), centroids (nc, 2), proj, labels.

    Axis 0 points cat -> dog, axis 1 points toward wild, so a slider reads as
    the thing it does. Q is orthonormalised because the sliders are meant to
    be displacements: an LDA basis is whitened by the within-class scatter and
    is not orthogonal, so moving 1.0 along one of its directions would not
    move the point 1.0 in the plot.
    """
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


def embed(x):
    return (x.flatten(1) - MEAN) @ Q


# Slider range = the displacement the kick actually has to be able to make.
# For every real sample and every class centroid, how far along each axis
# would you have to push to land on it? Take the 99.5th percentile of that,
# per axis, so aiming at any class from anywhere in the cloud stays on the
# slider instead of being silently clamped.
_need = (CM[None, :, :] - PDATA[:, None, :]).abs().reshape(-1, 2)
LIM = [float(torch.quantile(_need[:, i], 0.995)) for i in range(2)]

for i, c in enumerate(CLASSES):
    print(f"  {c:5s} centroid ({CM[i, 0]:+6.1f}, {CM[i, 1]:+6.1f})")
print(f"  within-class sd  ({PDATA[:, 0].std():.1f}, {PDATA[:, 1].std():.1f})")
print(f"  slider range     (+/-{LIM[0]:.0f}, +/-{LIM[1]:.0f})")

# ---------------------------------------------------------------- trajectories


class Branch:
    """One cached trajectory, plus where each state is HEADED in the plane.

    p[i] is the projection of the endpoint prediction at step i,

        xhat1 = x_t + (1 - t) v(x_t, t)

    not of x_t itself. x_t at small t is noise: it projects to a fixed offset
    nowhere near any class, so "which species is this" has no answer there.
    The endpoint prediction always lands inside the data cloud, so the plane
    stays meaningful at every t and you can watch a trajectory commit.
    """

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
        """Where step i is headed. At t = 1 the state is the endpoint."""
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


def reseed():
    torch.manual_seed(next(seeds))
    z = torch.randn(N, *SHAPE, device=DEV)
    ref.seed(z)
    alt.seed(z.clone())


def kick_vector():
    c = torch.tensor(P["disp"], device=DEV, dtype=torch.float32)
    return (Q @ c).view(*SHAPE)


@torch.no_grad()
def apply_kick(k, upto):
    """Copy the reference up to k, displace once there, integrate alt onward."""
    ref.advance(max(k, upto))

    alt.x[: k + 1] = ref.x[: k + 1]
    alt.p[: k + 1] = ref.p[: k + 1]
    alt.frontier = k

    d = kick_vector()
    if d.norm() > 1e-8:
        alt.x[k] = ref.x[k] + d
        alt.project(k)

    alt.advance(upto)


def aim(i):
    """Displacement that points the kicked state at class i's centroid.

    Q is orthonormal, so a displacement d on the state moves the plane
    position by d - to first order. The endpoint prediction then moves by
    roughly the same amount, but not exactly: xhat1 depends on the model, so
    the model answers back. Aiming lands you near the centroid, not on it,
    and how far off it lands is itself worth looking at.
    """
    ref.advance(P["k"])
    d = CM[i] - ref.p[P["k"]]
    P["disp"] = [float(d[0]), float(d[1])]


torch.manual_seed(0)
reseed()

# ---------------------------------------------------------------- figure

extents = [
    (0, 0.28, 0, 0.33),
    (0, 0.28, 0.33, 0.66),
    (0, 0.28, 0.66, 1.0),
    (0.28, 1.0, 0, 1.0),
]

figure = fpl.Figure(
    extents=extents,
    size=(1200, 820),
    cameras=["2d"] * 4,
    controller_types=["panzoom"] * 4,
    names=["reference", "kicked", "diff", "species plane"],
)
figure.canvas.set_title("Kicking across class boundaries")

for s in figure:
    s.tooltip.enabled = False

for nm in ("reference", "kicked", "diff"):
    figure[nm].axes.visible = False
    figure[nm].camera.local.scale_y = -1
    figure[nm].controller.enabled = False

img0 = to_images(ref.x[0])[0].cpu().numpy()
usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
img_ref = figure["reference"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)
img_alt = figure["kicked"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)
img_diff = figure["diff"].add_image(
    np.full(img0.shape, 0.5, dtype=np.float32), vmin=0, vmax=1, texture_usage=usage
)

# the real data, coloured by class: this is the map the kick moves around on
zeros = torch.zeros(PDATA.shape[0], 1, device=DEV)
cloud = C_CLASS[YDATA.cpu().numpy()].copy()
cloud[:, 3] = 0.35
figure["species plane"].add_scatter(
    torch.cat([PDATA, zeros], 1).cpu().numpy(), colors=cloud, sizes=3
)
figure["species plane"].add_scatter(
    torch.cat([CM, torch.zeros(len(CLASSES), 1, device=DEV)], 1).cpu().numpy(),
    colors=C_CLASS,
    sizes=26,
)

ref.advance(T)
alt.advance(T)

disp_ref = torch.zeros(T + 1, 3, device=DEV)
import itertools
import time

import mechiviz as mv
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

# ---------------------------------------------------------------- species plane

SUB, KPCA = 4000, 400


def fit_plane():
    """-> Q (D, 2) orthonormal, mean (1, D), centroids (nc, 2), proj, labels.

    Axis 0 points cat -> dog, axis 1 points toward wild, so a slider reads as
    the thing it does. Q is orthonormalised because the sliders are meant to
    be displacements: an LDA basis is whitened by the within-class scatter and
    is not orthogonal, so moving 1.0 along one of its directions would not
    move the point 1.0 in the plot.
    """
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


def embed(x):
    return (x.flatten(1) - MEAN) @ Q


# Slider range = the displacement the kick actually has to be able to make.
# For every real sample and every class centroid, how far along each axis
# would you have to push to land on it? Take the 99.5th percentile of that,
# per axis, so aiming at any class from anywhere in the cloud stays on the
# slider instead of being silently clamped.
_need = (CM[None, :, :] - PDATA[:, None, :]).abs().reshape(-1, 2)
LIM = [float(torch.quantile(_need[:, i], 0.995)) for i in range(2)]

for i, c in enumerate(CLASSES):
    print(f"  {c:5s} centroid ({CM[i, 0]:+6.1f}, {CM[i, 1]:+6.1f})")
print(f"  within-class sd  ({PDATA[:, 0].std():.1f}, {PDATA[:, 1].std():.1f})")
print(f"  slider range     (+/-{LIM[0]:.0f}, +/-{LIM[1]:.0f})")

# ---------------------------------------------------------------- trajectories


class Branch:
    """One cached trajectory, plus where each state is HEADED in the plane.

    p[i] is the projection of the endpoint prediction at step i,

        xhat1 = x_t + (1 - t) v(x_t, t)

    not of x_t itself. x_t at small t is noise: it projects to a fixed offset
    nowhere near any class, so "which species is this" has no answer there.
    The endpoint prediction always lands inside the data cloud, so the plane
    stays meaningful at every t and you can watch a trajectory commit.
    """

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
        """Where step i is headed. At t = 1 the state is the endpoint."""
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


def reseed():
    torch.manual_seed(next(seeds))
    z = torch.randn(N, *SHAPE, device=DEV)
    ref.seed(z)
    alt.seed(z.clone())


def kick_vector():
    c = torch.tensor(P["disp"], device=DEV, dtype=torch.float32)
    return (Q @ c).view(*SHAPE)


@torch.no_grad()
def apply_kick(k, upto):
    """Copy the reference up to k, displace once there, integrate alt onward."""
    ref.advance(max(k, upto))

    alt.x[: k + 1] = ref.x[: k + 1]
    alt.p[: k + 1] = ref.p[: k + 1]
    alt.frontier = k

    d = kick_vector()
    if d.norm() > 1e-8:
        alt.x[k] = ref.x[k] + d
        alt.project(k)

    alt.advance(upto)


def aim(i):
    """Displacement that points the kicked state at class i's centroid.

    Q is orthonormal, so a displacement d on the state moves the plane
    position by d - to first order. The endpoint prediction then moves by
    roughly the same amount, but not exactly: xhat1 depends on the model, so
    the model answers back. Aiming lands you near the centroid, not on it,
    and how far off it lands is itself worth looking at.
    """
    ref.advance(P["k"])
    d = CM[i] - ref.p[P["k"]]
    P["disp"] = [float(d[0]), float(d[1])]


torch.manual_seed(0)
reseed()

# ---------------------------------------------------------------- figure

extents = [
    (0, 0.28, 0, 0.33),
    (0, 0.28, 0.33, 0.66),
    (0, 0.28, 0.66, 1.0),
    (0.28, 1.0, 0, 1.0),
]

figure = fpl.Figure(
    extents=extents,
    size=(1200, 820),
    cameras=["2d"] * 4,
    controller_types=["panzoom"] * 4,
    names=["reference", "kicked", "diff", "species plane"],
)
figure.canvas.set_title("Kicking across class boundaries")

for s in figure:
    s.tooltip.enabled = False

for nm in ("reference", "kicked", "diff"):
    figure[nm].axes.visible = False
    figure[nm].camera.local.scale_y = -1
    figure[nm].controller.enabled = False

img0 = to_images(ref.x[0])[0].cpu().numpy()
usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
img_ref = figure["reference"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)
img_alt = figure["kicked"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)
img_diff = figure["diff"].add_image(
    np.full(img0.shape, 0.5, dtype=np.float32), vmin=0, vmax=1, texture_usage=usage
)

# the real data, coloured by class: this is the map the kick moves around on
zeros = torch.zeros(PDATA.shape[0], 1, device=DEV)
cloud = C_CLASS[YDATA.cpu().numpy()].copy()
cloud[:, 3] = 0.35
figure["species plane"].add_scatter(
    torch.cat([PDATA, zeros], 1).cpu().numpy(), colors=cloud, sizes=3
)
figure["species plane"].add_scatter(
    torch.cat([CM, torch.zeros(len(CLASSES), 1, device=DEV)], 1).cpu().numpy(),
    colors=C_CLASS,
    sizes=26,
)

# Two dots and no path: the eye follows where each sample is headed now
# rather than reading a trace. Nothing is precomputed either - the time
# slider drives the integration, which is the point of the demo.
heads = figure["species plane"].add_scatter(
    np.zeros((2, 3), dtype=np.float32),
    colors=np.array([C_REF, C_ALT], dtype=np.float32),
    sizes=22,
)
figure["species plane"].camera.maintain_aspect = False
figure["species plane"].auto_scale()
figure["species plane"].axes.visible = False
figure["species plane"].controller.enabled = False

# axis labels by hand, since the axes themselves are off. Placed off the data
# cloud rather than off the camera, so they sit where the scatter actually is.
_x0, _x1 = float(PDATA[:, 0].min()), float(PDATA[:, 0].max())
_y0, _y1 = float(PDATA[:, 1].min()), float(PDATA[:, 1].max())
_w, _h = _x1 - _x0, _y1 - _y0

figure["species plane"].add_text(
    "LDA 1",
    offset=((_x0 + _x1) / 2, _y0 - 0.10 * _h, 0),
    font_size=16,
    face_color="white",
    screen_space=True,
)
figure["species plane"].add_text(
    "LDA 2",
    offset=(_x0 - 0.10 * _w, (_y0 + _y1) / 2, 0),
    font_size=16,
    face_color="white",
    screen_space=True,
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
tex_diff = bind_texture(img_diff, img0.shape)
buf_heads = bind_points(heads)

head_xy = torch.zeros(2, 3, device=DEV)
diff_scale = 1e-3

# ---------------------------------------------------------------- GUI

P = {"t": 0, "k": 15, "disp": [0.0, 0.0]}
AXES = ("cat -> dog", "-> wild")


def rgb_diff(k):
    d = to_images(alt.x[k])[0] - to_images(ref.x[k])[0]
    return (d / diff_scale).clamp(-1, 1) * 0.5 + 0.5


def draw():
    k = P["t"]

    head_xy[0, :2] = ref.p[k]
    head_xy[1, :2] = alt.p[k]
    buf_heads.update(head_xy)

    tex_ref.update(to_images(ref.x[k])[0])
    tex_alt.update(to_images(alt.x[k])[0])
    tex_diff.update(rgb_diff(k))


def rebuild():
    global diff_scale
    apply_kick(P["k"], P["t"])
    n = alt.frontier + 1
    diff_scale = max(float((alt.x[:n] - ref.x[:n]).abs().max()), 1e-3)
    draw()


class Params(ImguiWindow):
    def update(self):
        touched = False

        imgui.text("displacement")
        for i, name in enumerate(AXES):
            imgui.set_next_item_width(150)
            ch, P["disp"][i] = imgui.slider_float(
                name, P["disp"][i], -LIM[i], LIM[i], "%.1f"
            )
            touched |= ch

        imgui.separator()
        imgui.set_next_item_width(150)
        ch, P["k"] = imgui.slider_int("kick at", P["k"], 1, T-1)
        touched |= ch

        if imgui.button("clear kick"):
            P["disp"] = [0.0, 0.0]
            touched = True

        imgui.separator()
        here = alt.p[P["t"]]
        near = int((CM - here).norm(dim=1).argmin())
        imgui.text(f"kicked is nearest")
        imgui.same_line()
        imgui.text_colored(tuple(float(v) for v in C_CLASS[near]), CLASSES[near])

        if touched:
            rebuild()


class TimeBar(ImguiWindow):
    def update(self):
        if imgui.button("Reseed"):
            reseed()
            P["t"] = 0
            rebuild()

        imgui.same_line()
        imgui.set_next_item_width(780)
        changed, P["t"] = imgui.slider_int("##t", P["t"], 0, T, "")
        imgui.same_line()
        imgui.text(f"t = {P['t'] * DT:.2f}")

        if changed:
            ref.advance(P["t"])
            alt.advance(P["t"])
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
figure.add_imgui_window(Params(), location="right", size=250, title="kick")
figure.imgui_windows["bottom"]._draw_resize_handle = lambda: None

rebuild()
figure.show()

if __name__ == "__main__":
    fpl.loop.run()