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
model, meta = load_model("rf_celeba64.pt", DEV)

N = 1
T = 100
DT = 1 / T
SHAPE = (meta["channels"], meta["size"], meta["size"])
NAN = float("nan")
seeds = itertools.count(1)

C_REF = (0.45, 0.55, 0.75, 1.0)
C_ALT = (0.95, 0.45, 0.15, 1.0)

# ---------------------------------------------------------------- PCA basis
# Fit on endpoints: these are the directions faces vary along, so the path is
# maximally spread in view and a push along one of them means something.

FIT_SEEDS, FIT_BATCH, FIT_STEPS = 512, 32, 50


@torch.no_grad()
def fit_basis():
    dt = 1.0 / FIT_STEPS
    out = []
    for b0 in range(0, FIT_SEEDS, FIT_BATCH):
        nb = min(FIT_BATCH, FIT_SEEDS - b0)
        x = torch.randn(nb, *SHAPE, device=DEV)
        for i in range(FIT_STEPS):
            t = torch.full((nb,), i * dt, device=DEV)
            x = x + model(x, t) * dt
        out.append(x.flatten(1))
        print(f"  endpoints {b0 + nb}/{FIT_SEEDS}", end="\r")

    X = torch.cat(out)
    mean = X.mean(0, keepdim=True)
    _, S, V = torch.pca_lowrank(X - mean, q=19, niter=4)
    B = V[:, :3].contiguous()
    flip = torch.sign(B[B.abs().argmax(0), torch.arange(3, device=DEV)])
    return B * flip, mean, (S[:3] / np.sqrt(FIT_SEEDS))


print("fitting pixel-space PCA on endpoints...")
_t0 = time.perf_counter()
B3, MEAN, SIGMA = fit_basis()
print(f"\nbasis {tuple(B3.shape)} in {time.perf_counter() - _t0:.1f}s")
print("sigma per PC:", [round(float(s), 2) for s in SIGMA])


def embed(x):
    return (x.flatten(1) - MEAN) @ B3


# ---------------------------------------------------------------- trajectories


class Branch:
    """One cached trajectory plus its PCA projection."""

    def __init__(self):
        self.x = torch.empty(T + 1, N, *SHAPE, device=DEV)
        self.p = torch.full((T + 1, 3), NAN, device=DEV)
        self.frontier = 0

    def seed(self, z):
        self.x[0] = z
        self.p[:] = NAN
        self.p[0] = embed(z)[0]
        self.frontier = 0

    @torch.no_grad()
    def advance(self, k):
        while self.frontier < k:
            i = self.frontier
            t = torch.full((N,), i * DT, device=DEV)
            self.x[i + 1] = self.x[i] + model(self.x[i], t) * DT
            self.frontier += 1
            self.p[self.frontier] = embed(self.x[self.frontier])[0]


ref, alt = Branch(), Branch()
kick_at = None                                  # index of the kick, None if unkicked
noise = torch.zeros(*SHAPE, device=DEV)         # std 1, held fixed while sigma moves


def new_noise():
    global noise
    noise = torch.randn(*SHAPE, device=DEV)


def reseed():
    global kick_at
    torch.manual_seed(next(seeds))
    z = torch.randn(N, *SHAPE, device=DEV)
    ref.seed(z)
    alt.seed(z.clone())
    new_noise()
    kick_at = None


def kick_vector():
    """The displacement to apply, from whichever tab is active."""
    if P["mode"] == "pca":
        c = torch.tensor(P["disp"], device=DEV, dtype=torch.float32)
        return (B3 @ c).view(*SHAPE)
    return P["sigma"] * noise


@torch.no_grad()
def apply_kick(k, upto):
    """Copy the reference up to k, displace once there, integrate alt onward."""
    global kick_at
    ref.advance(max(k, upto))

    alt.x[: k + 1] = ref.x[: k + 1]
    alt.p[: k + 1] = ref.p[: k + 1]
    alt.p[k + 1:] = NAN
    alt.frontier = k

    d = kick_vector()
    if d.norm() > 1e-8:
        alt.x[k] = ref.x[k] + d
        alt.p[k] = embed(alt.x[k])[0]
        kick_at = k
    else:
        kick_at = None

    alt.advance(upto)


torch.manual_seed(0)
reseed()

# ---------------------------------------------------------------- figure

extents = [
    (0, 0.3, 0, 0.33),
    (0, 0.3, 0.33, 0.66),
    (0, 0.3, 0.66, 1.0),
    (0.3, 1.0, 0, 1.0)
]

figure = fpl.Figure(
    extents=extents,
    size=(800, 900),
    cameras=["2d", "2d", "2d", "3d"],
    controller_types=["panzoom", "panzoom", "panzoom", "orbit"],
    names=["reference", "kicked", "diff", "pca"],
)
figure.canvas.set_title("Push Latent Trajectories")

for s in figure:
    s.tooltip.enabled = False

for nm in ("reference", "kicked", "diff"):
    figure[nm].axes.visible = False
    figure[nm].camera.local.scale_y = -1
    figure[nm].controller.enabled = False

img0 = to_images(ref.x[0])[0].cpu().numpy()
img_ref = figure["reference"].add_image(img0, vmin=0, vmax=1, texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
img_alt = figure["kicked"].add_image(img0, vmin=0, vmax=1, texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
img_diff = figure["diff"].add_image(
    np.full(img0.shape, 0.5, dtype=np.float32), vmin=0, vmax=1,
    texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
)

# fill both once so the camera frames the whole path
ref.advance(T)
alt.advance(T)

disp_ref = torch.full((T + 1, 3), NAN, device=DEV)
disp_alt = torch.full((T + 1, 3), NAN, device=DEV)
disp_ref[:] = ref.p
disp_alt[:] = alt.p

line_ref = figure["pca"].add_line(disp_ref.cpu().numpy(), colors=C_REF, thickness=2)
line_alt = figure["pca"].add_line(disp_alt.cpu().numpy(), colors=C_ALT, thickness=2)
heads = figure["pca"].add_scatter(
    np.zeros((2, 3), dtype=np.float32),
    colors=np.array([C_REF, C_ALT], dtype=np.float32),
    sizes=12,
)

figure["pca"].camera.maintain_aspect = False
figure["pca"].auto_scale()


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
buf_line_ref = bind_points(line_ref)
buf_line_alt = bind_points(line_alt)
buf_heads = bind_points(heads)

head_xy = torch.zeros(2, 3, device=DEV)

diff_scale = 1e-3
def rgb_diff(k):
    d = to_images(alt.x[k])[0] - to_images(ref.x[k])[0]      # both (H, W, 3) in [0,1]
    return (d / diff_scale).clamp(-1, 1) * 0.5 + 0.5


# ---------------------------------------------------------------- GUI

P = {"t": 0, "k": 15, "disp": [0.0, 0.0, 0.0], "mode": "pca", "sigma": 0.0}
KICK_RANGE = 0.5
LIM = [KICK_RANGE * float(s) for s in SIGMA]
SIGMA_LIM = 0.5             # per-element noise std; x itself is ~unit scale at t=0


def draw():
    k = P["t"]

    disp_ref[: k + 1] = ref.p[: k + 1]
    disp_ref[k + 1:] = NAN
    disp_alt[: k + 1] = alt.p[: k + 1]
    disp_alt[k + 1:] = NAN
    buf_line_ref.update(disp_ref)
    buf_line_alt.update(disp_alt)

    head_xy[0] = ref.p[k]
    head_xy[1] = alt.p[k]
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

        if imgui.begin_tab_bar("perturbation"):
            if imgui.begin_tab_item("PCA direction")[0]:
                if P["mode"] != "pca":
                    P["mode"] = "pca"
                    touched = True
                for i in range(3):
                    imgui.set_next_item_width(140)
                    ch, P["disp"][i] = imgui.slider_float(
                        f"PC{i + 1}", P["disp"][i], -LIM[i], LIM[i], "%.1f"
                    )
                    touched |= ch
                if imgui.button("clear"):
                    P["disp"] = [0.0, 0.0, 0.0]
                    touched = True
                imgui.end_tab_item()

            if imgui.begin_tab_item("Add noise")[0]:
                if P["mode"] != "noise":
                    P["mode"] = "noise"
                    touched = True
                imgui.set_next_item_width(140)
                ch, P["sigma"] = imgui.slider_float(
                    "sigma", P["sigma"], 0.0, SIGMA_LIM, "%.2f"
                )
                touched |= ch
                if imgui.button("resample"):
                    new_noise()
                    touched = True
                imgui.same_line()
                if imgui.button("clear"):
                    P["sigma"] = 0.0
                    touched = True
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.separator()
        imgui.set_next_item_width(140)
        ch, P["k"] = imgui.slider_int("kick at", P["k"], 0, T)
        touched |= ch

        if touched:
            rebuild()


class TimeBar(ImguiWindow):
    def update(self):
        if imgui.button("Reseed"):
            reseed()
            P["t"] = 0
            rebuild()

        imgui.same_line()
        imgui.set_next_item_width(640)
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

figure.add_imgui_window(TimeBar(), location="bottom", size=40, title=None, window_flags=flags)
figure.add_imgui_window(Params(), location="right", size=210, title="kick")
figure.imgui_windows["bottom"]._draw_resize_handle = lambda: None

rebuild()
figure.show()

if __name__ == "__main__":
    fpl.loop.run()