
import itertools

import mechiviz as mv
import fastplotlib as fpl
import numpy as np
import torch
import wgpu
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from model import load_model, to_images, to_grid

DEV = torch.device("cuda")
model, meta = load_model("rf_afhq64_cloud.pt", DEV)

N = 1
T = 100
DT = 1 / T
SHAPE = (meta["channels"], meta["size"], meta["size"])
seeds = itertools.count(1)

# ---------------------------------------------------------------- trajectories


class Branch:
    """One cached trajectory, integrated lazily as the slider asks for it."""

    def __init__(self):
        self.x = torch.empty(T + 1, N, *SHAPE, device=DEV)
        self.frontier = 0

    def seed(self, z):
        self.x[0] = z
        self.frontier = 0

    @torch.no_grad()
    def advance(self, k):
        while self.frontier < k:
            i = self.frontier
            t = torch.full((N,), i * DT, device=DEV)
            self.x[i + 1] = self.x[i] + model(self.x[i], t) * DT
            self.frontier += 1


ref, alt = Branch(), Branch()
noise = torch.zeros(N, *SHAPE, device=DEV)


def new_noise():
    global noise
    noise = torch.randn(N, *SHAPE, device=DEV)


def reseed():
    torch.manual_seed(next(seeds))
    z = torch.randn(N, *SHAPE, device=DEV)
    ref.seed(z)
    alt.seed(z.clone())
    new_noise()


# @torch.no_grad()
# def renoise(x, k):
#     """Throw the noise half of x_t away and rebuild around a fresh draw."""
#     t = k * DT
#     v = model(x, torch.full((N,), t, device=DEV))
#     xhat1 = x + (1 - t) * v
#     return (1 - t) * noise + t * xhat1

@torch.no_grad()
def renoise(k):
    t = k * DT
    return (1 - t) * noise + t * ref.x[T]

@torch.no_grad()
def apply_kick(k, upto):
    """Copy the reference up to k, re-noise once there, integrate alt onward."""
    ref.advance(max(k, upto))

    alt.x[: k + 1] = ref.x[: k + 1]
    alt.x[k] = renoise(k)
    alt.frontier = k
    alt.advance(upto)


torch.manual_seed(0)
reseed()

# ---------------------------------------------------------------- figure

extents = [
    (0.0, 1 / 3, 0, 1),
    (1 / 3, 2 / 3, 0, 1),
    (2 / 3, 1.0, 0, 1),
]

figure = fpl.Figure(
    extents=extents,
    size=(1150, 480),
    cameras=["2d"] * 3,
    controller_types=["panzoom"] * 3,
    names=["reference", "re-noised", "diff"],
)
figure.canvas.set_title("Re-noising partway through")

for s in figure:
    s.tooltip.enabled = False
    s.axes.visible = False
    s.camera.local.scale_y = -1
    s.controller.enabled = False

img0 = to_images(ref.x[0])[0].cpu().numpy()
usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
img_ref = figure["reference"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)
img_alt = figure["re-noised"].add_image(img0, vmin=0, vmax=1, texture_usage=usage)
img_diff = figure["diff"].add_image(
    np.full(img0.shape, 0.5, dtype=np.float32), vmin=0, vmax=1, texture_usage=usage
)

# ---------------------------------------------------------------- buffers


def bind_texture(graphic, shape):
    tt = mv.TorchTensorTexture(shape)
    tt.texture = graphic.data.buffer[0, 0]
    return tt


tex_ref = bind_texture(img_ref, img0.shape)
tex_alt = bind_texture(img_alt, img0.shape)
tex_diff = bind_texture(img_diff, img0.shape)

diff_scale = 1e-3

# ---------------------------------------------------------------- GUI

P = {"t": 0, "k": 50}


def rgb_diff(k):
    d = to_images(alt.x[k])[0] - to_images(ref.x[k])[0]
    return (d / diff_scale).clamp(-1, 1) * 0.5 + 0.5


def draw():
    k = P["t"]
    tex_ref.update(to_images(ref.x[k])[0])
    tex_alt.update(to_images(alt.x[k])[0])
    tex_diff.update(rgb_diff(k))


def rebuild():
    global diff_scale
    apply_kick(P["k"], P["t"])
    n = alt.frontier + 1
    diff_scale = max(float((alt.x[:n] - ref.x[:n]).abs().max()), 1e-3)
    draw()

SHOW_T = (0.2, 0.4, 0.6, 0.8, 1.0)       # increments of 0.2, five columns


def save_rows():
    """Reference on top, re-noised below, same sample, same column times."""
    import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref.advance(T)
    apply_kick(P["k"], T)                # run the kicked branch out to t = 1

    ks = [round(t * T) for t in SHOW_T]
    grid = torch.cat([ref.x[k] for k in ks] + [alt.x[k] for k in ks])
    img = to_grid(grid, nrow=2).cpu().numpy()

    n = len(SHOW_T)
    tile_h, tile_w = img.shape[0] // 2, img.shape[1] // n

    fig, ax = plt.subplots(figsize=(1.6 * n, 1.6 * 2 + 0.7))
    ax.imshow(img)
    ax.set_xticks([(j + 0.5) * tile_w for j in range(n)])
    ax.set_xticklabels([f"t = {t:g}" for t in SHOW_T])
    ax.set_yticks([0.5 * tile_h, 1.5 * tile_h])
    ax.set_yticklabels(["reference", "re-noised"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"re-noised at t = {P['k'] * DT:.2f}", fontsize=13, pad=10)

    name = f"renoise_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    fig.savefig(name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


class Kick(ImguiWindow):
    def update(self):
        touched = False

        imgui.set_next_item_width(320)
        ch, P["k"] = imgui.slider_int("##k", P["k"], 1, T-1)
        touched |= ch

        imgui.same_line()
        imgui.text(f"t0 = {P['k'] * DT:.2f}   noise level {1 - P['k'] * DT:.2f}")

        imgui.same_line()
        if imgui.button("new noise"):
            new_noise()
            touched = True

        if touched:
            rebuild()

        imgui.same_line()
        if imgui.button("save"):
            save_rows()


class TimeBar(ImguiWindow):
    def update(self):
        if imgui.button("Reseed"):
            reseed()
            P["t"] = 0
            rebuild()

        imgui.same_line()
        imgui.set_next_item_width(840)
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

figure.add_imgui_window(Kick(), location="top", size=40, title=None,
                        window_flags=flags)
figure.add_imgui_window(TimeBar(), location="bottom", size=40, title=None,
                        window_flags=flags)
for loc in ("top", "bottom"):
    figure.imgui_windows[loc]._draw_resize_handle = lambda: None

rebuild()
figure.show()

if __name__ == "__main__":
    fpl.loop.run()