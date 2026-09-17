"""
Perturb and compare.

Two trajectories from the same noise: a reference, and one that gets kicked at
whatever t you are parked on. Scrub to see how far apart they end up.

The point: the same size kick has completely different consequences depending
on where you inject it. Early it changes who the person is. Late it only moves
texture around.
"""

import itertools

import branchpoint as bp
import fastplotlib as fpl
import numpy as np
import torch
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from model import load_model, to_grid

DEV = torch.device("cuda")
model, meta = load_model("rf_celeba64.pt", DEV)

N, T = 16, 100
DT = 1 / T
SHAPE = (meta["channels"], meta["size"], meta["size"])
LUM = torch.tensor([0.299, 0.587, 0.114], device=DEV).view(1, 3, 1, 1)
seeds = itertools.count(1)


# ---------------------------------------------------------------- trajectories
class Branch:
    """One cached Euler trajectory. ~79 MB at 16x3x64x64, T=100."""

    def __init__(self):
        self.states = torch.empty(T + 1, N, *SHAPE, device=DEV)
        self.frontier = 0

    def reset(self, z):
        self.states[0] = z
        self.frontier = 0

    @torch.no_grad()
    def advance(self, k):
        while self.frontier < k:
            i = self.frontier
            t = torch.full((N,), i * DT, device=DEV)
            self.states[i + 1] = self.states[i] + model(self.states[i], t) * DT
            self.frontier = i + 1

    def at(self, k):
        return self.states[k]


ref, alt = Branch(), Branch()
branch_t = None          # index where alt was kicked, None if unkicked


def reseed():
    global branch_t
    torch.manual_seed(next(seeds))
    z = torch.randn(N, *SHAPE, device=DEV)
    ref.reset(z)
    alt.reset(z.clone())
    branch_t = None


@torch.no_grad()
def kick(k, eps, scale_with_t):
    """Copy the reference up to k, perturb there, let alt diverge from it."""
    global branch_t
    ref.advance(k)
    alt.states[: k + 1] = ref.states[: k + 1]
    n = torch.randn(N, *SHAPE, device=DEV)
    if scale_with_t:
        n *= 1 - k * DT                      # match the noise still left in x_t
    alt.states[k] = ref.states[k] + eps * n
    alt.frontier = k
    branch_t = k


def clear_branch():
    global branch_t
    alt.states[: ref.frontier + 1] = ref.states[: ref.frontier + 1]
    alt.frontier = ref.frontier
    branch_t = None


# ---------------------------------------------------------------- panels
@torch.no_grad()
def state_grids(x, t_val):
    """-> (x grid, endpoint grid, raw endpoint prediction)."""
    t = torch.full((N,), t_val, device=DEV)
    v = model(x, t)
    xhat = x + (1 - t_val) * v
    return to_grid(x), to_grid(xhat), xhat


def signed_lum(d):
    """(N,C,H,W) difference -> (H, W) in [0,1], 0.5 = no change, plus its scale."""
    s = (d * LUM).sum(1, keepdim=True)
    m = s.abs().amax().clamp_min(1e-8)
    return to_grid(s / m), float(m)


# ---------------------------------------------------------------- figure
reseed()
a0, b0, _ = state_grids(ref.at(0), 0.0)
d0, _ = signed_lum(torch.zeros(N, *SHAPE, device=DEV))

figure = fpl.Figure(
    shape=(2, 2),
    names=[["xt_ref", "xhat_ref"],
           ["xt_alt", "xhat_alt"]],
    size=(8_00, 760),
)
for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False
    s.toolbar = False

figure["xt_ref"].title = "xₜ  reference"
figure["xt_alt"].title = "xₜ  perturbed"
figure["xhat_ref"].title = "E[x₁|xₜ]  reference"
figure["xhat_alt"].title = "E[x₁|xₜ]  perturbed"


def rgb_panel(name, shape):
    g = figure[name].add_image(np.zeros(shape, dtype=np.float32), vmin=0, vmax=1)
    tt = bp.TorchTensorTexture(shape)
    tt.texture = g.data.buffer[0, 0]
    return tt


def diff_panel(name, shape):
    g = figure[name].add_image(np.zeros(shape, dtype=np.float32), cmap="RdBu_r", vmin=0, vmax=1)
    tt = bp.TorchTensorTexture(shape)
    tt.texture = g.data.buffer[0, 0]
    return tt


tt_xt_ref = rgb_panel("xt_ref", a0.shape)
tt_xt_alt = rgb_panel("xt_alt", a0.shape)
tt_xhat_ref = rgb_panel("xhat_ref", b0.shape)
tt_xhat_alt = rgb_panel("xhat_alt", b0.shape)

label = figure["xt_ref"].add_text("", offset=(a0.shape[1] / 2, -15, 0), font_size=14)


# ---------------------------------------------------------------- gui
class Controls(ImguiWindow):
    def __init__(self):
        super().__init__()
        self._t = 0
        self._eps = 0.30
        self._scale = True

    def _draw(self):
        k, tv = self._t, self._t * DT
        xr, xa = ref.at(k), alt.at(k)

        gr_x, gr_h, hr = state_grids(xr, tv)
        ga_x, ga_h, ha = state_grids(xa, tv)

        tt_xt_ref.update(gr_x)
        tt_xt_alt.update(ga_x)
        tt_xhat_ref.update(gr_h)
        tt_xhat_alt.update(ga_h)

        tag = "unkicked" if branch_t is None else f"kicked at t = {branch_t * DT:.2f}"
        label.text = f"t = {tv:.2f}    {tag} "

    def update(self):
        if imgui.button("Reseed"):
            reseed()
            ref.advance(self._t)
            alt.advance(self._t)
            self._draw()

        imgui.same_line()
        if imgui.button("Kick here"):
            kick(self._t, self._eps, self._scale)
            self._draw()

        imgui.same_line()
        if imgui.button("Clear"):
            clear_branch()
            self._draw()

        imgui.same_line()
        imgui.set_next_item_width(140)
        _, self._eps = imgui.slider_float("eps", self._eps, 0.01, 1.5)

        imgui.same_line()
        _, self._scale = imgui.checkbox("scale by (1-t)", self._scale)

        imgui.set_next_item_width(560)
        changed, self._t = imgui.slider_int("##t", self._t, 0, T, "")
        imgui.same_line()
        imgui.text("t")

        if changed:
            ref.advance(self._t)
            alt.advance(self._t)
            self._draw()


window_flags = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scroll_with_mouse
)

gui = Controls()
figure.add_imgui_window(gui, location="bottom", size=70, title=None, window_flags=window_flags)
figure.imgui_windows["bottom"]._draw_resize_handle = lambda: None

gui._draw()
figure.show()

if __name__ == "__main__":
    fpl.loop.run()