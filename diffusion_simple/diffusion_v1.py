import itertools


import branchpoint as bp
import fastplotlib as fpl
import numpy as np
import torch
from imgui_bundle import imgui
from fastplotlib.ui import ImguiWindow


from model import load_model, to_grid

DEV = torch.device("cuda")
model, meta = load_model("rf_celeba64.pt", DEV)

N, T = 16, 100
DT = 1 / T
SHAPE = (meta["channels"], meta["size"], meta["size"])
seeds = itertools.count(1)

LUM = torch.tensor([0.299, 0.587, 0.114], device=DEV).view(1, 3, 1, 1)

# -------------------- trajectory cache
cache = torch.empty(T + 1, N, *SHAPE, device=DEV)  # ~79 MB at 16x3x64x64, T=100
frontier = 0


def reset_cache(z):
    global frontier
    cache[0] = z
    frontier = 0


@torch.no_grad()
def advance_to(k):
    """Extend the cache up to index k. No-op if already there."""
    global frontier
    while frontier < k:
        i = frontier
        t = torch.full((N,), i * DT, device=DEV)
        cache[i + 1] = cache[i] + model(cache[i], t) * DT
        frontier += 1
    return cache[k]


@torch.no_grad()
def panels(x, t_val):
    t = torch.full((N,), t_val, device=DEV)
    v = model(x, t)
    xhat = x + (1 - t_val) * v
    vs = (v * LUM).sum(1, keepdim=True)  # signed, (N,1,H,W)
    s = vs.abs().amax().clamp_min(1e-8)
    vgrid = to_grid(vs / s)
    return to_grid(x), to_grid(xhat), vgrid


@torch.no_grad()
def reseed_at_t(t: int):
    """Fresh noise, same point on the timeline."""
    reset_cache(torch.randn(N, *SHAPE, device=DEV))
    advance_to(t)


# ---------------- figure
torch.manual_seed(0)
reset_cache(torch.randn(N, *SHAPE, device=DEV))
a, b, c = panels(cache[0], 0.0)


figure = fpl.Figure(
    shape=(1, 3), names=["xₜ", "E[x₁|xₜ]", "v · lum"], size=(1_000, 400)
)
for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False
    s.toolbar = False

x_t = figure[0, 0].add_image(np.zeros(a.shape, dtype=np.float32), vmin=0, vmax=1)
xt_tt = bp.TorchTensorTexture(a.shape)
xt_tt.texture = x_t.data.buffer[0, 0]

text_graphic = figure[0, 0].add_text(
    f"t = 0.00", offset=(int(a.shape[0] / 2), -15, 0), font_size=15
)

x_hat = figure[0, 1].add_image(np.zeros(b.shape, dtype=np.float32), vmin=0, vmax=1)
xhatt_tt = bp.TorchTensorTexture(b.shape)
xhatt_tt.texture = x_hat.data.buffer[0, 0]

v = figure[0, 2].add_image(
    np.zeros(c.shape, dtype=np.float32), cmap="RdBu_r", vmin=0, vmax=1
)
v_tt = bp.TorchTensorTexture(c.shape)
v_tt.texture = v.data.buffer[0, 0]


# ---------------------- gui


# imgui time slider at the bottom
class SliderWindow(ImguiWindow):
    def __init__(self):
        super().__init__()
        self._t = 0

    def _draw(self):
        a, b, c = panels(cache[self._t], self._t * DT)
        xt_tt.update(a)
        xhatt_tt.update(b)
        v_tt.update(c)
        text_graphic.text = f"t = {self._t * DT:.2f}"

    def update(self):
        if imgui.button("Reseed"):
            reseed_at_t(self._t)
            self._draw()

        imgui.same_line()
        imgui.set_next_item_width(420)
        changed, self._t = imgui.slider_int("##t", self._t, 0, T, "")
        imgui.same_line()
        imgui.text("t")

        if changed:
            advance_to(self._t)
            self._draw()


window_flags = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_move
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scroll_with_mouse
)

# make GUI instance
gui = SliderWindow()
figure.add_imgui_window(
    gui, location="bottom", size=40, title=None, window_flags=window_flags
)
# remove the resize handle
figure.imgui_windows["bottom"]._draw_resize_handle = lambda: None

figure.show()

if __name__ == "__main__":
    fpl.loop.run()
