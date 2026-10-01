import mechiviz as mv
import fastplotlib as fpl
import torch
import wgpu
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from model import load_model, to_images

DEV = torch.device("cuda")
model, meta = load_model("rf_celeba64.pt", DEV)

N, NROW, T = 16, 4, 100
DT = 1 / T
SHAPE = (meta["channels"], meta["size"], meta["size"])

# -------------------------- trajectory cache

cache = torch.empty(T + 1, N, *SHAPE, device=DEV)
frontier = 0


def reset_cache(z):
    global frontier
    cache[0] = z
    frontier = 0


@torch.no_grad()
def advance_to(k):
    """Integrate up to index k. No-op if already there."""
    global frontier
    while frontier < k:
        i = frontier
        t = torch.full((N,), i * DT, device=DEV)
        v = model(cache[i], t)                  # the flow field at this state
        cache[i + 1] = cache[i] + v * DT        # one Euler step
        frontier += 1


def reseed():
    reset_cache(torch.randn(N, *SHAPE, device=DEV))


reset_cache(torch.randn(N, *SHAPE, device=DEV))

# -------------------------- figure

figure = fpl.Figure(shape=(1, 1), size=(600, 600), names=["samples"])
figure.canvas.set_title("Scrub Samples")
figure["samples"].axes.visible = False
figure["samples"].tooltip.enabled = False
figure["samples"].camera.local.scale_y = -1
figure["samples"].toolbar = False

imgs0 = to_images(cache[0])                     # (N, 64, 64, 3) in [0, 1]

image_grid = figure["samples"].add_image_grid(
    data=[im.cpu().numpy() for im in imgs0],
    shape=(NROW, NROW),
    separation=(4, 4),
    texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
)

bbox = image_grid.world_object.get_world_bounding_box()
label = figure["samples"].add_text(
    "t = 0.00",
    offset=((bbox[0, 0] + bbox[1, 0]) / 2, bbox[0, 1] - 12, 0),
    font_size=16,
    anchor="bottom-center",
)

textures = []
for g in image_grid.graphics:
    tt = mv.TorchTensorTexture(g.data.value.shape)
    tt.texture = g.data.buffer[0, 0]
    textures.append(tt)

# -------------------------- GUI


class Controls(ImguiWindow):
    def __init__(self):
        super().__init__()
        self._t = 0

    def _draw(self):
        imgs = to_images(cache[self._t])
        for tt, im in zip(textures, imgs):
            tt.update(im)
        label.text = f"t = {self._t * DT:.2f}"

    def update(self):
        if imgui.button("Reseed"):
            reseed()
            self._t = 0
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

gui = Controls()
figure.add_imgui_window(gui, location="bottom", size=40, title=None, window_flags=window_flags)
figure.imgui_windows["bottom"]._draw_resize_handle = lambda: None

gui._draw()
figure.show()

if __name__ == "__main__":
    fpl.loop.run()