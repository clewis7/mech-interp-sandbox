import mechiviz as mv

import itertools

import torch
import fastplotlib as fpl
import wgpu
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

from model import load_model, to_images


# ------------------------- load model ckpt and PCA basis

DEV = torch.device("cuda")
model, meta = load_model("rf_celeba64.pt", DEV)

PCA_ckpt = torch.load("pca_basis.pt", map_location=DEV, weights_only=False)

N, NROW, T = 16, 4, 100
DT = 1 / T
SHAPE = (meta["channels"], meta["size"], meta["size"])
seeds = itertools.count(1)

B3 = PCA_ckpt["B"][:, :3].to(DEV).contiguous()          # (16384, 3)
h_shape = tuple(PCA_ckpt["h_shape"])

_tmp = PCA_ckpt["t_mean_proj"][:, :3].to(DEV)
_kt = PCA_ckpt["t_keys"].to(DEV)
idx = (torch.arange(T + 1, device=DEV)[:, None] - _kt[None]).abs().argmin(1)
T_MEAN3 = _tmp[idx].contiguous()            # (T+1, 3), 1.2 KB

colors = fpl.utils.make_colors(N, "tab20")

# -------------------------- hook

_buf = {}
_capture = False

def hook(module, inp, out):
    if _capture:
        _buf["h"] = out.detach().flatten(1)              # (N, 16384)

model.mid2.register_forward_hook(hook)

proj = torch.zeros(T + 1, N, 3, device=DEV)              # 19 KB, resident
proj_valid = -1                                           # highest filled index

# -------------------------- trajectory cache to make scrubbing seamless

cache = torch.empty(T + 1, N, *SHAPE, device=DEV)
NAN = float("nan")
proj = torch.full((N, T + 1, 3), NAN, device=DEV)   # computed values
disp = torch.full((N, T + 1, 3), NAN, device=DEV)   # what's on screen
frontier = 0


def reset_cache(z):
    global frontier
    cache[0] = z
    frontier = 0
    project_at(0)


@torch.no_grad()
def project_at(k):
    """One forward at index k purely to fill proj[k]."""
    global _capture, proj_valid
    _capture = True
    t = torch.full((N,), k * DT, device=DEV)
    model(cache[k], t)
    _capture = False
    proj[:, k] = _buf["h"] @ B3
    proj_valid = max(proj_valid, k)


@torch.no_grad()
def advance_to(k):
    global frontier, _capture, proj_valid
    while frontier < k:
        i = frontier
        t = torch.full((N,), i * DT, device=DEV)
        _capture = True
        v = model(cache[i], t)
        _capture = False
        proj[:, i] = _buf["h"] @ B3
        proj_valid = i
        cache[i + 1] = cache[i] + v * DT
        frontier += 1
    if proj_valid < k:
        project_at(k)                                     # tip only


def reseed():
    torch.manual_seed(next(seeds))
    reset_cache(torch.randn(N, *SHAPE, device=DEV))

# -------------------------- initial data

# NROW x NROW grid of samples
torch.manual_seed(0)
reset_cache(torch.randn(N, *SHAPE, device=DEV))
imgs0 = to_images(cache[0])                      # (16, 64, 64, 3)


# -------------------------- figure

figure = fpl.Figure(shape=(1, 2),
                    size=(1000, 500),
                    cameras=["2d", "3d"],
                    controller_types=["panzoom", "orbit"],
                    names=["samples", "latent trajectories"])

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False

# add initial data
image_grid = figure["samples"].add_image_grid(data=[im.cpu().numpy() for im in imgs0],
                                              shape=(NROW, NROW),
                                              separation=(4, 4),
                                              texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)


bbox = image_grid.world_object.get_world_bounding_box()
x_mid = (bbox[0, 0] + bbox[1, 0]) / 2
y_top = bbox[0, 1]

label = figure["samples"].add_text(
    "t = 0",
    offset=(x_mid, y_top - 12, 0),
    font_size=15,
    anchor="bottom-center",
)

figure["samples"].camera.local.scale_y = -1

current = figure["latent trajectories"].add_scatter(
    proj[:, 0].cpu().numpy(), cmap="tab20", sizes=10
)

advance_to(T)
disp[:] = proj
lc = figure["latent trajectories"].add_line_collection(
    [disp[i].cpu().numpy() for i in range(N)],
    cmap="tab20",
    thickness=2,
    name="traj",
)
figure["latent trajectories"].camera.maintain_aspect = False
figure["latent trajectories"].auto_scale()

figure["samples"].controller.enabled = False


# -------------------------- shared buffers

sample_textures = []
for g in image_grid.graphics:
    tt = mv.TorchTensorTexture(g.data.value.shape)
    tt.texture = g.data.buffer[0, 0]
    sample_textures.append(tt)

line_textures = []
for g in lc.graphics:
    lt = mv.TorchTensorBuffer(g.data.value.shape)
    lt.buffer = g.data.buffer
    line_textures.append(lt)

current_texture = mv.TorchTensorBuffer(current.data.value.shape)
current_texture.buffer = current.data.buffer

# -------------------------- GUI

class Controls(ImguiWindow):
    def __init__(self):
        super().__init__()
        self._t = 0
        self._detrend = True

    def _draw(self):
        k = self._t
        off = T_MEAN3 if self._detrend else 0.0      # (T+1, 3) broadcasts over N

        disp[:, : k + 1] = proj[:, : k + 1] - (off[: k + 1] if self._detrend else 0.0)
        disp[:, k + 1 :] = NAN
        for i, tb in enumerate(line_textures):
            tb.update(disp[i])
        current_texture.update(proj[:, k] - (T_MEAN3[k] if self._detrend else 0.0))

        imgs = to_images(cache[k])
        for tt, im in zip(sample_textures, imgs):
            tt.update(im)
        label.text = f"t = {k}"

    def update(self):
        if imgui.button("Reseed"):
            reseed()
            advance_to(self._t)
            self._draw()

        imgui.same_line()
        changed_d, self._detrend = imgui.checkbox("detrend t", self._detrend)
        if changed_d:
            self._draw()
            figure["latent trajectories"].auto_scale()

        imgui.same_line()
        imgui.set_next_item_width(440)
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


figure.show()

if __name__ == "__main__":
    fpl.loop.run()

