import mechiviz as mv
import fastplotlib as fpl
import numpy as np
import torch
import wgpu

from imgui_bundle import imgui

import plenoptic as po

# --------- configuration
DEVICE = torch.device("cuda")
DTYPE = torch.float32
MAX_ITER = 110
TEX_USAGE = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST

po.set_seed(0)

# ----------- plenoptic stuff
img = po.data.einstein().to(DEVICE).to(DTYPE)  # (1, 1, 256, 256)
model = po.models.Gaussian(30).eval()
model.to(DEVICE).to(DTYPE)
po.remove_grad(model)

met = po.Metamer(img, model)
#met.setup()

# ------- helper funcs
def to_hw(t):
    """(1, 1, H, W) torch tensor -> (H, W)"""
    return t[0, 0].detach()


def rep_error():
    """Representation error"""
    with torch.no_grad():
        return to_hw(met.model(met.metamer) - met.target_representation)



# log loss line, already on CPU so no need to use mechiviz
loss_ys = np.full(MAX_ITER + 1, np.nan, dtype=np.float32)
loss_ys[0] = np.log10(met.losses[0].item())
loss_pos = np.column_stack([np.arange(MAX_ITER + 1), loss_ys])

err = rep_error()

# ----------------- figure
fig = fpl.Figure(
    shape=(1, 3),
    size=(1200, 450),
    names=["metamer", "loss", "representation error"],
    controller_ids=[["metamer", "representation error"]],
)

for s in fig:
    s.tooltip.enabled = False

for name in ["metamer", "representation error"]:
    fig[name].axes.visible = False

# graphics
metamer = fig["metamer"].add_image(
    to_hw(met.metamer).cpu().numpy(), cmap="gray", vmin=0, vmax=1, texture_usage=TEX_USAGE
)
err_image = fig["representation error"].add_image(
    err.cpu().numpy(), cmap="RdBu_r", texture_usage=TEX_USAGE
)
loss_line = fig["loss"].add_line(loss_pos, thickness=2)

# log-scale tick labels
fig["loss"].axes.y.tick_format = lambda v, vmin, vmax: f"{10 ** v:.0e}"
fig["loss"].axes.x.tick_format = "0.0f"

# ----------------- mechiviz
metamer_texture = mv.TorchTensorTexture(shape=img.shape[-2:])
metamer_texture.texture = metamer.data.buffer[0, 0]

err_texture = mv.TorchTensorTexture(shape=err.shape)
err_texture.texture = err_image.data.buffer[0, 0]

fig.show()

fig["loss"].camera.maintain_aspect = False
fig["loss"].controller.enabled = False

# ----------------- synthesis loop
iteration = 0


def animate():
    global iteration
    if iteration >= MAX_ITER:
        return

    met.synthesize(max_iter=min(1, MAX_ITER - iteration), stop_iters_to_check=MAX_ITER)
    iteration = len(met.losses) - 1

    # metamer
    metamer_texture.update(to_hw(met.metamer))
    fig["metamer"].title = f"metamer [iteration={iteration}]"

    loss_line.data[iteration, 1] = np.log10(met.losses[-1].item())
    fig["loss"].auto_scale()

    # representation error, symmetric color limits like plenoptic
    err = rep_error()
    err_texture.update(err)
    vmax = err.abs().max().item()
    err_image.vmin, err_image.vmax = -vmax, vmax


def restart():
    global met, iteration
    met = po.Metamer(img, model)
   # met.setup()
    iteration = 0

    loss_line.data[:, 1] = np.nan
    loss_line.data[0, 1] = np.log10(met.losses[0].item())

    metamer_texture.update(to_hw(met.metamer))
    fig["metamer"].title = "metamer [iteration=0]"


@fig.add_imgui_window(location="top", size=40, title=None)
def gui(f):
    if imgui.button("replay"):
        restart()


fig["metamer"].add_animations(animate)

fpl.loop.run()