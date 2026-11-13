import mechiviz as mv
import fastplotlib as fpl
import numpy as np
import torch
import torchvision
import wgpu

from imgui_bundle import imgui

import plenoptic as po

# --------- configuration
DEVICE = torch.device("cuda")
DTYPE = torch.float32
MAX_ITER = 6000
CHUNK = 10  # synthesis iterations per rendered frame
TARGET_LAYER = "layer3"
TEX_USAGE = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST

po.set_seed(1)

# ----------- plenoptic stuff
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
deepnet = torchvision.models.resnet50(weights=weights)
deepnet.eval()

transform = weights.transforms()
norm = torchvision.transforms.Normalize(transform.mean, transform.std)

model = po.models.DeepNetFeatures(deepnet, TARGET_LAYER, norm)
model.to(DEVICE).to(DTYPE)
po.remove_grad(model)

img = po.data.macaque()
img = po.process.blur_downsample(img, 2)[..., :-59, :]
img = po.process.center_crop(img, transform.crop_size[0])
img = img.to(DEVICE).to(DTYPE)  # (1, 3, 224, 224)


def new_metamer():
    met = po.Metamer(img, model)
    met.setup(
        optimizer_kwargs={"amsgrad": False},
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_kwargs={"step_size": 3000, "gamma": 0.5},
    )
    return met


met = new_metamer()


# ------- helper funcs
def to_hwc(t):
    """(1, 3, H, W) torch tensor -> (H, W, 3)"""
    return t[0].detach().clamp(0, 1).permute(1, 2, 0)


def rep_error():
    """Representation error averaged across channels, (H, W)"""
    with torch.no_grad():
        err = met.model(met.metamer) - met.target_representation
    return met.model.convert_to_dict(err)[TARGET_LAYER][0].mean(0)


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
)

for s in fig:
    s.tooltip.enabled = False

for name in ["metamer", "representation error"]:
    fig[name].axes.visible = False

# graphics
metamer = fig["metamer"].add_image(
    to_hwc(met.metamer).cpu().numpy(), vmin=0, vmax=1, texture_usage=TEX_USAGE
)
err_image = fig["representation error"].add_image(
    err.cpu().numpy(), cmap="RdBu_r", texture_usage=TEX_USAGE
)
loss_line = fig["loss"].add_line(loss_pos, thickness=2)

# log-scale tick labels
fig["loss"].axes.y.tick_format = lambda v, vmin, vmax: f"{10 ** v:.0e}"
fig["loss"].axes.x.tick_format = "0.0f"

# ----------------- mechiviz
metamer_texture = mv.TorchTensorTexture(shape=to_hwc(met.metamer).shape)
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

    prev = iteration
    met.synthesize(max_iter=min(CHUNK, MAX_ITER - iteration), stop_iters_to_check=MAX_ITER)
    iteration = len(met.losses) - 1

    # metamer
    metamer_texture.update(to_hwc(met.metamer))
    fig["metamer"].title = f"metamer [iteration={iteration}]"

    # fill in the iterations from this chunk
    loss_line.data[prev + 1 : iteration + 1, 1] = np.log10(met.losses[prev + 1 :].numpy())
    fig["loss"].auto_scale()

    # representation error, symmetric color limits 
    err = rep_error()
    err_texture.update(err)
    vmax = err.abs().max().item()
    err_image.vmin, err_image.vmax = -vmax, vmax


def restart():
    global met, iteration
    met = new_metamer()
    iteration = 0

    loss_line.data[:, 1] = np.nan
    loss_line.data[0, 1] = np.log10(met.losses[0].item())

    metamer_texture.update(to_hwc(met.metamer))
    fig["metamer"].title = "metamer [iteration=0]"


@fig.add_imgui_window(location="top", size=40, title=None)
def gui(f):
    if imgui.button("replay"):
        restart()


fig["metamer"].add_animations(animate)

fpl.loop.run()