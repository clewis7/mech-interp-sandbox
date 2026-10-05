import mechiviz as mv
import fastplotlib as fpl
import wgpu

import numpy as np
import torch



device = torch.device("cuda")
shape = (4096, 4096, 3)

fig = fpl.Figure(
    shape=(1, 1),
    size=(750, 750),
    names=["shared buffer"],
    canvas_kwargs={"max_fps": 10_000, "vsync": False},
)

fig[0, 0].axes.visible = False
fig[0, 0].tooltip.enabled = False

init = np.zeros(shape, dtype=np.float32)
#naive_graphic = fig[0, 0].add_image(init, vmin=0, vmax=1)
shared_graphic = fig[0, 0].add_image(init, vmin=0, vmax=1, texture_usage= wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING)

#data = torch.randn(shape, dtype=torch.float32, device=device)
tt = mv.TorchTensorTexture(shape)
tt.texture = shared_graphic.data.buffer[0, 0]

def on_frame():
    new_data = torch.randn(shape, dtype=torch.float32, device=device)
    tt.update(new_data)
   # naive_graphic.data = new_data.cpu().numpy()

fig.add_animations(on_frame)

if fpl.IMGUI:
    # show fps with imgui overlay
    fig.imgui_show_fps = True

fig.show()


if __name__ == "__main__":
    fpl.loop.run()