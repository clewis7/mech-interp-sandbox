import mechiviz as mv
import fastplotlib as fpl
import wgpu

import numpy as np
import torch


device = torch.device("cuda")
shape = (4096, 4096)

SQUARE = 512  # side length of the square in pixels
STEP = 16  # pixels moved along the diagonal per frame
BG = 0.2  # background value, shows the image outline against the black canvas
FG = 1.0  # square value
CMAP = "viridis"

fig = fpl.Figure(
    shape=(1, 1),
    size=(750, 750),
    names=["host"],
    canvas_kwargs={"max_fps": 10_000, "vsync": False},
)

fig[0, 0].axes.visible = False
fig[0, 0].tooltip.enabled = False

init = np.full(shape, BG, dtype=np.float32)
naive_graphic = fig[0, 0].add_image(init, vmin=0, vmax=1, cmap=CMAP)
# shared_graphic = fig[0, 0].add_image(init, vmin=0, vmax=1, cmap=CMAP, texture_usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING)
#
# tt = mv.TorchTensorTexture(shape)
# tt.texture = shared_graphic.data.buffer[0, 0]

# preallocated frame, stays on the GPU and is updated in place every frame
frame = torch.full(shape, BG, dtype=torch.float32, device=device)
pos = 0


def on_frame():
    global pos
    frame[pos:pos + SQUARE, pos:pos + SQUARE] = BG  # erase the square at its old spot

    # move down the diagonal, wrap back to the top-left corner
    pos = (pos + STEP) % (shape[0] - SQUARE)

    frame[pos:pos + SQUARE, pos:pos + SQUARE] = FG
    # tt.update(frame)
    naive_graphic.data = frame.cpu().numpy()


fig.add_animations(on_frame)

if fpl.IMGUI:
    # show fps with imgui overlay
    fig.imgui_show_fps = True

fig.show()


if __name__ == "__main__":
    fpl.loop.run()