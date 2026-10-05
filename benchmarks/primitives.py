import numpy as np
import torch
import wgpu

PRIMITIVES = ("image", "scatter", "line")


def _make_image(subplot, size, device):
    """size = side length of a square float32 RGB image."""
    shape = (size, size, 3)
    graphic = subplot.add_image(np.zeros(shape, dtype=np.float32), vmin=0, vmax=1,
                                texture_usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)

    def compute(t, out):
        torch.rand(shape, device=device, out=out)

    return graphic, "texture", shape, compute


def _make_scatter(subplot, size, device):
    """size = number of points."""
    shape = (size, 3)
    base = torch.rand(shape, dtype=torch.float32, device=device) * 100
    base[:, 2] = 0
    graphic = subplot.add_scatter(base.cpu().numpy(), sizes=1)

    def compute(t, out):
        out.copy_(base)
        out[:, 1] += 2 * torch.sin(base[:, 0] * 0.5 + t)

    return graphic, "buffer", shape, compute


def _make_line(subplot, size, device):
    """size = number of points in a single line."""
    shape = (size, 3)
    x = torch.linspace(0, 100, size, device=device)
    init = torch.zeros(shape, dtype=torch.float32, device=device)
    init[:, 0] = x
    graphic = subplot.add_line(init.cpu().numpy(), thickness=1)

    def compute(t, out):
        out[:, 0].copy_(x)
        torch.sin(x * 0.5 + t, out=out[:, 1])
        out[:, 2].zero_()

    return graphic, "buffer", shape, compute


_BUILDERS = {"image": _make_image, "scatter": _make_scatter, "line": _make_line}


def make_primitive(subplot, primitive, size, device):
    """
    Returns (graphic, kind, shape, compute).
    kind: "texture" or "buffer"; compute(t, out) writes the frame in place into a float32 CUDA tensor of `shape`.
    """
    return _BUILDERS[primitive](subplot, size, device)