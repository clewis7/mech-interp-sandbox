"""
Common update interface for the benchmark: three ways to get a CUDA tensor
onto a wgpu texture.

Every writer implements the SAME contract:

    w = make_writer(kind, device, height, width)
    w.update(cuda_tensor)   # returns once the pixels are in the texture
    w.close()

Fairness rules baked in here, because the benchmark is only interesting if the
comparison is honest:

  * All writers end at the SAME kind of destination -- a real wgpu texture in
    the format pygfx samples -- and all submit their work.  Comparing a bare
    copy against copy+blit would flatter the host path.
  * update() is synchronous by construction: it returns when the pixels are
    ready, which is the latency a 60 fps frame loop actually experiences.
  * The host path is implemented WELL (pinned staging buffer, reused, no
    per-frame allocation).  A strawman `.cpu()` version is included separately
    as `host_naive` so the post can show both, but `host` is the fair rival.

These writers are standalone (they own their textures) so the benchmark does
not depend on pygfx scene setup.  The production path in branchpoint.gpu
follows the same shape.
"""

from __future__ import annotations

import numpy as np
import torch
import wgpu

# Row pitch rule for copy_buffer_to_texture / write_texture: bytes_per_row must
# be a multiple of 256.
ROW_ALIGN_BYTES = 256
F32 = 4


def padded_row_texels(width: int, bytes_per_texel: int = F32) -> int:
    """Texels per row after padding to the 256-byte pitch."""
    row_bytes = width * bytes_per_texel
    padded = ((row_bytes + ROW_ALIGN_BYTES - 1) // ROW_ALIGN_BYTES) * ROW_ALIGN_BYTES
    return padded // bytes_per_texel


def make_texture(device, width, height):
    return device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.r32float,
        usage=wgpu.TextureUsage.COPY_DST
              | wgpu.TextureUsage.TEXTURE_BINDING
              | wgpu.TextureUsage.COPY_SRC,   # readback for verification
    )


class Writer:
    """Interface + shared bookkeeping."""

    #: bytes that cross PCIe per update (0 for the shared path)
    host_bytes_per_update: int = 0

    def __init__(self, device, height: int, width: int):
        self.device = device
        self.height = int(height)
        self.width = int(width)
        self.row_texels = padded_row_texels(self.width)
        self.texture = make_texture(device, self.width, self.height)

    def update(self, src: torch.Tensor) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Shared-memory path: torch -> dual-mapped buffer -> blit -> texture
# --------------------------------------------------------------------------- #

class SharedWriter(Writer):
    """The Branchpoint path. Two device-side ops, zero host bytes."""

    name = "shared"
    host_bytes_per_update = 0

    def __init__(self, device, height: int, width: int):
        super().__init__(device, height, width)
        from branchpoint.gpu.torch_shared import SharedTensorBuffer

        self.buf = SharedTensorBuffer(device, (self.height, self.row_texels))

    def update(self, src: torch.Tensor) -> None:
        # 1. one fused D2D copy (cast + contiguity + relocation)
        self.buf.view[:, : self.width].copy_(src)
        self.buf._event.record()
        # 2. order CUDA writes before the Vulkan read
        self.buf.sync()
        # 3. GPU-side blit into the tiled texture
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_texture(
            {"buffer": self.buf.gpu_buffer, "offset": 0,
             "bytes_per_row": F32 * self.row_texels,
             "rows_per_image": self.height},
            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
            (self.width, self.height, 1),
        )
        self.device.queue.submit([enc.finish()])

    def close(self) -> None:
        self.buf.close()


# --------------------------------------------------------------------------- #
# Host paths
# --------------------------------------------------------------------------- #

class HostPinnedWriter(Writer):
    """The fair rival: pinned staging buffer, reused, no per-frame allocation.

    Frame cost = D2H over PCIe + write_texture upload over PCIe.  The pinned
    buffer is what any competent implementation would do, so this -- not the
    naive version -- is the number the post should headline.
    """

    name = "host"

    def __init__(self, device, height: int, width: int):
        super().__init__(device, height, width)
        # Persistent pinned host staging, padded to the row pitch so the
        # upload can go straight from it.
        self.staging = torch.empty(
            (self.height, self.row_texels), dtype=torch.float32,
            device="cpu", pin_memory=True,
        )
        self.staging_np = self.staging.numpy()
        self.host_bytes_per_update = 2 * self.height * self.width * F32

    def update(self, src: torch.Tensor) -> None:
        # 1. D2H into pinned memory (async, then wait -- pixels must be ready)
        self.staging[:, : self.width].copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        # 2. H2D upload straight into the texture
        self.device.queue.write_texture(
            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
            self.staging_np,                       # zero-copy view of pinned mem
            {"offset": 0, "bytes_per_row": F32 * self.row_texels,
             "rows_per_image": self.height},
            (self.width, self.height, 1),
        )
        # force the staged write to execute, so "pixels ready" is true on return
        self.device.queue.submit([self.device.create_command_encoder().finish()])
        self.device._poll(block=True)


class HostNaiveWriter(Writer):
    """The strawman: `t.cpu().numpy()` every frame.

    Included only so the post can show what most notebooks actually do -- a
    fresh pageable host allocation per frame plus a sync.  Not the fair
    comparison; report it as a third line, clearly labelled.
    """

    name = "host_naive"

    def __init__(self, device, height: int, width: int):
        super().__init__(device, height, width)
        self.host_bytes_per_update = 2 * self.height * self.width * F32

    def update(self, src: torch.Tensor) -> None:
        arr = src.detach().to(torch.float32).cpu().numpy()   # alloc + D2H + sync
        if self.row_texels != self.width:
            padded = np.zeros((self.height, self.row_texels), dtype=np.float32)
            padded[:, : self.width] = arr
            arr = padded
        else:
            arr = np.ascontiguousarray(arr)
        self.device.queue.write_texture(
            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
            arr,
            {"offset": 0, "bytes_per_row": F32 * self.row_texels,
             "rows_per_image": self.height},
            (self.width, self.height, 1),
        )
        self.device.queue.submit([self.device.create_command_encoder().finish()])
        self.device._poll(block=True)


WRITERS = {
    "shared": SharedWriter,
    "host": HostPinnedWriter,
    "host_naive": HostNaiveWriter,
}


def make_writer(kind: str, device, height: int, width: int) -> Writer:
    try:
        cls = WRITERS[kind]
    except KeyError:
        raise ValueError(f"unknown writer {kind!r}; pick from {list(WRITERS)}")
    return cls(device, height, width)


def get_offscreen_device():
    """A wgpu device with no window: benchmarks must not be vsync-bound.

    branchpoint must already be imported (it sets WGPU_LIB_PATH) -- importing
    it here rather than relying on the caller keeps the ordering rule local.
    """
    import branchpoint  # noqa: F401  (import-time: points wgpu at the patch)

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    if "nvidia" not in str(info).lower():
        raise RuntimeError(f"benchmark needs the NVIDIA adapter, got: {info}")
    return adapter.request_device_sync(), info
