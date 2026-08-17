"""
Phase 2 end-to-end test: patched wgpu-native exportable buffer <-> torch.

Run with the patched .so:
    WGPU_LIB_PATH=/path/to/patched/libwgpu_native.so python phase2_shared_buffer_test.py

Proof chain (each step gates the next):
  A. patched .so loads; adapter is the NVIDIA GPU on Vulkan
  B. wgpuDeviceCreateExportableBuffer returns a working *ordinary* wgpu buffer
     (queue.write_buffer -> queue.read_buffer round trip through wgpu only)
  C. the exported fd imports into CUDA; a torch view exists over the SAME
     memory wgpu renders from
  D. torch writes -> wgpu reads them back        (the Branchpoint direction)
  E. wgpu writes  -> torch reads them back       (reverse, for completeness)
  F. teardown in strict ownership order
"""

import ctypes
import sys

import numpy as np
import torch
import wgpu
import wgpu.backends.wgpu_native as wn
from wgpu.backends.wgpu_native import _api as wnapi
from wgpu.backends.wgpu_native._ffi import ffi

try:
    from cuda.bindings import driver as cu
except ImportError:
    from cuda import cuda as cu
import cupy as cp

NBYTES = 4 * 1024 * 1024
NFLOATS = NBYTES // 4


def ck(err, what):
    if isinstance(err, tuple):
        code, *rest = err
    else:
        code, rest = err, []
    if code != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(code)
        raise RuntimeError(f"CUDA error in {what}: {name}")
    return rest[0] if len(rest) == 1 else rest


# --------------------------------------------------------------------------- #
# A. adapter/device on the NVIDIA GPU, Vulkan backend, via the patched .so
# --------------------------------------------------------------------------- #
print(f"[..] wgpu-py {wgpu.__version__}, lib: {wn.lib_path}")
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
info = adapter.info
print(f"[..] adapter: {info}")
assert "vulkan" in str(info.get("backend", info)).lower() or \
       "vulkan" in str(info).lower(), "[FAIL] not on the Vulkan backend"
assert "nvidia" in str(info).lower() or "rtx" in str(info).lower(), \
    "[FAIL] wrong adapter (iGPU/llvmpipe?) -- pin by UUID / power_preference"
device = adapter.request_device_sync()
print("[ok] A: NVIDIA + Vulkan adapter via patched .so")

# --------------------------------------------------------------------------- #
# B. create the exportable buffer through the new symbol
# --------------------------------------------------------------------------- #
lib = ctypes.CDLL(wn.lib_path)
lib.wgpuDeviceCreateExportableBuffer.restype = ctypes.c_void_p
lib.wgpuDeviceCreateExportableBuffer.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
]
lib.wgpuExportableBufferFreeMemory.restype = None
lib.wgpuExportableBufferFreeMemory.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

device_ptr = int(ffi.cast("uintptr_t", device._internal))
USAGE = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX
         | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

fd_out = ctypes.c_int32(-1)
alloc_out = ctypes.c_uint64(0)
mem_out = ctypes.c_uint64(0)
raw_buf = lib.wgpuDeviceCreateExportableBuffer(
    device_ptr, NBYTES, int(USAGE),
    ctypes.byref(fd_out), ctypes.byref(alloc_out), ctypes.byref(mem_out))
assert raw_buf, "[FAIL] wgpuDeviceCreateExportableBuffer returned NULL"
fd, alloc_size, vk_memory = fd_out.value, alloc_out.value, mem_out.value
print(f"[ok] B1: exportable buffer created (fd={fd}, allocSize={alloc_size}, "
      f"vkMemory={vk_memory:#x})")

# Wrap the raw handle as a wgpu-py GPUBuffer so all normal APIs work on it.
internal = ffi.cast("WGPUBuffer", raw_buf)
gpubuf = wnapi.GPUBuffer("branchpoint-shared", internal, device,
                         NBYTES, int(USAGE), "unmapped")

# Pure-wgpu round trip: proves create_buffer_from_hal wrapping is sound.
patA = np.full(NFLOATS, 42.5, dtype=np.float32)
device.queue.write_buffer(gpubuf, 0, patA.tobytes())
back = np.frombuffer(device.queue.read_buffer(gpubuf), dtype=np.float32)
assert np.array_equal(back, patA), "[FAIL] wgpu-only round trip broken"
print("[ok] B2: buffer behaves as a first-class wgpu buffer (write/read via wgpu)")

# --------------------------------------------------------------------------- #
# C. CUDA import + torch view (same recipe as the passed Phase 1 spike)
# --------------------------------------------------------------------------- #
ck(cu.cuInit(0), "cuInit")
torch.cuda.init()
_ = torch.zeros(1, device="cuda")
code, cur = cu.cuCtxGetCurrent()
if code != cu.CUresult.CUDA_SUCCESS or int(cur) == 0:
    dev0 = ck(cu.cuDeviceGet(0), "cuDeviceGet")
    ctx = ck(cu.cuDevicePrimaryCtxRetain(dev0), "cuDevicePrimaryCtxRetain")
    ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

hdesc = cu.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
hdesc.type = cu.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
hdesc.handle.fd = fd
hdesc.size = alloc_size                       # driver allocation size, not NBYTES
ext_mem = ck(cu.cuImportExternalMemory(hdesc), "cuImportExternalMemory")

bdesc = cu.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
bdesc.offset, bdesc.size, bdesc.flags = 0, NBYTES, 0
dptr = ck(cu.cuExternalMemoryGetMappedBuffer(ext_mem, bdesc),
          "cuExternalMemoryGetMappedBuffer")

umem = cp.cuda.UnownedMemory(int(dptr), NBYTES, owner=None)
carr = cp.ndarray((NFLOATS,), dtype=cp.float32,
                  memptr=cp.cuda.MemoryPointer(umem, 0))
t_shared = torch.from_dlpack(carr)
print(f"[ok] C: CUDA import + torch view over the wgpu buffer ({int(dptr):#x})")

# --------------------------------------------------------------------------- #
# D. torch writes -> wgpu reads  (the direction Branchpoint uses every frame)
# --------------------------------------------------------------------------- #
expected = torch.arange(NFLOATS, dtype=torch.float32, device="cuda") * 0.25
t_shared.copy_(expected)               # THE one D2D copy
t_shared[0], t_shared[-1] = 1234.5, 6789.25
torch.cuda.synchronize()               # v1 coarse sync

got = np.frombuffer(device.queue.read_buffer(gpubuf), dtype=np.float32)
exp = expected.cpu().numpy().copy()
exp[0], exp[-1] = 1234.5, 6789.25
assert np.array_equal(got, exp), "[FAIL] torch writes not visible through wgpu"
print("[ok] D: *** wgpu read exactly what torch wrote — zero host bytes ***")

# --------------------------------------------------------------------------- #
# E. wgpu writes -> torch reads
# --------------------------------------------------------------------------- #
patB = np.linspace(-1, 1, NFLOATS, dtype=np.float32)
device.queue.write_buffer(gpubuf, 0, patB.tobytes())
enc = device.create_command_encoder()
device.queue.submit([enc.finish()])   # force the staged write to execute
device._poll(block=True)              # wait for GPU completion
back2 = t_shared.cpu().numpy()
assert np.array_equal(back2, patB), "[FAIL] wgpu writes not visible to torch"
print("[ok] E: torch read exactly what wgpu wrote")

# --------------------------------------------------------------------------- #
# F. teardown in strict ownership order (design doc Section 5)
# --------------------------------------------------------------------------- #
del t_shared, carr, umem                           # 1. torch/cupy views
ck(cu.cuDestroyExternalMemory(ext_mem), "cuDestroyExternalMemory")  # 2. import
gpubuf.destroy()                                   # 3. wgpu buffer (VkBuffer)
del gpubuf
for _ in range(3):
    device._poll()                                 #    let drops flush
lib.wgpuExportableBufferFreeMemory(device_ptr, vk_memory)  # 4. VkDeviceMemory
print("[ok] F: clean teardown")
print("\nPHASE 2 PASSED — SharedTensorBuffer backend is viable. "
      "Next: fastplotlib demo (torch-written heatmap).")
