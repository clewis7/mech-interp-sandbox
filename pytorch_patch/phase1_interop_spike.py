"""
Phase 1 derisk spike: Vulkan <-> CUDA external-memory interop, no wgpu involved.

Proves, on this exact machine/driver:
  1. Vulkan can allocate an exportable DEVICE_LOCAL buffer (OPAQUE_FD) on the
     NVIDIA GPU (UUID-pinned so the Intel iGPU / llvmpipe are never selected).
  2. The exported fd imports into CUDA (cuImportExternalMemory) and maps to a
     device pointer.
  3. A *torch* tensor view of that pointer can be written by ordinary torch ops.
  4. Vulkan observes the bytes torch wrote (readback through a Vulkan
     staging-buffer copy, NOT through CUDA -- this is the actual proof).
  5. The reverse direction: Vulkan writes, torch reads.

If every step prints [ok] and the final asserts pass, Phase 2 (wgpu-native
patch) is justified. If it fails, note WHERE -- each failure mode points at a
different culprit (driver, extension, memory type, cupy, sync).
"""

import ctypes
import struct
import sys

import numpy as np
import torch
from vulkan import *  # noqa: F403  (the python `vulkan` package is star-import styled)

# cuda-python moved modules around across versions; shim both layouts.
try:
    from cuda.bindings import driver as cu
except ImportError:
    from cuda import cuda as cu

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False
    print("[warn] cupy not found -- torch-view test will be skipped; "
          "driver-level memcpy test still runs")

NBYTES = 4 * 1024 * 1024  # 4 MiB shared region (1M float32)
NFLOATS = NBYTES // 4


def ck(err, what):
    """Check a cuda-python driver call result."""
    if isinstance(err, tuple):  # cuda-python returns (err, *results)
        code, *rest = err
    else:
        code, rest = err, []
    if code != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(code)
        raise RuntimeError(f"CUDA error in {what}: {name}")
    return rest[0] if len(rest) == 1 else rest


# --------------------------------------------------------------------------- #
# Step 0: CUDA device UUID (ground truth for pinning the Vulkan device)
# --------------------------------------------------------------------------- #
ck(cu.cuInit(0), "cuInit")
cu_dev = ck(cu.cuDeviceGet(0), "cuDeviceGet")
cu_uuid = bytes(ck(cu.cuDeviceGetUuid(cu_dev), "cuDeviceGetUuid").bytes)
print(f"[ok] CUDA device 0 UUID: {cu_uuid.hex()}")

# --------------------------------------------------------------------------- #
# Step 1: Vulkan instance + UUID-pinned physical device
# --------------------------------------------------------------------------- #
app = VkApplicationInfo(
    pApplicationName="interop-spike",
    applicationVersion=VK_MAKE_VERSION(0, 0, 1),
    pEngineName="none",
    engineVersion=0,
    apiVersion=VK_MAKE_VERSION(1, 1, 0),  # 1.1: gets us GetPhysicalDeviceProperties2 core
)
instance = vkCreateInstance(
    VkInstanceCreateInfo(pApplicationInfo=app, enabledLayerCount=0), None
)

phys = None
for pd in vkEnumeratePhysicalDevices(instance):
    id_props = VkPhysicalDeviceIDProperties()
    props2 = VkPhysicalDeviceProperties2(pNext=id_props)
    vkGetPhysicalDeviceProperties2(pd, props2)
    dev_uuid = bytes(id_props.deviceUUID)
    name = props2.properties.deviceName
    tag = "  <-- MATCH" if dev_uuid == cu_uuid else ""
    print(f"     vk device: {name}  uuid={dev_uuid.hex()}{tag}")
    if dev_uuid == cu_uuid:
        phys = pd
if phys is None:
    sys.exit("[FAIL] no Vulkan physical device matches the CUDA UUID -- "
             "driver/ICD problem, stop here")
print("[ok] Vulkan physical device pinned by UUID")

# --------------------------------------------------------------------------- #
# Step 2: confirm extensions, create logical device + a transfer-capable queue
# --------------------------------------------------------------------------- #
exts = {e.extensionName for e in vkEnumerateDeviceExtensionProperties(phys, None)}
need = {"VK_KHR_external_memory", "VK_KHR_external_memory_fd"}
missing = need - exts
if missing:
    sys.exit(f"[FAIL] missing device extensions: {missing}")
print("[ok] VK_KHR_external_memory(_fd) present on the NVIDIA ICD")

qfams = vkGetPhysicalDeviceQueueFamilyProperties(phys)
qfi = next(i for i, f in enumerate(qfams)
           if f.queueFlags & (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT
                              | VK_QUEUE_GRAPHICS_BIT))
device = vkCreateDevice(
    phys,
    VkDeviceCreateInfo(
        queueCreateInfoCount=1,
        pQueueCreateInfos=[VkDeviceQueueCreateInfo(
            queueFamilyIndex=qfi, queueCount=1, pQueuePriorities=[1.0])],
        ppEnabledExtensionNames=sorted(need),
    ),
    None,
)
queue = vkGetDeviceQueue(device, qfi, 0)
vkGetMemoryFdKHR = vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR")
print(f"[ok] logical device created (queue family {qfi})")

mem_props = vkGetPhysicalDeviceMemoryProperties(phys)


def pick_mem_type(type_bits, want_flags):
    for i in range(mem_props.memoryTypeCount):
        if (type_bits & (1 << i)) and \
           (mem_props.memoryTypes[i].propertyFlags & want_flags) == want_flags:
            return i
    raise RuntimeError(f"no memory type with flags {want_flags:#x} "
                       f"in bits {type_bits:#x}")


# --------------------------------------------------------------------------- #
# Step 3: exportable DEVICE_LOCAL buffer + dedicated allocation + fd
# --------------------------------------------------------------------------- #
ext_buf_info = VkExternalMemoryBufferCreateInfo(
    handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
shared_buf = vkCreateBuffer(
    device,
    VkBufferCreateInfo(
        pNext=ext_buf_info,
        size=NBYTES,
        usage=(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    ),
    None,
)
reqs = vkGetBufferMemoryRequirements(device, shared_buf)
print(f"[ok] exportable VkBuffer: logical={NBYTES} allocSize={reqs.size} "
      f"align={reqs.alignment} typeBits={reqs.memoryTypeBits:#x}")

export_info = VkExportMemoryAllocateInfo(
    handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
dedicated = VkMemoryDedicatedAllocateInfo(pNext=export_info, buffer=shared_buf)
shared_mem = vkAllocateMemory(
    device,
    VkMemoryAllocateInfo(
        pNext=dedicated,
        allocationSize=reqs.size,
        memoryTypeIndex=pick_mem_type(reqs.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    ),
    None,
)
vkBindBufferMemory(device, shared_buf, shared_mem, 0)

fd = vkGetMemoryFdKHR(device, VkMemoryGetFdInfoKHR(
    memory=shared_mem, handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT))
print(f"[ok] exported opaque fd = {fd}")

# --------------------------------------------------------------------------- #
# Step 4: CUDA import (fd is CONSUMED here -- never close/import it again)
# --------------------------------------------------------------------------- #
# Make sure torch's primary context is current on this thread before driver calls.
torch.cuda.init()
_ = torch.zeros(1, device="cuda")
code, cur_ctx = cu.cuCtxGetCurrent()
if code != cu.CUresult.CUDA_SUCCESS or int(cur_ctx) == 0:
    ctx = ck(cu.cuDevicePrimaryCtxRetain(cu_dev), "cuDevicePrimaryCtxRetain")
    ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

hdesc = cu.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
hdesc.type = cu.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
hdesc.handle.fd = fd
hdesc.size = reqs.size          # MUST be the Vulkan allocationSize, not NBYTES
ext_mem = ck(cu.cuImportExternalMemory(hdesc), "cuImportExternalMemory")

bdesc = cu.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
bdesc.offset = 0
bdesc.size = NBYTES
bdesc.flags = 0
dptr = ck(cu.cuExternalMemoryGetMappedBuffer(ext_mem, bdesc),
          "cuExternalMemoryGetMappedBuffer")
print(f"[ok] CUDA mapped device pointer = {int(dptr):#x}")

# --------------------------------------------------------------------------- #
# Step 5: CUDA -> Vulkan.  torch writes a pattern; Vulkan reads it back.
# --------------------------------------------------------------------------- #
expected = np.arange(NFLOATS, dtype=np.float32)
expected[0], expected[-1] = 1234.5, 6789.25  # unmistakable end markers

if HAVE_CUPY:
    umem = cp.cuda.UnownedMemory(int(dptr), NBYTES, owner=None)
    carr = cp.ndarray((NFLOATS,), dtype=cp.float32,
                      memptr=cp.cuda.MemoryPointer(umem, 0))
    t_shared = torch.from_dlpack(carr)
    src = torch.from_numpy(expected).cuda()      # normal torch tensor
    t_shared.copy_(src)                          # THE one D2D copy
    t_shared[0], t_shared[-1] = 1234.5, 6789.25  # in-place torch ops on shared mem
    print("[ok] torch wrote pattern into shared memory (D2D copy_ + in-place ops)")
else:
    host = expected.tobytes()
    ck(cu.cuMemcpyHtoD(dptr, host, NBYTES), "cuMemcpyHtoD")
    print("[ok] driver-API wrote pattern into shared memory (cupy absent)")

torch.cuda.synchronize()  # v1 coarse sync: CUDA done before Vulkan reads

# Vulkan-side readback via HOST_VISIBLE staging buffer + copy command
staging_buf = vkCreateBuffer(device, VkBufferCreateInfo(
    size=NBYTES, usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)
sreqs = vkGetBufferMemoryRequirements(device, staging_buf)
staging_mem = vkAllocateMemory(device, VkMemoryAllocateInfo(
    allocationSize=sreqs.size,
    memoryTypeIndex=pick_mem_type(
        sreqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)),
    None)
vkBindBufferMemory(device, staging_buf, staging_mem, 0)

cmd_pool = vkCreateCommandPool(device, VkCommandPoolCreateInfo(
    flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    queueFamilyIndex=qfi), None)
cmd = vkAllocateCommandBuffers(device, VkCommandBufferAllocateInfo(
    commandPool=cmd_pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount=1))[0]
fence = vkCreateFence(device, VkFenceCreateInfo(), None)


def run_copy(src, dst):
    vkResetCommandBuffer(cmd, 0)
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    vkCmdCopyBuffer(cmd, src, dst,
                    1, [VkBufferCopy(srcOffset=0, dstOffset=0, size=NBYTES)])
    vkEndCommandBuffer(cmd)
    vkResetFences(device, 1, [fence])
    vkQueueSubmit(queue, 1, [VkSubmitInfo(
        commandBufferCount=1, pCommandBuffers=[cmd])], fence)
    vkWaitForFences(device, 1, [fence], VK_TRUE, int(5e9))


run_copy(shared_buf, staging_buf)  # shared -> staging
mapped = vkMapMemory(device, staging_mem, 0, NBYTES, 0)
got = np.frombuffer(mapped, dtype=np.float32, count=NFLOATS).copy()
vkUnmapMemory(device, staging_mem)

assert got[0] == 1234.5 and got[-1] == 6789.25, \
    f"[FAIL] end markers wrong: {got[0]}, {got[-1]}"
assert np.array_equal(got[1:-1], expected[1:-1]), \
    "[FAIL] interior bytes mismatch -- CUDA writes NOT visible to Vulkan"
print("[ok] *** Vulkan read back exactly what torch wrote (CUDA -> Vulkan) ***")

# --------------------------------------------------------------------------- #
# Step 6: Vulkan -> CUDA.  Vulkan writes; torch reads.
# --------------------------------------------------------------------------- #
reverse = (np.arange(NFLOATS, dtype=np.float32) * -0.5).astype(np.float32)
mapped = vkMapMemory(device, staging_mem, 0, NBYTES, 0)
mapped[:NBYTES] = reverse.tobytes()
vkUnmapMemory(device, staging_mem)
run_copy(staging_buf, shared_buf)  # staging -> shared (fence-waited)

if HAVE_CUPY:
    back = t_shared.cpu().numpy()  # read THROUGH the torch view of shared mem
else:
    buf = bytearray(NBYTES)
    ck(cu.cuMemcpyDtoH(buf, dptr, NBYTES), "cuMemcpyDtoH")
    back = np.frombuffer(bytes(buf), dtype=np.float32)
assert np.array_equal(back, reverse), "[FAIL] Vulkan writes NOT visible to CUDA"
print("[ok] *** torch read back exactly what Vulkan wrote (Vulkan -> CUDA) ***")

# --------------------------------------------------------------------------- #
# Step 7: teardown in reverse ownership order (Section 5 of the design doc)
# --------------------------------------------------------------------------- #
if HAVE_CUPY:
    del t_shared, carr, umem
ck(cu.cuDestroyExternalMemory(ext_mem), "cuDestroyExternalMemory")
vkDestroyFence(device, fence, None)
vkDestroyCommandPool(device, cmd_pool, None)
vkDestroyBuffer(device, staging_buf, None)
vkFreeMemory(device, staging_mem, None)
vkDestroyBuffer(device, shared_buf, None)
vkFreeMemory(device, shared_mem, None)
vkDestroyDevice(device, None)
vkDestroyInstance(instance, None)
print("[ok] clean teardown")
print("\nPHASE 1 PASSED -- proceed to the wgpu-native patch (Phase 2).")
