# bench/verify.py
import branchpoint, torch, numpy as np
from bench.writers import get_offscreen_device, make_writer

device, _ = get_offscreen_device()
SIZE, N = 2048, 200
w = make_writer("shared", device, SIZE, SIZE)

bad = 0
for i in range(N):
    # unique per-iteration pattern so a stale frame is detectable
    src = torch.full((SIZE, SIZE), float(i % 64), device="cuda", dtype=torch.bfloat16)
    src[0, 0] = -128.0  # exactly representable in bf16
    src[-1, -1] = 64.0


    w.update(src)

    got = np.frombuffer(
        device.queue.read_texture(
            {"texture": w.texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": 4 * w.row_texels, "rows_per_image": SIZE},
            (SIZE, SIZE, 1),
        ), dtype=np.float32,
    ).reshape(SIZE, w.row_texels)[:, :SIZE]

    # compare against what bf16 actually holds:
    expected = src.float().cpu().numpy()
    if not np.array_equal(got, expected):
        bad += 1
        print(got[0, 0], got[-1, -1], got[SIZE // 2, SIZE // 2])
print(f"{bad}/{N} bad frames")