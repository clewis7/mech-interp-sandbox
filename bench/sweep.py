"""
Timing sweep: cost of getting a CUDA tensor onto a wgpu texture, per backend.

    python -m bench.sweep --out results/sweep.json

Measures wall-clock latency of a full update() -- the number a 60 fps frame
loop actually experiences -- across writer x size x dtype, and derives
effective bandwidth.  Writes one self-documenting JSON (versions, GPU, driver)
so the figures are reproducible months later.

Protocol notes:
  * medians and p10/p90, never means: one driver hiccup ruins a mean.
  * warmup per cell: first updates pay allocator, context and driver costs
    that steady state never sees again.
  * conditions are INTERLEAVED across repeats, not run in blocks, so thermal
    drift on a laptop cannot systematically favour whichever ran first.
  * allocation delta per cell catches accidental per-frame allocations.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

import branchpoint  # noqa: F401  -- MUST precede wgpu; sets WGPU_LIB_PATH
import torch

torch.zeros(1, device="cuda")     # force torch's primary context first
print(torch.cuda.current_device())

from bench.writers import get_offscreen_device, make_writer

DEFAULT_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_WRITERS = ["shared", "host", "host_naive"]
DEFAULT_DTYPES = ["float32", "bfloat16"]


def time_cell(device, kind: str, size: int, dtype: torch.dtype,
              iters: int, warmup: int) -> dict:
    """Time `iters` updates of a size x size tensor through one writer."""
    src = torch.rand(size, size, device="cuda", dtype=torch.float32).to(dtype)

    with make_writer(kind, device, size, size) as w:
        import gc
        gc.collect()
        device._poll(block=True)
        torch.cuda.empty_cache()

        for _ in range(warmup):
            w.update(src)
        torch.cuda.synchronize()

        alloc_before = torch.cuda.memory_allocated()
        samples = []
        for _ in tqdm(range(iters)):
            t0 = time.perf_counter()
            w.update(src)              # writers are synchronous by contract
            samples.append((time.perf_counter() - t0) * 1e3)   # ms
        alloc_after = torch.cuda.memory_allocated()

        host_bytes = w.host_bytes_per_update

    samples.sort()
    med = statistics.median(samples)
    payload_bytes = size * size * torch.tensor([], dtype=dtype).element_size()

    return {
        "writer": kind,
        "size": size,
        "dtype": str(dtype).replace("torch.", ""),
        "iters": iters,
        "ms_median": med,
        "ms_p10": samples[int(0.10 * len(samples))],
        "ms_p90": samples[int(0.90 * len(samples))],
        "ms_min": samples[0],
        "payload_bytes": payload_bytes,
        "host_bytes_per_update": host_bytes,
        # effective throughput of the payload, GB/s
        "gbps": payload_bytes / (med * 1e-3) / 1e9,
        "alloc_delta_bytes": alloc_after - alloc_before,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("./results/sweep.json"))
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--writers", nargs="+", default=DEFAULT_WRITERS)
    ap.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3,
                    help="passes over the whole grid; results are per pass, "
                         "interleaved to spread thermal drift")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device, adapter_info = get_offscreen_device()

    import wgpu
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "wgpu_py": wgpu.__version__,
        "adapter": str(adapter_info),
        "shared_backend_active": True,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    print(json.dumps(meta, indent=2))
    if not meta["shared_backend_active"]:
        raise SystemExit("patched wgpu-native not active -- nothing to compare")

    rows = []
    for rep in range(args.repeats):
        for size in args.sizes:
            for dname in args.dtypes:
                dtype = getattr(torch, dname)
                for kind in args.writers:      # interleaved, not blocked
                    row = time_cell(device, kind, size, dtype,
                                    args.iters, args.warmup)
                    row["repeat"] = rep
                    rows.append(row)
                    print(f"rep{rep} {kind:11s} {size:5d}^2 {dname:9s} "
                          f"{row['ms_median']:8.3f} ms  "
                          f"{row['gbps']:7.2f} GB/s"
                          + ("  [alloc leak!]" if row["alloc_delta_bytes"] else ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
