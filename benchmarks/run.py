import argparse
import csv
import time
from pathlib import Path

import mechiviz as mv
import fastplotlib as fpl
import numpy as np
import torch

from interactions import MODES, make_driver
from primitives import PRIMITIVES, make_primitive

BACKENDS = ("shared", "naive")
FIELDS = ["backend", "primitive", "size", "mode", "rep", "n_frames", "fps_mean", "frame_ms_p50", "frame_ms_p99"]
# MODES = ("none",)

def make_target(backend, graphic, kind, shape, device):
    out = torch.empty(shape, dtype=torch.float32, device=device)

    if backend == "shared":
        if kind == "texture":
            wrapper = mv.TorchTensorTexture(shape)
        else:
            wrapper = mv.TorchTensorBuffer(shape)
        adopted = {"done": False}

        def push():
            # adopt on first frame, after pygfx has created the GPU resource
            if not adopted["done"]:
                if kind == "texture":
                    wrapper.texture = graphic.data.buffer[0, 0]
                else:
                    wrapper.buffer = graphic.data.buffer
                adopted["done"] = True
            wrapper.update(out)  # D2D into shared memory + GPU blit, no host

        return out, push, wrapper.close

    def push():
        graphic.data = out.cpu().numpy()  # D2H + H2D every frame

    return out, push, lambda: None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--primitive", choices=PRIMITIVES, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--mode", choices=MODES, default="none")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--out", type=Path, default=Path("results/results.csv"))
    return p.parse_args()


def write_row(args, stamps):
    stamps = np.asarray(stamps)
    intervals_ms = np.diff(stamps) * 1000
    row = {
        "backend": args.backend,
        "primitive": args.primitive,
        "size": args.size,
        "mode": args.mode,
        "rep": args.rep,
        "n_frames": len(stamps),
        "fps_mean": (len(stamps) - 1) / (stamps[-1] - stamps[0]),
        "frame_ms_p50": np.percentile(intervals_ms, 50),
        "frame_ms_p99": np.percentile(intervals_ms, 99),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(f"{args.backend:7s} {args.primitive:8s} {args.size:>10d} {args.mode:5s} "
          f"fps={row['fps_mean']:.1f} p99={row['frame_ms_p99']:.2f}ms")


def main():
    args = parse_args()
    device = torch.device("cuda")

    fig = fpl.Figure(
        size=(1000, 800),
        canvas_kwargs={"max_fps": 10_000, "vsync": False, "update_mode": "continuous"},
    )
    graphic, kind, shape, compute = make_primitive(fig[0, 0], args.primitive, args.size, device)
    out, push, close_target = make_target(args.backend, graphic, kind, shape, device)
    drive = make_driver(args.mode, fig)

    stamps = []
    state = {"t0": None, "done": False}

    def on_frame():
        if state["done"]:
            return
        now = time.perf_counter()
        if state["t0"] is None:
            state["t0"] = now
        elapsed = now - state["t0"]

        compute(elapsed, out)
        push()
        drive(elapsed)

        if elapsed >= args.warmup:
            stamps.append(now)
        if elapsed >= args.warmup + args.duration:
            state["done"] = True
            fpl.loop.call_later(0, fig.canvas.close)

    fig.add_animations(on_frame)
    try:
        fig.show()
        fpl.loop.run()
    finally:
        torch.cuda.synchronize()
        close_target()
    write_row(args, stamps)


if __name__ == "__main__":
    main()