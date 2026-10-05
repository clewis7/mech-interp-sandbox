#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=results/nsys
mkdir -p "$OUT_DIR"

# largest size per primitive
CASES=(
    "image 4096"
#    "scatter 5000000"
#    "line 5000000"
)

for backend in shared naive; do
    for case in "${CASES[@]}"; do
        read -r primitive size <<< "$case"
        name="${backend}_${primitive}_${size}"

        python -c "import torch; torch.zeros(1, device='cuda')" || { echo "CUDA wedged, stopping"; exit 1; }

        nsys profile \
            --trace=cuda,vulkan,osrt \
            --delay=5 \
            --force-overwrite=true \
            -o "$OUT_DIR/$name" \
            python run.py --backend "$backend" --primitive "$primitive" --size "$size" \
                --mode none --warmup 0 --duration 10 --out "$OUT_DIR/_ignore.csv" \
            || echo "nsys profile exited non-zero for $name, continuing"

        nsys stats \
            --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_gpu_kern_sum \
            --format csv \
            --force-export=true \
            --output "$OUT_DIR/$name" \
            "$OUT_DIR/$name.nsys-rep" \
            || echo "nsys stats failed for $name, continuing"
    done
done