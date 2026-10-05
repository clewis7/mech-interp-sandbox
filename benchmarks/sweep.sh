#!/usr/bin/env bash
set -euo pipefail

OUT=results/results.csv
REPS=1
WARMUP=2
DURATION=5

MODES=(none pan zoom)
IMAGE_SIZES=(256 512 1024 2048 4096)
POINT_SIZES=(10000 100000 1000000 3000000, 5000000)

run() {
    python run.py --backend "$1" --primitive "$2" --size "$3" --mode "$4" \
        --rep "$5" --warmup "$WARMUP" --duration "$DURATION" --out "$OUT"
}

for rep in $(seq 0 $((REPS - 1))); do
    for mode in "${MODES[@]}"; do
        for backend in shared naive; do
            for size in "${IMAGE_SIZES[@]}"; do
                run "$backend" image "$size" "$mode" "$rep"
            done
            for primitive in scatter line; do
                for size in "${POINT_SIZES[@]}"; do
                    run "$backend" "$primitive" "$size" "$mode" "$rep"
                done
            done
        done
    done
done