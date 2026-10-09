OUT=results/panzoom.csv

for rep in 0 1 2; do
  for backend in shared naive; do
    for size in 256 512 1024 2048 4096; do
      python -c "import torch; torch.zeros(1, device='cuda')" || { echo "CUDA wedged, stopping"; break 3; }
      timeout 60 python run.py --backend "$backend" --primitive image --size "$size" \
        --mode panzoom --rep "$rep" --warmup 2 --duration 5 --out "$OUT"
    done
    for primitive in scatter line; do
      # N = side² so points match the image MB/frame exactly
      for size in 65536 262144 1048576 4194304 16777216; do
        python -c "import torch; torch.zeros(1, device='cuda')" || { echo "CUDA wedged, stopping"; break 4; }
        timeout 60 python run.py --backend "$backend" --primitive "$primitive" --size "$size" \
          --mode panzoom --rep "$rep" --warmup 2 --duration 5 --out "$OUT"
      done
    done
  done
done