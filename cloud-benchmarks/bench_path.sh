#!/usr/bin/env bash
# Measure raw throughput + latency of a Jupyter connection path.
# Downloads test files through Jupyter's /files/ endpoint with curl.
#
# Usage:
#   ./bench_path.sh <label> <base_url> <token>
#
# SSH tunnel:  ./bench_path.sh ssh_tunnel http://localhost:8888 <JUPYTER_TOKEN>
# JupyterHub:  ./bench_path.sh jupyterhub https://<HUB_HOST>/user/clewis <HUB_API_TOKEN>

LABEL=$1
BASE_URL=$2
TOKEN=$3

N_BIG=5
N_SMALL=20
OUT="path_${LABEL}_$(date +%Y%m%d_%H%M%S).csv"

echo "label,test,run,bytes,total_s,mb_per_s,ttfb_s" > "$OUT"

# ---------- throughput: 50 MB file
for i in $(seq 1 $N_BIG); do
    read bytes total speed ttfb < <(curl -s -o /dev/null \
        -H "Authorization: token $TOKEN" \
        -w "%{size_download} %{time_total} %{speed_download} %{time_starttransfer}\n" \
        "$BASE_URL/files/bench/big.bin")
    if [ "$bytes" -lt 52000000 ]; then
        echo "ERROR: got $bytes bytes instead of ~50 MB (auth/redirect issue?). Check with:"
        echo "  curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' -H 'Authorization: token <TOKEN>' $BASE_URL/files/bench/big.bin"
        rm "$OUT"
        exit 1
    fi
    mbps=$(awk "BEGIN {print $speed / 1048576}")
    echo "$LABEL,throughput,$i,$bytes,$total,$mbps,$ttfb" >> "$OUT"
    echo "throughput run $i: $mbps MB/s"
done

# ---------- latency: 1 KB file
for i in $(seq 1 $N_SMALL); do
    read bytes total speed ttfb < <(curl -s -o /dev/null \
        -H "Authorization: token $TOKEN" \
        -w "%{size_download} %{time_total} %{speed_download} %{time_starttransfer}\n" \
        "$BASE_URL/files/bench/small.bin")
    echo "$LABEL,latency,$i,$bytes,$total,,$ttfb" >> "$OUT"
done

# ---------- summary
echo ""
echo "saved: $OUT"
awk -F, '$2=="throughput" {print $6}' "$OUT" | sort -n | awk '{a[NR]=$1} END {print "median throughput:", a[int((NR+1)/2)], "MB/s"}'
awk -F, '$2=="latency" {print $7 * 1000}' "$OUT" | sort -n | awk '{a[NR]=$1} END {print "median latency (ttfb):", a[int((NR+1)/2)], "ms"}'
