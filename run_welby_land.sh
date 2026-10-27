#!/usr/bin/env bash
set -euo pipefail

for gamma in 0.5 0.8 0.95; do
    output="results/welby_land_gamma_${gamma}"

    python3 run_project.py train \
        --map maps/welby_land.txt \
        --start S \
        --goal W \
        --gamma "$gamma" \
        --output "$output"

    python3 summarize_results.py "$output"
done