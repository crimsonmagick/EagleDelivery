#!/usr/bin/env bash
set -euo pipefail

for map in emu_campus emu_detour; do
    for task in "S W" "L D" "H P"; do
        read -r start goal <<< "$task"

        for algorithm in bfs astar; do
            echo "===== $map | $start -> $goal | $algorithm ====="

            python3 run_project.py search \
                --map "maps/${map}.txt" \
                --start "$start" \
                --goal "$goal" \
                --algorithm "$algorithm" \
                | grep -E 'Expanded:|Path length:'

            echo
        done
    done
done