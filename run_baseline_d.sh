#!/usr/bin/env bash
set -euo pipefail
python3 run_project.py train --start S --goal W --output results/baseline_SW
python3 run_project.py train --start L --goal D --output results/baseline_LD
python3 run_project.py train --start H --goal P --output results/baseline_HP