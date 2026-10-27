# Eagle Delivery

COSC 561 Project 1.

Author: Welby Seely

Eagle Delivery is a simulated campus delivery robot project exploring pathfinding and reinforcement learning. The project implements A* search alongside a provided BFS baseline, as well as tabular Q-learning with epsilon-greedy exploration.

Three maps are used: the original EMU campus map, a detour variation, and a custom W-shaped map called *Welby Land*.

## Requirements

- Python 3.9+
- NumPy, Pandas, and Matplotlib for experimental analysis and plotting if running `python3 analyze_q_learning.py`. Don't forget to set up your venv.

## Project Report

Project report can be found in the base directory under `cosc_561_project_1_welby_seely`.

## Running the Project

### Display a Map

```bash
python3 run_project.py show
python3 run_project.py show --map maps/welby_land.txt
```

### BFS and A* Search

```bash
python3 run_project.py search --algorithm bfs --start S --goal W
python3 run_project.py search --algorithm astar --start S --goal W
```

Use `--map maps/emu_detour.txt` or `--map maps/welby_land.txt` to select another map. `welby_land.txt` is the new custom map made for this project.

### Q-Learning Baseline Experiments

Each experiment uses three independent seeds (0, 1, 2), with a fresh Q-table for each run.

```bash
python3 run_project.py train --start S --goal W --output results/baseline_SW
python3 run_project.py train --start L --goal D --output results/baseline_LD
python3 run_project.py train --start H --goal P --output results/baseline_HP
```

Generate summaries and learning-curve CSVs:

```bash
python3 summarize_results.py results/baseline_SW
python3 summarize_results.py results/baseline_LD
python3 summarize_results.py results/baseline_HP
```

### Custom Map Experiment

Welby Land features a 52-step optimal S→W route through narrow, winding corridors. The experiment investigates how different discount factors affect Q-learning performance. 

This can be run with `./run_welby_land.sh`


## Experiment Settings

| Parameter | Baseline | Custom Experiment |
|---|---|---|
| Training episodes | 2000 | 2000 |
| Evaluation episodes | 100 | 100 |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Maximum steps | 200 | 200 |
| Learning rate (α) | 0.3 | 0.3 |
| Discount factor (γ) | 0.95 | 0.5, 0.8, 0.95 |
| Exploration rate (ε) | 0.2 | 0.2 |
| Slip probability | 0.0 | 0.0 |
| Evaluation ε | 0.0 | 0.0 |

## Results

BFS and A* found identical shortest-path lengths for all tested tasks, with A* expanding fewer nodes in the six baseline comparisons.

All nine baseline Q-learning agents achieved 100% evaluation success and optimal route lengths.

On Welby Land, all three discount-factor settings achieved 100% evaluation success and optimal 52-step routes. Higher discount factors showed somewhat faster initial learning.

Training and evaluation results are saved as CSV files under `results/`.

## Tests

Run the supplied test suite:

```bash
python3 -m unittest discover -v
```