import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = {"S-W": 18, "L-D": 22, "H-P": 16}  # BFS-optimal lengths, hardcoding 'cause I'm lazy

matplotlib.use("Agg")

def read_runs(folder, prefix):
    files = sorted(folder.glob(f"{prefix}_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_seed*.csv in {folder}")
    frames = []
    for file in files:
        seed = int(file.stem.rsplit("seed", 1)[1])
        frame = pd.read_csv(file)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def block_stats(training, block):
    data = training.copy()
    data["block"] = (data["episode"] - 1) // block
    per_seed = (
        data.groupby(["seed", "block"])
        .agg(
            episode_end=("episode", "max"),
            success_rate=("success", "mean"),
            mean_return=("return", "mean"),
        )
        .reset_index()
    )

    successful = data[data["success"] == 1]
    success_steps = (
        successful.groupby(["seed", "block"])["steps"]
        .mean().rename("steps_on_success")
        .reset_index()
    )
    per_seed = per_seed.merge(success_steps, on=["seed", "block"], how="left")
    return per_seed


def plot_training_success(stats, destination):
    fig, ax = plt.subplots(figsize=(9, 5))

    stats = stats.groupby(["task", "seed"]).head(10)
    for task, group in stats.groupby("task", sort=False):
        agg = (
            group.groupby("block")
            .agg(
                x=("episode_end", "max"),
                mean=("success_rate", "mean"),
                sd=("success_rate", "std"),
            )
            .reset_index()
        )

        x = agg["x"].to_numpy()
        y = agg["mean"].to_numpy()
        sd = agg["sd"].fillna(0).to_numpy()

        (line,) = ax.plot(x, y, label=task, linewidth=2)
        ax.fill_between(
            x, y - sd, y + sd,
            alpha=0.13,
            color=line.get_color()
        )

    ax.set(
        xlabel="Training episode",
        ylabel="Training success rate"
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Task")

    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_training_steps(stats, destination):
    fig, ax = plt.subplots(figsize=(9, 5))

    for task, group in stats.groupby("task", sort=False):
        agg = (
            group.groupby("block")
            .agg(
                x=("episode_end", "max"),
                mean=("steps_on_success", "mean"),
                sd=("steps_on_success", "std"),
            )
            .reset_index()
        )

        x = agg["x"].to_numpy()
        y = agg["mean"].to_numpy()
        sd = agg["sd"].fillna(0).to_numpy()

        (line,) = ax.plot(x, y, label=task, linewidth=2)
        ax.fill_between(
            x, y - sd, y + sd,
            alpha=0.13,
            color=line.get_color()
        )

        ax.axhline(
            TASKS[task],
            color=line.get_color(),
            linestyle=":",
            alpha=0.7
        )

    ax.set(
        xlabel="Training episode",
        ylabel="Mean steps"
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Task")

    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_training_return(stats, destination):
    fig, ax = plt.subplots(figsize=(9, 5))

    for task, group in stats.groupby("task", sort=False):
        agg = (
            group.groupby("block")
            .agg(
                x=("episode_end", "max"),
                mean=("mean_return", "mean"),
                sd=("mean_return", "std"),
            )
            .reset_index()
        )

        x = agg["x"].to_numpy()
        y = agg["mean"].to_numpy()
        sd = agg["sd"].fillna(0).to_numpy()

        (line,) = ax.plot(x, y, label=task, linewidth=2)
        ax.fill_between(
            x, y - sd, y + sd,
            alpha=0.13,
            color=line.get_color()
        )

    ax.set(
        xlabel="Training episode",
        ylabel="Mean training return"
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Task")

    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sw", type=Path, default=Path("results/baseline_SW"))
    parser.add_argument("--ld", type=Path, default=Path("results/baseline_LD"))
    parser.add_argument("--hp", type=Path, default=Path("results/baseline_HP"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    parser.add_argument("--block", type=int, default=25)
    args = parser.parse_args()
    if args.block < 1:
        parser.error("--block must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    folders = {"S-W": args.sw, "L-D": args.ld, "H-P": args.hp}
    all_stats, evaluation_rows, summary_rows = [], [], []
    for task, folder in folders.items():
        training = read_runs(folder, "training")
        evaluation = read_runs(folder, "evaluation")
        stats = block_stats(training, args.block)
        stats["task"] = task
        all_stats.append(stats)
        for seed, subset in evaluation.groupby("seed"):
            successful = subset[subset["success"] == 1]
            evaluation_rows.append({
                "task": task, "seed": seed,
                "success_rate": subset["success"].mean(),
                "mean_return": subset["return"].mean(),
                "mean_steps_success": successful["steps"].mean() if len(successful) else np.nan,
                "optimal_steps": TASKS[task],
            })
        by_block = stats.groupby("block")["success_rate"].mean().sort_index()
        first_sustained_95 = None
        for i in range(len(by_block) - 2):
            if (by_block.iloc[i:i + 3] >= 0.95).all():
                first_sustained_95 = int(stats.loc[stats["block"] == by_block.index[i], "episode_end"].max())
                break
        first = stats[stats["block"] == 0]["success_rate"].mean()
        final = stats[stats["block"] == stats["block"].max()]
        summary_rows.append({
            "task": task, "first_block_success": first,
            "final_block_success": final["success_rate"].mean(),
            "final_block_steps_success": final["steps_on_success"].mean(),
            "first_sustained_95_block_end": first_sustained_95,
            "optimal_steps": TASKS[task]
        })

    stats = pd.concat(all_stats, ignore_index=True)
    evaluations = pd.DataFrame(evaluation_rows)
    summary = pd.DataFrame(summary_rows)
    stats.to_csv(args.output / "training_blocks_by_seed.csv", index=False)
    evaluations.to_csv(args.output / "evaluation_by_seed.csv", index=False)
    summary.to_csv(args.output / "convergence_summary.csv", index=False)

    plot_training_success(stats, args.output / "training_success.png")
    plot_training_steps(stats, args.output / "training_steps.png")
    plot_training_return(stats, args.output / "training_return.png")

    print("\nConvergence (>=95% in three successive blocks):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nFinal evaluation by seed:")
    print(evaluations.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved CSV summaries and three PNG plots to {args.output}")


if __name__ == "__main__":
    main()
