"""Run multi-seed MAPPO experiments and paired statistical analysis.

This script trains and evaluates one model per seed, then compares the paired
mean queue results from the same generated scenario and evaluation horizon.
It writes raw per-seed observations and a 95% confidence interval for the
baseline-minus-MAPPO queue difference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated integer seeds")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("results/marl/statistical_analysis"))
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def paired_analysis(results: pd.DataFrame) -> dict:
    differences = results["fixed_mean_queue"] - results["mappo_mean_queue"]
    count = len(differences)
    mean_difference = float(differences.mean())
    if count > 1:
        standard_error = float(stats.sem(differences))
        critical_value = float(stats.t.ppf(0.975, count - 1))
        confidence_interval = [
            mean_difference - critical_value * standard_error,
            mean_difference + critical_value * standard_error,
        ]
        test = stats.ttest_rel(results["fixed_mean_queue"], results["mappo_mean_queue"])
        t_statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        confidence_interval = [None, None]
        t_statistic = None
        p_value = None

    return {
        "seeds": results["seed"].astype(int).tolist(),
        "sample_size": count,
        "metric": "mean network queue (vehicles)",
        "paired_difference": "fixed-time mean queue - MAPPO mean queue",
        "mean_difference": round(mean_difference, 6),
        "confidence_level": 0.95,
        "confidence_interval_95": [
            round(value, 6) if value is not None else None for value in confidence_interval
        ],
        "paired_t_statistic": round(t_statistic, 6) if t_statistic is not None else None,
        "p_value": round(p_value, 8) if p_value is not None else None,
        "significant_at_0_05": bool(p_value < 0.05) if p_value is not None else None,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for seed in seeds:
        model_dir = ROOT / "models" / "checkpoints" / "statistical_analysis" / f"seed_{seed}"
        results_dir = args.output_dir / "runs" / f"seed_{seed}"
        plots_dir = args.output_dir / "plots" / f"seed_{seed}"
        train_command = [
            sys.executable, "src/train_marl.py", "--seed", str(seed),
            "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
            "--model-dir", str(model_dir), "--results-dir", str(results_dir),
        ]
        evaluate_command = [
            sys.executable, "src/evaluate_marl.py", "--seed", str(seed),
            "--eval-steps", str(args.eval_steps),
            "--model-path", str(model_dir / "mappo_traffic_final.pt"),
            "--results-dir", str(results_dir), "--plots-dir", str(plots_dir),
        ]
        run(train_command)
        run(evaluate_command)

        summary = pd.read_csv(results_dir / "benchmark_comparison_summary.csv")
        fixed = float(summary.loc[summary["Controller"] == "Fixed-Time Baseline", "Mean_Queue_Veh"].iloc[0])
        mappo = float(summary.loc[summary["Controller"] == "MAPPO + Dynamic Routing", "Mean_Queue_Veh"].iloc[0])
        records.append({"seed": seed, "fixed_mean_queue": fixed, "mappo_mean_queue": mappo})

    per_seed = pd.DataFrame(records)
    per_seed.to_csv(args.output_dir / "per_seed_results.csv", index=False)
    analysis = paired_analysis(per_seed)
    (args.output_dir / "paired_t_test.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("\nStatistical analysis")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
