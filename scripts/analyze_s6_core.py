"""Preregistered paired analysis for the six formal S6 VAL runs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (20260812, 20260813, 20260814)
METHODS = ("pure_ppo", "diffusion_ppo")
STEPS = np.arange(0, 500_001, 50_000, dtype=float)
ARTIFACTS = ROOT / "artifacts" / "s6"
PRIOR_RATE = 37.0 / 54.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_runs() -> dict[tuple[str, int], list[dict[str, str]]]:
    runs = {}
    for method in METHODS:
        for seed in SEEDS:
            run_dir = ARTIFACTS / "runs" / f"{method}_seed_{seed}"
            train = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
            evaluation = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
            if train["actual_env_steps"] != 500_000 or evaluation["checkpoint_count"] != 11:
                raise RuntimeError(f"incomplete run {method} {seed}")
            if train["test_accessed"] or evaluation["test_accessed"]:
                raise RuntimeError(f"TEST leakage in {method} {seed}")
            rows = read_csv(run_dir / "curve.csv")
            if [int(row["env_steps"]) for row in rows] != STEPS.astype(int).tolist():
                raise RuntimeError(f"schedule mismatch in {method} {seed}")
            runs[(method, seed)] = rows
    return runs


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def auc(rows: list[dict[str, str]], key: str, end: int = 500_000) -> float:
    mask = STEPS <= end
    return float(np.trapz(values(rows, key)[mask], STEPS[mask]) / float(end))


def first_threshold(rows: list[dict[str, str]]) -> int | str:
    for row in rows:
        if int(float(row["success"])) >= 27:
            return int(row["env_steps"])
    return "NOT_REACHED_WITHIN_500K"


def bootstrap_ci(deltas: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260810)
    samples = rng.choice(deltas, size=(100_000, len(deltas)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def per_run_metrics(runs: dict[tuple[str, int], list[dict[str, str]]]) -> list[dict[str, Any]]:
    result = []
    for method in METHODS:
        for seed in SEEDS:
            rows = runs[(method, seed)]
            successes = values(rows, "success")
            record: dict[str, Any] = {
                "method": method, "seed": seed,
                "success_auc_0_500k": auc(rows, "success_rate"),
                "success_auc_0_200k": auc(rows, "success_rate", 200_000),
                "collision_auc_0_500k": auc(rows, "collision_rate"),
                "unsafe_auc_0_500k": auc(rows, "unsafe_rate"),
                "return_auc_0_500k": auc(rows, "mean_return"),
                "obstacle_success_auc_0_500k": auc(rows, "obstacle_success_rate"),
                "steps_to_50_percent_success": first_threshold(rows),
                "best_success": int(successes.max()),
                "best_step": int(rows[int(np.argmax(successes))]["env_steps"]),
                "best_post_training_success": int(successes[1:].max()),
                "best_post_training_step": int(rows[1 + int(np.argmax(successes[1:]))]["env_steps"]),
                "best_post_warmup_success": int(successes[2:].max()),
                "final_success": int(successes[-1]),
            }
            for step in (0, 50_000, 100_000, 200_000, 500_000):
                index = int(step // 50_000)
                record[f"success_at_{step}"] = int(successes[index])
                if step:
                    record[f"collision_at_{step}"] = int(float(rows[index]["collision"]))
            if method == "diffusion_ppo":
                record["max_improvement_over_prior_count"] = record["best_post_training_success"] - 37
                record["auc_relative_to_constant_prior"] = record["success_auc_0_500k"] - PRIOR_RATE
            result.append(record)
    return result


def aggregate_curves(runs: dict[tuple[str, int], list[dict[str, str]]]) -> list[dict[str, Any]]:
    output = []
    keys = ("success_rate", "collision_rate", "unsafe_rate", "mean_return", "obstacle_success_rate")
    for method in METHODS:
        for index, step in enumerate(STEPS.astype(int)):
            record: dict[str, Any] = {"method": method, "env_steps": step}
            for key in keys:
                sample = np.asarray([float(runs[(method, seed)][index][key]) for seed in SEEDS])
                record[f"{key}_mean"] = float(sample.mean())
                record[f"{key}_std"] = float(sample.std(ddof=1))
            output.append(record)
    return output


def make_plots(aggregates: list[dict[str, Any]]) -> None:
    specs = (
        ("success_rate", "Success rate", "success_vs_env_steps.png"),
        ("collision_rate", "Collision rate", "collision_vs_env_steps.png"),
        ("mean_return", "Mean return", "return_vs_env_steps.png"),
        ("obstacle_success_rate", "BLOCK + SBEND success rate", "obstacle_success_vs_env_steps.png"),
    )
    colors = {"pure_ppo": "#3B6FB6", "diffusion_ppo": "#D65F5F"}
    labels = {"pure_ppo": "Pure PPO", "diffusion_ppo": "Frozen Diffusion + PPO residual"}
    for key, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        for method in METHODS:
            rows = [row for row in aggregates if row["method"] == method]
            x = np.asarray([row["env_steps"] for row in rows])
            mean = np.asarray([row[f"{key}_mean"] for row in rows])
            std = np.asarray([row[f"{key}_std"] for row in rows])
            ax.plot(x, mean, marker="o", linewidth=2, color=colors[method], label=labels[method])
            ax.fill_between(x, mean - std, mean + std, color=colors[method], alpha=0.18)
        ax.set_xlabel("Online environment interactions")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(frameon=False)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        fig.savefig(ARTIFACTS / filename, dpi=240)
        fig.savefig(ARTIFACTS / filename.replace(".png", ".pdf"))
        plt.close(fig)


def main() -> None:
    runs = load_runs()
    metrics = per_run_metrics(runs)
    aggregates = aggregate_curves(runs)
    write_csv(ARTIFACTS / "per_run_metrics.csv", metrics)
    write_csv(ARTIFACTS / "aggregate_curves.csv", aggregates)
    paired = []
    for seed in SEEDS:
        pure = next(row for row in metrics if row["method"] == "pure_ppo" and row["seed"] == seed)
        diffusion = next(row for row in metrics if row["method"] == "diffusion_ppo" and row["seed"] == seed)
        paired.append({
            "seed": seed,
            "delta_success_auc_0_500k": diffusion["success_auc_0_500k"] - pure["success_auc_0_500k"],
            "delta_success_auc_0_200k": diffusion["success_auc_0_200k"] - pure["success_auc_0_200k"],
            "delta_collision_auc_0_500k": diffusion["collision_auc_0_500k"] - pure["collision_auc_0_500k"],
            "delta_unsafe_auc_0_500k": diffusion["unsafe_auc_0_500k"] - pure["unsafe_auc_0_500k"],
            "delta_return_auc_0_500k": diffusion["return_auc_0_500k"] - pure["return_auc_0_500k"],
            "delta_obstacle_success_auc_0_500k": diffusion["obstacle_success_auc_0_500k"] - pure["obstacle_success_auc_0_500k"],
        })
    write_csv(ARTIFACTS / "paired_metrics.csv", paired)
    deltas = np.asarray([row["delta_success_auc_0_500k"] for row in paired])
    h1 = bool(np.all(deltas > 0.0) and deltas.mean() >= 0.10)
    residual = [row for row in metrics if row["method"] == "diffusion_ppo"]
    residual_positive_auc = sum(row["auc_relative_to_constant_prior"] > 0 for row in residual)
    residual_better_best = sum(row["best_post_warmup_success"] > 37 for row in residual)
    residual_lower_final = sum(row["final_success"] < 37 for row in residual)
    residual_negative_auc = sum(row["auc_relative_to_constant_prior"] < 0 for row in residual)
    if residual_better_best >= 2 and np.mean([row["auc_relative_to_constant_prior"] for row in residual]) > 0:
        residual_label = "RESIDUAL_LEARNING_IMPROVES_PRIOR"
    elif residual_lower_final >= 2 and residual_negative_auc >= 2:
        residual_label = "RESIDUAL_LEARNING_DEGRADES_PRIOR"
    else:
        residual_label = "RESIDUAL_LEARNING_NEUTRAL"
    safety = bool(
        np.mean([row["delta_collision_auc_0_500k"] for row in paired]) < 0
        and np.mean([row["delta_unsafe_auc_0_500k"] for row in paired]) < 0
        and sum(row["delta_collision_auc_0_500k"] < 0 for row in paired) >= 2
        and sum(row["delta_unsafe_auc_0_500k"] < 0 for row in paired) >= 2
    )
    key_steps = (50_000, 100_000, 200_000)
    stability_checks = []
    for step in key_steps:
        p = next(row for row in aggregates if row["method"] == "pure_ppo" and row["env_steps"] == step)
        d = next(row for row in aggregates if row["method"] == "diffusion_ppo" and row["env_steps"] == step)
        stability_checks.append(d["success_rate_mean"] > p["success_rate_mean"]
                                and d["success_rate_std"] <= p["success_rate_std"] + 0.05)
    stability = bool(all(stability_checks))
    if not h1:
        level = "NO_GO_CORE_HYPOTHESIS"
        label = "NO_GO_S6_ONLINE_SAMPLE_EFFICIENCY_NOT_SUPPORTED"
    elif residual_label == "RESIDUAL_LEARNING_IMPROVES_PRIOR":
        level = "FULL_GO"
        label = "PASS_S6_THREE_SEED_FULL_CORE_HYPOTHESIS_SUPPORTED"
    else:
        level = "CONDITIONAL_GO_PRIOR_EFFECT_ONLY"
        label = "PASS_S6_ONLINE_SAMPLE_EFFICIENCY_SUPPORTED_RESIDUAL_NOT_SUPPORTED"
    make_plots(aggregates)
    summary = {
        "task": "S6-R0-THREE-SEED-CORE-SAMPLE-EFFICIENCY-VALIDATION-V2",
        "final_label": label, "decision_level": level,
        "h1_online_sample_efficiency": "SUPPORTED" if h1 else "NOT_SUPPORTED",
        "h1_rule": "3/3 paired deltas > 0 and mean absolute normalized SUCCESS_AUC_0_500K improvement >= 0.10",
        "paired_success_auc_deltas": deltas.tolist(), "paired_delta_mean": float(deltas.mean()),
        "paired_delta_std": float(deltas.std(ddof=1)), "paired_delta_bootstrap_95_ci": bootstrap_ci(deltas),
        "residual_learning": residual_label,
        "residual_rule_frozen_before_analysis": {
            "improves": "at least 2/3 best post-warmup success > 37 and mean AUC vs constant prior > 0",
            "degrades": "at least 2/3 final success < 37 and at least 2/3 AUC vs constant prior < 0",
            "neutral": "otherwise",
        },
        "residual_seeds_positive_auc_vs_prior": residual_positive_auc,
        "residual_seeds_best_post_warmup_above_prior": residual_better_best,
        "h3_safety": "SUPPORTED" if safety else "NOT_SUPPORTED",
        "h_stability": "SUPPORTED" if stability else "NOT_SUPPORTED",
        "test_accessed": False, "test_status": "LOCKED_PENDING_FROZEN_VAL_DECISION",
        "historical_seed1_excluded": True, "formal_runs": 6,
        "formal_progress": "70%", "unique_next_task": "NONE - WAIT_FOR_CONTROLLER_REVIEW",
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
