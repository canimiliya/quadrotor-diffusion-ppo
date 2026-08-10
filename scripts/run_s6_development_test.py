"""One-shot development TEST after the complete S6 VAL decision is frozen."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
if EXTERNAL.exists():
    sys.path.insert(0, str(EXTERNAL))

import torch
from stable_baselines3 import PPO

from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.ppo.residual import FrozenDiffusionPrior
from scripts.run_s4r2_ppo import load_tasks
from scripts.run_s6_core import (
    METHODS, SEEDS, S3R2_CHECKPOINT, evaluate_pure, evaluate_residual,
    preflight_fast, sha256, write_csv, write_json,
)


def main() -> None:
    summary_path = ROOT / "artifacts" / "s6" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["h1_online_sample_efficiency"] != "SUPPORTED" or summary["test_accessed"]:
        raise RuntimeError("TEST gate is not open or TEST was already recorded")
    tasks = load_tasks(); test_tasks = tasks["test"]
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    outputs = []
    for method in METHODS:
        for seed in SEEDS:
            run_name = f"{method}_seed_{seed}"
            run_dir = ROOT / "artifacts" / "s6" / "runs" / run_name
            checkpoint_dir = ROOT / "checkpoints" / "s6" / run_name
            evaluation = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
            step = int(evaluation["best_step"])
            checkpoint = checkpoint_dir / f"checkpoint_{step:06d}.zip"
            expected = next(item["sha256"] for item in json.loads(
                (run_dir / "training_manifest.json").read_text(encoding="utf-8")
            )["checkpoint_schedule"] if int(item["env_steps"]) == step)
            if sha256(checkpoint) != expected:
                raise RuntimeError(f"selected checkpoint identity failed: {run_name}")
            model = PPO.load(str(checkpoint), device="cuda")
            started = time.perf_counter()
            if method == "pure_ppo":
                rows, result = evaluate_pure(model, test_tasks, scenes, step)
                fast = None
            else:
                prior = FrozenDiffusionPrior(S3R2_CHECKPOINT, "cuda")
                fast = preflight_fast(prior)
                rows, result, _ = evaluate_residual(model, prior, test_tasks, scenes, step)
            wall = time.perf_counter() - started
            write_csv(run_dir / "test_rollouts.csv", rows)
            manifest = {
                "method": method, "seed": seed, "selected_step_frozen_before_test": step,
                "selected_checkpoint_sha256": expected, "result": result,
                "wall_time_s": wall, "fast_preflight": fast,
                "test_used_for_selection": False, "test_used_for_method_change": False,
                "test_run_count": 1,
            }
            write_json(run_dir / "test_manifest.json", manifest)
            outputs.append(manifest)
            print(f"TEST method={method} seed={seed} selected={step} "
                  f"success={result['success']}/54 unsafe={result['unsafe']} wall={wall:.2f}s", flush=True)
    write_json(ROOT / "artifacts" / "s6" / "development_test_summary.json", {
        "gate": "H1_SUPPORTED_AFTER_ALL_VAL_FROZEN", "runs": outputs,
        "used_for_selection": False, "used_for_method_change": False,
    })
    summary["test_accessed"] = True
    summary["test_status"] = "DEVELOPMENT_TEST_COMPLETE_AFTER_FROZEN_VAL"
    summary["test_used_for_selection"] = False
    summary["test_used_for_method_change"] = False
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
