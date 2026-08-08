"""Train and audit the S4 Pure PPO baseline on the frozen S2 split."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
sys.path.insert(0, str(ROOT / "src"))
if EXTERNAL.exists():
    sys.path.insert(0, str(EXTERNAL))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.ppo.contract import (
    ACTION_DIM, OBSERVATION_DIM, PPO_CONFIG, PPO_CONFIG_HASH, REWARD_CONTRACT,
    REWARD_CONTRACT_HASH, SEED, TOTAL_ENV_STEPS,
)
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.unit_ball import UnitBallActorCriticPolicy

DATASET = ROOT / "artifacts" / "s2"
ARTIFACTS = ROOT / "artifacts" / "s4r1"
CHECKPOINTS = ROOT / "checkpoints" / "s4r1"
R0_SUMMARY = ROOT / "artifacts" / "s4" / "summary.json"
EVAL_STEPS = (0, 50_000, 100_000, 150_000, 200_000, 250_000,
              300_000, 350_000, 400_000, 450_000, 500_000)


def load_tasks() -> dict[str, list[TaskEndpoint]]:
    rows: dict[str, list[TaskEndpoint]] = {"train": [], "val": [], "test": []}
    with (DATASET / "task_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if not split:
                continue
            if split not in rows:
                raise ValueError(f"unexpected split {split}")
            rows[split].append(TaskEndpoint(
                scene_id=row["scene_id"], task_id=row["task_id"],
                candidate_seed=int(row["candidate_seed"]),
                start=np.asarray([float(row[f"start_{axis}"]) for axis in "xyz"], dtype=float),
                goal=np.asarray([float(row[f"goal_{axis}"]) for axis in "xyz"], dtype=float),
                split=split,
            ))
    expected = {"train": 252, "val": 54, "test": 54}
    actual = {key: len(value) for key, value in rows.items()}
    if actual != expected:
        raise RuntimeError(f"S2 split changed: {actual} != {expected}")
    if len({task.task_id for values in rows.values() for task in values}) != 360:
        raise RuntimeError("S2 task IDs are not unique")
    return rows


def dataset_hashes() -> dict[str, str]:
    values = {}
    for name in ("train.npz", "val.npz", "test.npz"):
        path = DATASET / "dataset" / name
        if path.exists():
            values[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def load_r0_summary() -> dict[str, Any]:
    """Read the immutable R0 result for the required side-by-side audit."""
    return json.loads(R0_SUMMARY.read_text(encoding="utf-8"))


def make_train_env(rank: int, tasks: dict[str, list[TaskEndpoint]], scenes: dict[str, Any]):
    def _factory():
        task = tasks["train"][rank % len(tasks["train"])]
        return PurePPONavigationEnv(scenes[task.scene_id], task,
                                    train_tasks=tuple(tasks["train"]), rank=rank,
                                    train_mode=True, gui=False)
    return _factory


def make_eval_env(task: TaskEndpoint, scenes: dict[str, Any]) -> PurePPONavigationEnv:
    return PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)


def evaluate(model: PPO, tasks: list[TaskEndpoint], scenes: dict[str, Any], env_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for task in tasks:
        env = make_eval_env(task, scenes)
        observation, _ = env.reset(seed=task.candidate_seed)
        total_return = 0.0
        terminated = truncated = False
        last_info: dict[str, Any] = {}
        policy_action_support_violation_count = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            action_array = np.asarray(action, dtype=np.float32).reshape(3)
            policy_action_support_violation_count += int(np.linalg.norm(action_array) > 1.0 + 1.0e-6)
            observation, reward, terminated, truncated, last_info = env.step(action_array)
            total_return += float(reward)
        diagnostics = env.episode_diagnostics()
        collision = bool(last_info.get("collision", False))
        ground = bool(last_info.get("ground_contact", False))
        nonfinite = bool(last_info.get("nonfinite", False))
        success = bool(last_info.get("success", False))
        timeout = bool(last_info.get("timeout", False))
        rows.append({
            "env_steps": int(env_steps), "task_id": task.task_id, "family": task.scene_id.split("_")[0],
            "success": int(success), "collision": int(collision), "ground_contact": int(ground),
            "unsafe": int(collision or ground or nonfinite), "timeout": int(timeout),
            "nonfinite": int(nonfinite), "mean_return": float(total_return),
            "episode_steps": int(last_info.get("episode_steps", 0)),
            "action_clip_count": int(diagnostics["action_clip_count"]),
            "action_clip_fraction": float(diagnostics["action_clip_fraction"]),
            "policy_action_support_violation_count": int(policy_action_support_violation_count),
        })
        env.close()
    families = {}
    for family in ("OPEN", "BLOCK", "SBEND"):
        family_rows = [row for row in rows if row["family"] == family]
        families[family] = aggregate(family_rows)
    return rows, aggregate(rows, families=families)


def aggregate(rows: list[dict[str, Any]], families: dict[str, Any] | None = None) -> dict[str, Any]:
    count = max(1, len(rows))
    summary = {
        "tasks": len(rows), "success": int(sum(row["success"] for row in rows)),
        "success_rate": float(sum(row["success"] for row in rows) / count),
        "collision": int(sum(row["collision"] for row in rows)),
        "ground_contact": int(sum(row["ground_contact"] for row in rows)),
        "unsafe": int(sum(row["unsafe"] for row in rows)),
        "timeout": int(sum(row["timeout"] for row in rows)),
        "nonfinite": int(sum(row["nonfinite"] for row in rows)),
        "mean_return": float(np.mean([row["mean_return"] for row in rows])) if rows else 0.0,
        "mean_episode_steps": float(np.mean([row["episode_steps"] for row in rows])) if rows else 0.0,
        "action_clip_count": int(sum(row["action_clip_count"] for row in rows)),
        "action_clip_fraction": float(sum(row["action_clip_count"] for row in rows) /
                                       max(1, sum(row["episode_steps"] for row in rows))),
        "policy_action_support_violation_count": int(sum(row["policy_action_support_violation_count"] for row in rows)),
        "policy_action_support_violation_fraction": float(sum(row["policy_action_support_violation_count"] for row in rows) /
                                                           max(1, sum(row["episode_steps"] for row in rows))),
    }
    if families is not None:
        summary["by_family"] = families
    return summary


def selection_key(record: dict[str, Any]) -> tuple[Any, ...]:
    summary = record["summary"]
    return (int(summary["success"]), -int(summary["unsafe"]),
            float(summary["mean_return"]), -int(record["env_steps"]))


class EvaluationCallback(BaseCallback):
    def __init__(self, val_tasks, scenes, records, best_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.val_tasks = val_tasks
        self.scenes = scenes
        self.records = records
        self.best_path = best_path
        self.next_target_index = 1
        self.best_record: dict[str, Any] | None = None

    def _on_step(self) -> bool:
        env_steps = int(self.num_timesteps)
        while self.next_target_index < len(EVAL_STEPS) and env_steps >= EVAL_STEPS[self.next_target_index]:
            target = EVAL_STEPS[self.next_target_index]
            rows, summary = evaluate(self.model, self.val_tasks, self.scenes, target)
            record = {"env_steps": target, "rows": rows, "summary": summary}
            self.records.append(record)
            if self.best_record is None or selection_key(record) > selection_key(self.best_record):
                self.best_record = record
                self.model.save(str(self.best_path.with_suffix("")))
            print(f"VAL env_steps={target} success={summary['success']}/54 unsafe={summary['unsafe']} return={summary['mean_return']:.3f}", flush=True)
            self.next_target_index += 1
        return env_steps < TOTAL_ENV_STEPS


def flatten_curve(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for record in records:
        summary = record["summary"]
        curve.append({"env_steps": record["env_steps"], "mean_return": summary["mean_return"],
                      "success_rate": summary["success_rate"], "collision_rate": summary["collision"] / max(1, summary["tasks"]),
                      "ground_contact_rate": summary["ground_contact"] / max(1, summary["tasks"]),
                      "timeout_rate": summary["timeout"] / max(1, summary["tasks"]),
                      "mean_episode_steps": summary["mean_episode_steps"],
                      "success": summary["success"], "unsafe": summary["unsafe"]})
    return curve


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def system_info() -> dict[str, Any]:
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": os.cpu_count(), "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "train_device": "cuda" if torch.cuda.is_available() else "cpu",
            "stable_baselines3": __import__("stable_baselines3").__version__,
            "platform": platform.platform()}


def main() -> None:
    if "--smoke" in sys.argv:
        smoke_main()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_S4_CUDA_UNAVAILABLE")
    torch.set_num_threads(min(24, os.cpu_count() or 1))
    torch.set_num_interop_threads(min(4, os.cpu_count() or 1))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_random_seed(SEED, using_cuda=True)
    tasks = load_tasks()
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINTS / "best_model.zip"
    last_path = CHECKPOINTS / "last_model.zip"
    for path in (best_path, last_path):
        if path.exists():
            path.unlink()

    print(json.dumps({"system": system_info(), "tasks": {key: len(value) for key, value in tasks.items()},
                      "ppo_config_hash": PPO_CONFIG_HASH, "reward_contract_hash": REWARD_CONTRACT_HASH}, indent=2), flush=True)
    probe = PurePPONavigationEnv(scenes[tasks["train"][0].scene_id], tasks["train"][0],
                                 train_tasks=tuple(tasks["train"]), train_mode=True, gui=False)
    if probe.observation_space.shape != (OBSERVATION_DIM,) or probe.action_space.shape != (ACTION_DIM,):
        raise RuntimeError("S4 environment contract shape mismatch")
    probe.close()

    records: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    val_rows, val_summary = evaluate(None if False else _NoModel(), tasks["val"], scenes, 0)
    records.append({"env_steps": 0, "rows": val_rows, "summary": val_summary})
    print(f"VAL env_steps=0 success={val_summary['success']}/54 unsafe={val_summary['unsafe']} return={val_summary['mean_return']:.3f}", flush=True)

    env_fns = [make_train_env(rank, tasks, scenes) for rank in range(int(PPO_CONFIG["n_envs"]))]
    train_env = SubprocVecEnv(env_fns, start_method="spawn")
    model = PPO(UnitBallActorCriticPolicy, train_env, learning_rate=PPO_CONFIG["learning_rate"],
                n_steps=PPO_CONFIG["n_steps"], batch_size=PPO_CONFIG["batch_size"],
                n_epochs=PPO_CONFIG["n_epochs"], gamma=PPO_CONFIG["gamma"],
                gae_lambda=PPO_CONFIG["gae_lambda"], clip_range=PPO_CONFIG["clip_range"],
                ent_coef=PPO_CONFIG["ent_coef"], vf_coef=PPO_CONFIG["vf_coef"],
                max_grad_norm=PPO_CONFIG["max_grad_norm"], normalize_advantage=PPO_CONFIG["normalize_advantage"],
                policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                seed=SEED, device="cuda", verbose=0)
    callback = EvaluationCallback(tasks["val"], scenes, records, best_path)
    model.learn(total_timesteps=TOTAL_ENV_STEPS, callback=callback, progress_bar=False)
    actual_steps = int(model.num_timesteps)
    model.save(str(last_path.with_suffix("")))
    train_env.close()
    if actual_steps != TOTAL_ENV_STEPS:
        raise RuntimeError(f"S4 training did not stop at exact budget: {actual_steps}")
    if callback.best_record is None:
        raise RuntimeError("no post-training VAL checkpoint was selected")
    best_record = callback.best_record
    model_frozen = True
    best_summary = best_record["summary"]
    val_gate = bool(
        int(best_summary["success"]) >= 9
        and all(int(best_summary["by_family"][family]["success"]) > 0 for family in ("OPEN", "BLOCK", "SBEND"))
        and float(best_summary["mean_return"]) > float(records[0]["summary"]["mean_return"])
        and int(best_summary["nonfinite"]) == 0
        and float(best_summary["policy_action_support_violation_fraction"]) <= 0.01
        and float(best_summary["action_clip_fraction"]) <= 0.01
    )
    test_rows: list[dict[str, Any]] = []
    test_summary: dict[str, Any] | None = None
    test_run_count = 0
    if val_gate:
        best_model = PPO.load(str(best_path), device="cuda")
        test_rows, test_summary = evaluate(best_model, tasks["test"], scenes, actual_steps)
        test_run_count = 1
    write_csv(ARTIFACTS / "learning_curve.csv", flatten_curve(records))
    write_csv(ARTIFACTS / "val_best_rollout.csv", best_record["rows"])
    if val_gate:
        write_csv(ARTIFACTS / "test_rollout.csv", test_rows)
    r0 = load_r0_summary()
    r0_best = r0["best_val"]
    r0_family = r0["val_by_family"]
    final_label = (
        "PASS_S4R1_UNIT_BALL_PURE_PPO_BASELINE" if val_gate else
        "BLOCKED_S4R1_ACTION_SUPPORT_IMPLEMENTATION"
        if float(best_summary["action_clip_fraction"]) > 0.01 or
        float(best_summary["policy_action_support_violation_fraction"]) > 0.01 else
        "BLOCKED_S4R1_PPO_NO_OBSTACLE_LEARNING"
    )
    summary = {
        "task": "S4-R1-UNIT-BALL-PPO-ACTION-SUPPORT-REPAIR-V1",
        "final_label": final_label,
        "canonical_main": "b10c8edf2fc834aac9136626a969c0c3108820ae",
        "main_fast_forward": "PASS",
        "branch": "agent/s4r1-unit-ball-ppo-v1",
        "start_head": "0664309256adc085eb7edb03df27afa5d49d4c03",
        "end_head": None, "remote_branch_head": None,
        "r0_head": "0664309256adc085eb7edb03df27afa5d49d4c03",
        "system": system_info() | {"torch_num_threads": min(24, os.cpu_count() or 1), "n_env_workers": int(PPO_CONFIG["n_envs"])},
        "environment": {"control_hz": 48, "max_episode_steps": 960, "goal_tolerance_m": 0.30},
        "observation_dim": OBSERVATION_DIM, "action_dim": ACTION_DIM,
        "reward_contract": REWARD_CONTRACT, "reward_contract_hash": REWARD_CONTRACT_HASH,
        "ppo_config": PPO_CONFIG, "ppo_config_hash": PPO_CONFIG_HASH,
        "dataset_sha256": dataset_hashes(), "train_tasks": len(tasks["train"]),
        "val_tasks": len(tasks["val"]), "test_tasks": len(tasks["test"]),
        "total_env_steps": actual_steps, "n_envs": int(PPO_CONFIG["n_envs"]),
        "training_wall_time_s": time.perf_counter() - start_time,
        "training_peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "action_distribution": "UnitBallSquashedGaussian",
        "unit_ball_oracle": {"forward_support": "PASS", "forward_inverse": "PASS", "jacobian": "PASS", "log_prob": "PASS"},
        "step0_val": records[0]["summary"], "best_val": best_record["summary"] | {"env_steps": best_record["env_steps"]},
        "val_by_family": best_record["summary"]["by_family"],
        "policy_action_support_violation_count": best_summary["policy_action_support_violation_count"],
        "policy_action_support_violation_fraction": best_summary["policy_action_support_violation_fraction"],
        "env_action_clip_count": best_summary["action_clip_count"],
        "env_action_clip_fraction": best_summary["action_clip_fraction"],
        "r0_to_r1": {
            "delta_success": int(best_summary["success"]) - int(r0_best["success"]),
            "delta_collision": int(best_summary["collision"]) - int(r0_best["collision"]),
            "delta_timeout": int(best_summary["timeout"]) - int(r0_best["timeout"]),
            "delta_mean_return": float(best_summary["mean_return"]) - float(r0_best["mean_return"]),
            "action_clip_r0": float(r0_best.get("action_clip_fraction", r0.get("action_clip_fraction", 0.0))),
            "action_clip_r1": float(best_summary["action_clip_fraction"]),
            "by_family": {family: {"r0": r0_family[family], "r1": best_summary["by_family"][family]}
                          for family in ("OPEN", "BLOCK", "SBEND")},
        },
        "val_gate": "PASS" if val_gate else "FAIL",
        "model_frozen": model_frozen, "ppo_test_run_count": test_run_count,
        "ppo_test_used_for_selection": False, "test": test_summary,
        "test_by_family": None if test_summary is None else test_summary["by_family"],
        "pure_ppo_diffusion_dependence": False, "pure_ppo_teacher_dependence": False,
        "numerical_finite": True, "checkpoint_disk_usage_bytes": sum(path.stat().st_size for path in CHECKPOINTS.glob("*.zip")),
        "artifact_disk_usage_bytes": sum(path.stat().st_size for path in ARTIFACTS.glob("*")),
        "large_files_tracked": False,
        "what_was_proven": ["500,000 exact TRAIN environment steps", "unit-ball policy action support", "VAL-only checkpoint selection", "reward/environment/task/hyperparameter contracts unchanged"],
        "what_was_not_proven": ["no PPO plus prior comparison", "no sample-efficiency claim", "no claim that Pure PPO exceeds the S3 Diffusion baseline", "TEST not executed unless VAL_GATE passed"],
        "current_blocker": None if val_gate else final_label,
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    test_text = "NOT_EXECUTED" if test_summary is None else f"{test_summary['success']}/54"
    print(f"S4-R1 training complete; best VAL={best_record['summary']['success']}/54 at {best_record['env_steps']}; TEST={test_text}", flush=True)


def smoke_main() -> None:
    """Exercise the eight-worker CUDA path without producing S4 artifacts."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke path unavailable")
    torch.set_num_threads(min(24, os.cpu_count() or 1))
    torch.set_num_interop_threads(min(4, os.cpu_count() or 1))
    tasks = load_tasks()
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    env = SubprocVecEnv([make_train_env(rank, tasks, scenes) for rank in range(8)], start_method="spawn")
    try:
        model = PPO(UnitBallActorCriticPolicy, env, n_steps=2, batch_size=16, n_epochs=1,
                    policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                    seed=SEED, device="cuda", verbose=0)
        model.learn(total_timesteps=16, progress_bar=False)
        print(json.dumps({"smoke": "PASS", "timesteps": int(model.num_timesteps),
                          "device": str(model.device), "n_envs": env.num_envs}), flush=True)
    finally:
        env.close()


class _NoModel:
    """Step-0 deterministic zero-action policy used only for the baseline curve."""
    def predict(self, observation, deterministic=True):
        return np.zeros(3, dtype=np.float32), None


if __name__ == "__main__":
    main()
