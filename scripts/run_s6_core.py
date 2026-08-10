"""Run one preregistered S6 seed with deferred, VAL-only evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
if EXTERNAL.exists():
    sys.path.insert(0, str(EXTERNAL))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.evaluation.runtime import DiffusionInferenceScheduler, RuntimeMode
from quadrotor_diffusion_ppo.ppo.contract import (
    ACTION_DIM, OBSERVATION_DIM, PPO_CONFIG, PPO_CONFIG_HASH, REWARD_CONTRACT,
    REWARD_CONTRACT_HASH, TOTAL_ENV_STEPS,
)
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.normalization import derive_train_statistics
from quadrotor_diffusion_ppo.ppo.residual import (
    EXPECTED_S3R2_SHA256, RESIDUAL_LOG_STD_INIT, RESIDUAL_SCALE,
    FrozenDiffusionPrior, compose_residual_action, stable_prior_seed,
)
from quadrotor_diffusion_ppo.ppo.runtime import capture_rng_state
from quadrotor_diffusion_ppo.ppo.unit_ball import UnitBallActorCriticPolicy
from scripts.run_s4r2_ppo import aggregate, load_tasks, make_train_env, policy_kwargs, selection_key
from scripts.run_s5_residual_ppo import DiffusionResidualVecEnv, initialize_residual_policy
from scripts.run_s6p0_runtime import ResourceSampler


TASK = "S6-R0-THREE-SEED-CORE-SAMPLE-EFFICIENCY-VALIDATION-V2"
SEEDS = (20260812, 20260813, 20260814)
METHODS = ("pure_ppo", "diffusion_ppo")
EVAL_STEPS = tuple(range(0, 500_001, 50_000))
S3R2_CHECKPOINT = ROOT / "checkpoints" / "s3r2" / "best.pt"
TRAIN_NPZ = ROOT / "artifacts" / "s2" / "dataset" / "train.npz"
VAL_NPZ = ROOT / "artifacts" / "s2" / "dataset" / "val.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rng_equal(left: dict, right: dict) -> bool:
    return (
        left["python"] == right["python"]
        and left["numpy"][:1] == right["numpy"][:1]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))
    )


class ExactCheckpointCallback(BaseCallback):
    """Save the frozen schedule and stop at exactly 500k interactions."""

    def __init__(self, checkpoint_dir: Path):
        super().__init__(verbose=0)
        self.checkpoint_dir = checkpoint_dir
        self.next_index = 0
        self.saved: list[dict[str, Any]] = []

    def _save(self, target: int) -> None:
        before = capture_rng_state()
        path = self.checkpoint_dir / f"checkpoint_{target:06d}"
        self.model.save(str(path))
        after = capture_rng_state()
        if not rng_equal(before, after):
            raise RuntimeError("checkpoint serialization changed training RNG")
        final_path = path.with_suffix(".zip")
        self.saved.append({
            "env_steps": target,
            "actual_model_timesteps": int(self.model.num_timesteps),
            "path": str(final_path),
            "sha256": sha256(final_path),
            "rng_unchanged": True,
        })
        self.next_index += 1
        print(f"CHECKPOINT env_steps={target} sha256={self.saved[-1]['sha256']}", flush=True)

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save(0)

    def _on_step(self) -> bool:
        steps = int(self.num_timesteps)
        while self.next_index < len(EVAL_STEPS) and steps >= EVAL_STEPS[self.next_index]:
            self._save(EVAL_STEPS[self.next_index])
        return steps < TOTAL_ENV_STEPS


def scientific_config(method: str, seed: int) -> dict[str, Any]:
    value = dict(PPO_CONFIG)
    value["seed"] = seed
    value["observation_normalization"] = "fixed_s2_train_standardization"
    value["evaluation"] = "deferred_val_only_0_50k_to_500k"
    value["test_accessed"] = False
    if method == "diffusion_ppo":
        value.update({
            "runtime_mode": "FAST_CUDA_GRAPH_EXACT_BATCH_ONE",
            "diffusion_checkpoint_sha256": EXPECTED_S3R2_SHA256,
            "residual_scale": RESIDUAL_SCALE,
            "residual_log_std_init": RESIDUAL_LOG_STD_INIT,
        })
    else:
        value["diffusion_dependency"] = False
    return value


def hardware() -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "stable_baselines3": __import__("stable_baselines3").__version__,
        "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
    }


def base_row(task: TaskEndpoint, env_steps: int, info: dict[str, Any], total_return: float,
             diagnostics: dict[str, Any]) -> dict[str, Any]:
    collision = bool(info.get("collision", False))
    ground = bool(info.get("ground_contact", False))
    nonfinite = bool(info.get("nonfinite", False))
    return {
        "env_steps": env_steps, "task_id": task.task_id, "family": task.scene_id.split("_")[0],
        "success": int(bool(info.get("success", False))), "collision": int(collision),
        "ground_contact": int(ground), "unsafe": int(collision or ground or nonfinite),
        "timeout": int(bool(info.get("timeout", False))), "nonfinite": int(nonfinite),
        "mean_return": float(total_return), "episode_steps": int(info.get("episode_steps", 0)),
        "action_clip_count": int(diagnostics["action_clip_count"]),
    }


def evaluate_pure(model: PPO, tasks: list[TaskEndpoint], scenes: dict[str, Any],
                  env_steps: int, workers: int = 16) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy_lock = threading.Lock()

    def one(task: TaskEndpoint) -> dict[str, Any]:
        env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)
        total_return = 0.0; last_info: dict[str, Any] = {}; support = 0
        try:
            observation, _ = env.reset(seed=task.candidate_seed)
            terminated = truncated = False
            while not (terminated or truncated):
                with policy_lock:
                    action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(3)
                support += int(np.linalg.norm(action) > 1.0 + 1.0e-6)
                observation, reward, terminated, truncated, last_info = env.step(action)
                total_return += float(reward)
            diagnostics = env.episode_diagnostics()
        finally:
            env.close()
        row = base_row(task, env_steps, last_info, total_return, diagnostics)
        row["policy_action_support_violation_count"] = support
        return row

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="s6-pure-val") as executor:
        rows = list(executor.map(one, tasks))
    families = {name: aggregate([row for row in rows if row["family"] == name])
                for name in ("OPEN", "BLOCK", "SBEND")}
    return rows, aggregate(rows, families=families)


def evaluate_residual(model: PPO, prior: FrozenDiffusionPrior, tasks: list[TaskEndpoint],
                      scenes: dict[str, Any], env_steps: int,
                      workers: int = 16) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scheduler = DiffusionInferenceScheduler(prior, RuntimeMode.FAST, max_batch_size=workers)
    policy_lock = threading.Lock()

    def one(task: TaskEndpoint) -> dict[str, Any]:
        env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)
        seed = stable_prior_seed(task.task_id); total_return = 0.0; last_info: dict[str, Any] = {}
        projection_count = support = 0; prior_sum = residual_sum = scaled_sum = final_sum = 0.0
        try:
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                with policy_lock:
                    residual, _ = model.predict(observation, deterministic=True)
                residual = np.asarray(residual, dtype=np.float32).reshape(1, 3)
                prior_action = scheduler.predict(observation, seed + env.episode_steps).reshape(1, 3)
                executed, projected = compose_residual_action(prior_action, residual, RESIDUAL_SCALE)
                prior_sum += float(np.linalg.norm(prior_action[0]))
                residual_norm = float(np.linalg.norm(residual[0])); residual_sum += residual_norm
                scaled_sum += RESIDUAL_SCALE * residual_norm
                final_norm = float(np.linalg.norm(executed[0])); final_sum += final_norm
                projection_count += int(projected[0]); support += int(final_norm > 1.0 + 1.0e-6)
                observation, reward, terminated, truncated, last_info = env.step(executed[0])
                total_return += float(reward)
            diagnostics = env.episode_diagnostics()
        finally:
            env.close()
        row = base_row(task, env_steps, last_info, total_return, diagnostics)
        count = max(1, row["episode_steps"])
        row.update({
            "policy_action_support_violation_count": support,
            "prior_norm_mean": prior_sum / count, "residual_norm_mean": residual_sum / count,
            "scaled_residual_norm_mean": scaled_sum / count, "final_action_norm_mean": final_sum / count,
            "projection_count": projection_count, "projection_fraction": projection_count / count,
        })
        return row

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="s6-residual-val") as executor:
            rows = list(executor.map(one, tasks))
    finally:
        scheduler.close()
    families = {name: aggregate([row for row in rows if row["family"] == name])
                for name in ("OPEN", "BLOCK", "SBEND")}
    summary = aggregate(rows, families=families)
    total_steps = max(1, sum(row["episode_steps"] for row in rows))
    for key in ("prior_norm_mean", "residual_norm_mean", "scaled_residual_norm_mean", "final_action_norm_mean"):
        summary[key] = sum(row[key] * row["episode_steps"] for row in rows) / total_steps
    summary["projection_count"] = sum(row["projection_count"] for row in rows)
    summary["projection_fraction"] = summary["projection_count"] / total_steps
    return rows, summary, scheduler.diagnostics()


def curve_row(step: int, summary: dict[str, Any]) -> dict[str, Any]:
    count = max(1, summary["tasks"])
    row = {
        "env_steps": step, "success": summary["success"], "success_rate": summary["success_rate"],
        "collision": summary["collision"], "collision_rate": summary["collision"] / count,
        "unsafe": summary["unsafe"], "unsafe_rate": summary["unsafe"] / count,
        "mean_return": summary["mean_return"], "timeout": summary["timeout"],
    }
    obstacle_success = 0
    for family in ("OPEN", "BLOCK", "SBEND"):
        family_summary = summary["by_family"][family]
        row[f"{family.lower()}_success"] = family_summary["success"]
        row[f"{family.lower()}_success_rate"] = family_summary["success_rate"]
        obstacle_success += family_summary["success"] if family != "OPEN" else 0
    row["obstacle_success"] = obstacle_success
    row["obstacle_success_rate"] = obstacle_success / 36.0
    return row


def preflight_fast(prior: FrozenDiffusionPrior) -> dict[str, Any]:
    with np.load(TRAIN_NPZ, allow_pickle=False) as data:
        observations = np.asarray(data["observations"][[0, 100, 1000, 10000]], dtype=np.float32)
    seeds = [stable_prior_seed(f"S6_PREFLIGHT_{index}") + index for index in range(len(observations))]
    before = capture_rng_state()
    reference = prior.predict_reference(observations, seeds)
    fast = prior.predict_fast(observations, seeds)
    after = capture_rng_state()
    error = float(np.max(np.abs(reference - fast)))
    result = {
        "checkpoint_sha256": prior.checkpoint_sha256,
        "checkpoint_identity": prior.checkpoint_sha256 == EXPECTED_S3R2_SHA256,
        "diffusion_frozen": prior.frozen, "runtime_mode": "FAST_CUDA_GRAPH_EXACT_BATCH_ONE",
        "reference_fast_max_abs_error": error, "reference_fast_bit_exact": bool(np.array_equal(reference, fast)),
        "rng_unchanged": rng_equal(before, after), "historical_batched_used": False,
    }
    if not all((result["checkpoint_identity"], result["diffusion_frozen"],
                result["reference_fast_bit_exact"], result["rng_unchanged"])):
        raise RuntimeError(f"BLOCKED_S6_FAST_PREFLIGHT: {result}")
    return result


def train(method: str, seed: int, run_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    if (run_dir / "training_manifest.json").exists():
        return json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
    if any(checkpoint_dir.glob("checkpoint_*.zip")):
        raise RuntimeError("partial checkpoint set exists without a completed manifest; preserve and classify before restart")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=True)
    tasks = load_tasks(); scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    statistics = derive_train_statistics(TRAIN_NPZ)
    config = scientific_config(method, seed)
    prior = None; fast_preflight = None; initialization = None
    if method == "diffusion_ppo":
        prior = FrozenDiffusionPrior(S3R2_CHECKPOINT, "cuda")
        fast_preflight = preflight_fast(prior)
        base_env = DummyVecEnv([make_train_env(rank, tasks, scenes) for rank in range(8)])
        train_env = DiffusionResidualVecEnv(base_env, prior, runtime_mode="fast")
    else:
        # Eight synchronous environments preserve the frozen vectorized PPO
        # contract without loading eight private copies of CUDA/PyTorch DLLs on
        # Windows (which can exhaust the system commit limit before training).
        train_env = DummyVecEnv([make_train_env(rank, tasks, scenes) for rank in range(8)])
    callback = ExactCheckpointCallback(checkpoint_dir)
    sampler = ResourceSampler()
    started = time.perf_counter()
    try:
        model = PPO(
            UnitBallActorCriticPolicy, train_env,
            learning_rate=PPO_CONFIG["learning_rate"], n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"], n_epochs=PPO_CONFIG["n_epochs"],
            gamma=PPO_CONFIG["gamma"], gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"], ent_coef=PPO_CONFIG["ent_coef"],
            vf_coef=PPO_CONFIG["vf_coef"], max_grad_norm=PPO_CONFIG["max_grad_norm"],
            normalize_advantage=PPO_CONFIG["normalize_advantage"],
            policy_kwargs=policy_kwargs(statistics), seed=seed, device="cuda", verbose=0,
        )
        if method == "diffusion_ppo":
            initialization = initialize_residual_policy(model)
        torch.cuda.reset_peak_memory_stats()
        with sampler:
            model.learn(total_timesteps=TOTAL_ENV_STEPS, callback=callback, progress_bar=False)
        actual_steps = int(model.num_timesteps)
        diagnostics = train_env.action_diagnostics() if method == "diffusion_ppo" else None
        peak_memory = int(torch.cuda.max_memory_allocated())
    finally:
        train_env.close()
    wall = time.perf_counter() - started
    if actual_steps != TOTAL_ENV_STEPS or [item["env_steps"] for item in callback.saved] != list(EVAL_STEPS):
        raise RuntimeError(f"incomplete S6 training: steps={actual_steps}, checkpoints={callback.saved}")
    manifest = {
        "task": TASK, "method": method, "seed": seed, "status": "TRAINING_COMPLETE_VAL_NOT_YET_FROZEN",
        "scientific_config": config, "scientific_config_sha256": stable_hash(config),
        "base_ppo_config_hash": PPO_CONFIG_HASH, "reward_contract": REWARD_CONTRACT,
        "reward_contract_hash": REWARD_CONTRACT_HASH,
        "dataset_sha256": {"train.npz": sha256(TRAIN_NPZ), "val.npz": sha256(VAL_NPZ)},
        "test_accessed": False, "n_envs": 8, "vec_env_backend": "DummyVecEnv",
        "actual_env_steps": actual_steps,
        "checkpoint_schedule": callback.saved, "checkpoint_serialization_rng_unchanged": True,
        "training_wall_time_s": wall, "env_steps_per_s": actual_steps / wall,
        "resources": sampler.result(), "peak_gpu_memory_bytes": peak_memory,
        "hardware": hardware(), "fast_preflight": fast_preflight,
        "residual_initialization": initialization, "train_action_diagnostics": diagnostics,
        "historical_seed1_reused": False,
    }
    write_json(run_dir / "training_manifest.json", manifest)
    return manifest


def evaluate_run(method: str, seed: int, run_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    output = run_dir / "evaluation_manifest.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    training = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
    tasks = load_tasks(); val_tasks = tasks["val"]
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    prior = FrozenDiffusionPrior(S3R2_CHECKPOINT, "cuda") if method == "diffusion_ppo" else None
    # CUDA Graph capture must occur before the evaluator starts worker threads:
    # otherwise residual-policy CUDA inference can overlap first-time capture
    # from the scheduler thread, which PyTorch explicitly forbids.  Replaying
    # the already verified graph remains the exact formal fast path.
    evaluation_fast_preflight = preflight_fast(prior) if prior is not None else None
    records = []; all_rows = []; eval_walls = []
    sampler = ResourceSampler(); total_started = time.perf_counter()
    with sampler:
        for checkpoint in training["checkpoint_schedule"]:
            step = int(checkpoint["env_steps"]); path = Path(checkpoint["path"])
            if not path.exists() or sha256(path) != checkpoint["sha256"]:
                raise RuntimeError(f"checkpoint identity failure: {path}")
            model = PPO.load(str(path), device="cuda")
            started = time.perf_counter()
            if method == "pure_ppo":
                rows, summary = evaluate_pure(model, val_tasks, scenes, step)
                scheduler = None
            else:
                assert prior is not None
                rows, summary, scheduler = evaluate_residual(model, prior, val_tasks, scenes, step)
            wall = time.perf_counter() - started
            eval_walls.append({"env_steps": step, "wall_time_s": wall, "scheduler": scheduler})
            records.append({"env_steps": step, "summary": summary})
            all_rows.extend(rows)
            print(f"VAL method={method} seed={seed} step={step} success={summary['success']}/54 "
                  f"unsafe={summary['unsafe']} return={summary['mean_return']:.4f} wall={wall:.2f}s", flush=True)
    total_wall = time.perf_counter() - total_started
    best = max(records, key=selection_key)
    curve = [curve_row(item["env_steps"], item["summary"]) for item in records]
    write_csv(run_dir / "val_rollouts.csv", all_rows)
    write_csv(run_dir / "curve.csv", curve)
    best_source = checkpoint_dir / f"checkpoint_{best['env_steps']:06d}.zip"
    last_source = checkpoint_dir / "checkpoint_500000.zip"
    shutil.copy2(best_source, checkpoint_dir / "best_model.zip")
    shutil.copy2(last_source, checkpoint_dir / "last_model.zip")
    manifest = {
        "task": TASK, "method": method, "seed": seed, "status": "POSTHOC_VAL_COMPLETE",
        "runtime_mode": "FAST_CUDA_GRAPH_EXACT_BATCH_ONE" if method == "diffusion_ppo" else "NOT_APPLICABLE",
        "evaluation_fast_preflight": evaluation_fast_preflight,
        "evaluation_steps": list(EVAL_STEPS), "checkpoint_count": len(records),
        "evaluation_wall_time_s": total_wall, "per_checkpoint_wall_time": eval_walls,
        "resources": sampler.result(), "best_step": best["env_steps"], "best_val": best["summary"],
        "final_500k_val": records[-1]["summary"], "test_accessed": False,
        "selection_rule": "success desc, unsafe asc, mean_return desc, earlier step",
        "raw_rollouts_sha256": sha256(run_dir / "val_rollouts.csv"),
        "curve_sha256": sha256(run_dir / "curve.csv"),
        "best_checkpoint_sha256": sha256(checkpoint_dir / "best_model.zip"),
        "last_checkpoint_sha256": sha256(checkpoint_dir / "last_model.zip"),
    }
    write_json(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_S6_CUDA_UNAVAILABLE")
    torch.set_num_threads(min(24, os.cpu_count() or 1))
    torch.set_num_interop_threads(min(4, os.cpu_count() or 1))
    run_name = f"{args.method}_seed_{args.seed}"
    run_dir = ROOT / "artifacts" / "s6" / "runs" / run_name
    checkpoint_dir = ROOT / "checkpoints" / "s6" / run_name
    run_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("train", "all"):
        result = train(args.method, args.seed, run_dir, checkpoint_dir)
        print(json.dumps({"training": result["status"], "wall_time_s": result["training_wall_time_s"]}, indent=2))
    if args.phase in ("evaluate", "all"):
        result = evaluate_run(args.method, args.seed, run_dir, checkpoint_dir)
        print(json.dumps({"evaluation": result["status"], "best_step": result["best_step"]}, indent=2))


if __name__ == "__main__":
    main()
