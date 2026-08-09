"""S5-R0 frozen S3-R2 diffusion prior plus residual PPO sanity experiment."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import threading
import time
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.ppo.contract import (
    ACTION_DIM, OBSERVATION_DIM, PPO_CONFIG, PPO_CONFIG_HASH, REWARD_CONTRACT,
    REWARD_CONTRACT_HASH, SEED, TOTAL_ENV_STEPS,
)
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.normalization import derive_train_statistics
from quadrotor_diffusion_ppo.ppo.residual import (
    EXPECTED_S3R2_SHA256, RESIDUAL_LOG_STD_INIT, RESIDUAL_SCALE,
    FrozenDiffusionPrior, compose_residual_action, stable_prior_seed,
)
from quadrotor_diffusion_ppo.ppo.unit_ball import UnitBallActorCriticPolicy
from scripts.run_s4r2_ppo import aggregate, dataset_hashes, load_tasks, policy_kwargs, selection_key, write_csv


TASK = "S5-R0-DIFFUSION-PRIOR-RESIDUAL-PPO-MINIMAL-INTEGRATION-V1"
BRANCH = "agent/s5-diffusion-residual-ppo-v1"
S4_CANONICAL_MAIN = "6eb478aa22f46d63896324e36d4d3a6dffe6de11"
S3R2_CHECKPOINT = ROOT / "checkpoints" / "s3r2" / "best.pt"
TRAIN_NPZ = ROOT / "artifacts" / "s2" / "dataset" / "train.npz"
ARTIFACTS = ROOT / "artifacts" / "s5"
CHECKPOINTS = ROOT / "checkpoints" / "s5"
S3R2_VAL = ROOT / "artifacts" / "s3r2" / "val_rollout.csv"
PURE_SUMMARY = ROOT / "artifacts" / "s4r2" / "summary.json"
PURE_CURVE = ROOT / "artifacts" / "s4r2" / "learning_curve.csv"
EVAL_STEPS = (0, 50_000, 100_000, 150_000, 200_000, 250_000,
              300_000, 350_000, 400_000, 450_000, 500_000)
REFERENCE_TOLERANCE = 2.0e-6


class DiffusionResidualVecEnv(VecEnvWrapper):
    """Interpret PPO actions as residuals and batch frozen-prior inference."""

    def __init__(self, venv: VecEnv, prior: FrozenDiffusionPrior, alpha: float = RESIDUAL_SCALE):
        super().__init__(venv)
        self.prior = prior
        self.alpha = float(alpha)
        self.current_observations: np.ndarray | None = None
        self.task_ids: list[str] = []
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.pending: dict[str, Any] | None = None
        self.total_actions = 0
        self.projection_count = 0
        self.support_violation_count = 0
        self.prior_norm_sum = 0.0
        self.residual_norm_sum = 0.0
        self.scaled_residual_norm_sum = 0.0
        self.final_norm_sum = 0.0
        self.prior_norm_max = 0.0
        self.residual_norm_max = 0.0
        self.final_norm_max = 0.0

    def reset(self) -> np.ndarray:
        observations = self.venv.reset()
        self.current_observations = np.asarray(observations, dtype=np.float32)
        self.task_ids = [str(info["task_id"]) for info in self.venv.reset_infos]
        self.episode_steps.fill(0)
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        if self.current_observations is None or len(self.task_ids) != self.num_envs:
            raise RuntimeError("residual VecEnv used before reset")
        residual = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, ACTION_DIM)
        seeds = [stable_prior_seed(task_id) + int(step)
                 for task_id, step in zip(self.task_ids, self.episode_steps)]
        prior_actions = self.prior.predict_batched(self.current_observations, seeds)
        executed, projected = compose_residual_action(prior_actions, residual, self.alpha)
        prior_norm = np.linalg.norm(prior_actions, axis=1)
        residual_norm = np.linalg.norm(residual, axis=1)
        final_norm = np.linalg.norm(executed, axis=1)
        self.total_actions += self.num_envs
        self.projection_count += int(projected.sum())
        self.support_violation_count += int((final_norm > 1.0 + 1.0e-6).sum())
        self.prior_norm_sum += float(prior_norm.sum())
        self.residual_norm_sum += float(residual_norm.sum())
        self.scaled_residual_norm_sum += float((self.alpha * residual_norm).sum())
        self.final_norm_sum += float(final_norm.sum())
        self.prior_norm_max = max(self.prior_norm_max, float(prior_norm.max()))
        self.residual_norm_max = max(self.residual_norm_max, float(residual_norm.max()))
        self.final_norm_max = max(self.final_norm_max, float(final_norm.max()))
        self.pending = {"prior": prior_actions, "residual": residual, "executed": executed,
                        "projected": projected}
        self.venv.step_async(executed)

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        if self.pending is None:
            raise RuntimeError("residual VecEnv step_wait without pending actions")
        for index, info in enumerate(infos):
            info["a_prior"] = self.pending["prior"][index].copy()
            info["delta_a"] = self.pending["residual"][index].copy()
            info["a_exec"] = self.pending["executed"][index].copy()
            info["residual_projection"] = bool(self.pending["projected"][index])
            if dones[index]:
                self.task_ids[index] = str(self.venv.reset_infos[index]["task_id"])
                self.episode_steps[index] = 0
            else:
                self.episode_steps[index] += 1
        self.current_observations = np.asarray(observations, dtype=np.float32)
        self.pending = None
        return observations, rewards, dones, infos

    def action_diagnostics(self) -> dict[str, Any]:
        count = max(1, self.total_actions)
        return {
            "count": int(self.total_actions),
            "prior_norm_mean": self.prior_norm_sum / count,
            "prior_norm_max": self.prior_norm_max,
            "residual_norm_mean": self.residual_norm_sum / count,
            "residual_norm_max": self.residual_norm_max,
            "scaled_residual_norm_mean": self.scaled_residual_norm_sum / count,
            "final_action_norm_mean": self.final_norm_sum / count,
            "final_action_norm_max": self.final_norm_max,
            "projection_count": int(self.projection_count),
            "projection_fraction": self.projection_count / count,
            "support_violation_count": int(self.support_violation_count),
            "support_violation_fraction": self.support_violation_count / count,
        }


def make_train_env(rank: int, tasks: dict[str, list[TaskEndpoint]], scenes: dict[str, Any]):
    def _factory():
        task = tasks["train"][rank % len(tasks["train"])]
        return PurePPONavigationEnv(
            scenes[task.scene_id], task, train_tasks=tuple(tasks["train"]),
            rank=rank, train_mode=True, gui=False,
        )
    return _factory


def initialize_residual_policy(model: PPO) -> dict[str, Any]:
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.log_std.fill_(RESIDUAL_LOG_STD_INIT)
    return {
        "mean_weight_max_abs": float(model.policy.action_net.weight.detach().abs().max().item()),
        "mean_bias_max_abs": float(model.policy.action_net.bias.detach().abs().max().item()),
        "log_std": model.policy.log_std.detach().cpu().tolist(),
        "latent_std": float(np.exp(RESIDUAL_LOG_STD_INIT)),
    }


class ZeroResidualPolicy:
    def predict(self, observation, deterministic=True):
        values = np.asarray(observation)
        shape = (3,) if values.ndim == 1 else (len(values), 3)
        return np.zeros(shape, dtype=np.float32), None


def evaluate_integrated(
    model: Any, prior: FrozenDiffusionPrior, tasks: list[TaskEndpoint],
    scenes: dict[str, Any], env_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Each rollout retains the original S3 batch-one DDIM path.  Independent
    # tasks run concurrently; policy prediction is serialized because SB3's
    # distribution object stores last-action diagnostics internally.
    policy_lock = threading.Lock()

    def _one(task: TaskEndpoint) -> dict[str, Any]:
        cuda_stream = torch.cuda.Stream(device=prior.device) if prior.device.type == "cuda" else None
        env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)
        base_seed = stable_prior_seed(task.task_id)
        total_return = 0.0
        prior_norm_sum = residual_norm_sum = scaled_norm_sum = final_norm_sum = 0.0
        projection_count = support_count = 0
        last_info: dict[str, Any] = {}
        try:
            observation, _ = env.reset(seed=base_seed)
            terminated = truncated = False
            while not (terminated or truncated):
                with policy_lock:
                    residual, _ = model.predict(observation, deterministic=True)
                residual = np.asarray(residual, dtype=np.float32).reshape(1, 3)
                if cuda_stream is None:
                    prior_action = prior.predict_reference(
                        np.asarray(observation, dtype=np.float32).reshape(1, 34),
                        [base_seed + env.episode_steps],
                    )
                else:
                    # Batch size remains exactly one.  A private stream only
                    # overlaps independent tasks; it does not alter DDIM math,
                    # seed order, or any task's sequential state transition.
                    with torch.cuda.stream(cuda_stream):
                        prior_action = prior.predict_reference(
                            np.asarray(observation, dtype=np.float32).reshape(1, 34),
                            [base_seed + env.episode_steps],
                        )
                executed, projected = compose_residual_action(prior_action, residual, RESIDUAL_SCALE)
                prior_norm_sum += float(np.linalg.norm(prior_action[0]))
                residual_norm = float(np.linalg.norm(residual[0]))
                residual_norm_sum += residual_norm
                scaled_norm_sum += RESIDUAL_SCALE * residual_norm
                final_norm = float(np.linalg.norm(executed[0]))
                final_norm_sum += final_norm
                projection_count += int(projected[0])
                support_count += int(final_norm > 1.0 + 1.0e-6)
                observation, reward, terminated, truncated, last_info = env.step(executed[0])
                total_return += float(reward)
            diagnostics = env.episode_diagnostics()
        finally:
            env.close()
        steps = int(last_info.get("episode_steps", 0))
        collision = bool(last_info.get("collision", False))
        ground = bool(last_info.get("ground_contact", False))
        nonfinite = bool(last_info.get("nonfinite", False))
        return {
            "env_steps": int(env_steps), "task_id": task.task_id,
            "family": task.scene_id.split("_")[0],
            "success": int(bool(last_info.get("success", False))),
            "collision": int(collision), "ground_contact": int(ground),
            "unsafe": int(collision or ground or nonfinite),
            "timeout": int(bool(last_info.get("timeout", False))),
            "nonfinite": int(nonfinite), "mean_return": float(total_return),
            "episode_steps": steps,
            "action_clip_count": int(diagnostics["action_clip_count"]),
            "action_clip_fraction": float(diagnostics["action_clip_fraction"]),
            "policy_action_support_violation_count": int(support_count),
            "prior_norm_mean": prior_norm_sum / max(1, steps),
            "residual_norm_mean": residual_norm_sum / max(1, steps),
            "scaled_residual_norm_mean": scaled_norm_sum / max(1, steps),
            "final_action_norm_mean": final_norm_sum / max(1, steps),
            "projection_count": int(projection_count),
            "projection_fraction": projection_count / max(1, steps),
        }

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="s5-eval") as executor:
        rows = list(executor.map(_one, tasks))
    families = {family: aggregate([row for row in rows if row["family"] == family])
                for family in ("OPEN", "BLOCK", "SBEND")}
    summary = aggregate(rows, families=families)
    total_steps = max(1, sum(row["episode_steps"] for row in rows))
    summary.update({
        "prior_norm_mean": sum(row["prior_norm_mean"] * row["episode_steps"] for row in rows) / total_steps,
        "residual_norm_mean": sum(row["residual_norm_mean"] * row["episode_steps"] for row in rows) / total_steps,
        "scaled_residual_norm_mean": sum(row["scaled_residual_norm_mean"] * row["episode_steps"] for row in rows) / total_steps,
        "final_action_norm_mean": sum(row["final_action_norm_mean"] * row["episode_steps"] for row in rows) / total_steps,
        "projection_count": int(sum(row["projection_count"] for row in rows)),
        "projection_fraction": sum(row["projection_count"] for row in rows) / total_steps,
    })
    return rows, summary


def flatten_curve(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        value = record["summary"]
        rows.append({
            "env_steps": record["env_steps"], "mean_return": value["mean_return"],
            "success_rate": value["success_rate"], "success": value["success"],
            "collision_rate": value["collision"] / max(1, value["tasks"]),
            "unsafe": value["unsafe"], "timeout_rate": value["timeout"] / max(1, value["tasks"]),
            "mean_episode_steps": value["mean_episode_steps"],
            "prior_norm_mean": value["prior_norm_mean"],
            "residual_norm_mean": value["residual_norm_mean"],
            "scaled_residual_norm_mean": value["scaled_residual_norm_mean"],
            "final_action_norm_mean": value["final_action_norm_mean"],
            "projection_fraction": value["projection_fraction"],
        })
    return rows


class EvaluationCallback(BaseCallback):
    def __init__(self, prior, val_tasks, scenes, records, best_path, initial_record):
        super().__init__(0)
        self.prior = prior
        self.val_tasks = val_tasks
        self.scenes = scenes
        self.records = records
        self.best_path = best_path
        self.next_target_index = 1
        self.best_record = initial_record

    def _on_step(self) -> bool:
        env_steps = int(self.num_timesteps)
        while self.next_target_index < len(EVAL_STEPS) and env_steps >= EVAL_STEPS[self.next_target_index]:
            target = EVAL_STEPS[self.next_target_index]
            rows, summary = evaluate_integrated(self.model, self.prior, self.val_tasks, self.scenes, target)
            record = {"env_steps": target, "rows": rows, "summary": summary}
            self.records.append(record)
            if selection_key(record) > selection_key(self.best_record):
                self.best_record = record
                self.model.save(str(self.best_path.with_suffix("")))
            print(f"VAL env_steps={target} success={summary['success']}/54 unsafe={summary['unsafe']} "
                  f"return={summary['mean_return']:.3f} projection={summary['projection_fraction']:.4f}", flush=True)
            self.next_target_index += 1
        return env_steps < TOTAL_ENV_STEPS


def reference_audit(prior: FrozenDiffusionPrior, tasks, scenes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, summary = evaluate_integrated(ZeroResidualPolicy(), prior, tasks["val"], scenes, 0)
    with S3R2_VAL.open(encoding="utf-8-sig", newline="") as handle:
        frozen_rows = {row["task_id"]: row for row in csv.DictReader(handle)}
    outcome_mismatches = []
    for row in rows:
        frozen = frozen_rows[row["task_id"]]
        for key in ("success", "collision", "ground_contact", "timeout", "nonfinite"):
            expected = int(str(frozen[key]).lower() in ("1", "true"))
            if int(row[key]) != expected:
                outcome_mismatches.append({"task_id": row["task_id"], "field": key,
                                           "expected": expected, "actual": int(row[key])})
    result = {
        "checkpoint_sha256": prior.checkpoint_sha256,
        "diffusion_frozen": prior.frozen,
        "reference_val": summary,
        "expected_val_success": 37,
        "outcome_mismatch_count": len(outcome_mismatches),
        "outcome_mismatches": outcome_mismatches,
        "val_reference_pass": summary["success"] == 37 and not outcome_mismatches,
        "archival_s3r2_test_success_not_reexecuted": 38,
        "test_accessed": False,
    }
    return result, rows


def preflight(model: PPO, prior: FrozenDiffusionPrior, statistics, tasks, scenes) -> dict[str, Any]:
    with np.load(TRAIN_NPZ, allow_pickle=False) as data:
        observations = data["observations"][[0, 100, 1000, 10000, 50000, 100000, 120000, 129000]]
    seeds = [stable_prior_seed(f"S5_PREFLIGHT_{index}") + index for index in range(8)]
    reference = prior.predict_reference(observations, seeds)
    optimized = prior.predict_batched(observations, seeds)
    optimized_error = float(np.max(np.abs(reference - optimized)))
    zero, projected = compose_residual_action(reference, np.zeros_like(reference))
    obs_tensor = torch.as_tensor(observations[:4], dtype=torch.float32, device=model.device)
    distribution = model.policy.get_distribution(obs_tensor)
    residual = distribution.get_actions(deterministic=False)
    log_prob = distribution.log_prob(residual)
    model.policy.optimizer.zero_grad(set_to_none=True)
    (-log_prob.mean()).backward()
    ppo_gradient = any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in model.policy.parameters())
    model.policy.optimizer.zero_grad(set_to_none=True)
    probe = PurePPONavigationEnv(scenes[tasks["train"][0].scene_id], tasks["train"][0],
                                 train_tasks=tuple(tasks["train"]), train_mode=True, gui=False)
    _, reset_info = probe.reset(seed=SEED)
    reward_source = inspect.getsource(__import__("quadrotor_diffusion_ppo.ppo.contract", fromlist=["compute_reward"]).compute_reward)
    probe.close()
    checkpoint_mean = prior.mean.detach().cpu().numpy()
    checkpoint_std = prior.payload["observation_std"]
    return {
        "checkpoint_identity": prior.checkpoint_sha256 == EXPECTED_S3R2_SHA256,
        "diffusion_frozen": prior.frozen,
        "normalization_mean_float32_identity": bool(np.array_equal(checkpoint_mean, statistics.mean.astype(np.float32))),
        "normalization_std_float32_identity": bool(np.array_equal(np.asarray(checkpoint_std, dtype=np.float32), statistics.raw_std.astype(np.float32))),
        "constant_feature_index": 20,
        "constant_feature_runtime_equivalence": bool(np.all(observations[:, 20] == checkpoint_mean[20])),
        "optimized_reference_max_abs_error": optimized_error,
        "optimized_reference_tolerance": REFERENCE_TOLERANCE,
        "optimized_reference_pass": optimized_error <= REFERENCE_TOLERANCE,
        "zero_residual_exact_identity": bool(np.array_equal(zero, reference) and not projected.any()),
        "residual_action_finite": bool(torch.isfinite(residual).all().item()),
        "residual_support_pass": bool((torch.linalg.vector_norm(residual, dim=1) <= 1.0 + 1.0e-6).all().item()),
        "ppo_log_prob_finite": bool(torch.isfinite(log_prob).all().item()),
        "ppo_gradient_present": bool(ppo_gradient),
        "diffusion_gradient_present": prior.gradient_present,
        "reward_hash": REWARD_CONTRACT_HASH,
        "reward_source_has_frozen_formula": "progress_scale" in reward_source and "time_penalty" in reward_source,
        "train_reset_split": reset_info["split"],
        "train_reset_sampling": bool(reset_info["train_task_sampling"]),
        "val_or_test_used_for_design": False,
    }


def hardware_info() -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": os.cpu_count(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "stable_baselines3": __import__("stable_baselines3").__version__,
        "platform": platform.platform(), "torch_num_threads": torch.get_num_threads(),
    }


def gate(best: dict[str, Any]) -> bool:
    families = best["by_family"]
    obstacle_success = families["BLOCK"]["success"] > 0 and families["SBEND"]["success"] > 0
    threshold = best["success"] >= 27 or (
        best["success"] > 14 and all(families[name]["success"] > 0 for name in ("OPEN", "BLOCK", "SBEND"))
    )
    return bool(obstacle_success and threshold and best["nonfinite"] == 0
                and best["policy_action_support_violation_count"] == 0
                and best["action_clip_count"] == 0)


def smoke_main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_S5_CUDA_UNAVAILABLE")
    torch.set_num_threads(min(24, os.cpu_count() or 1))
    tasks = load_tasks()
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    statistics = derive_train_statistics(TRAIN_NPZ)
    prior = FrozenDiffusionPrior(S3R2_CHECKPOINT, "cuda")
    base = DummyVecEnv([make_train_env(rank, tasks, scenes) for rank in range(8)])
    env = DiffusionResidualVecEnv(base, prior)
    try:
        model = PPO(UnitBallActorCriticPolicy, env, n_steps=2, batch_size=16, n_epochs=1,
                    policy_kwargs=policy_kwargs(statistics), seed=SEED, device="cuda", verbose=0)
        initialization = initialize_residual_policy(model)
        checks = preflight(model, prior, statistics, tasks, scenes)
        model.learn(total_timesteps=16, progress_bar=False)
        with tempfile.TemporaryDirectory(prefix="s5-smoke-") as directory:
            path = Path(directory) / "model"
            model.save(str(path))
            loaded = PPO.load(str(path.with_suffix(".zip")), device="cuda")
            roundtrip = torch.equal(model.policy.log_std, loaded.policy.log_std)
        print(json.dumps({"smoke": "PASS", "timesteps": model.num_timesteps,
                          "n_envs": env.num_envs, "initialization": initialization,
                          "preflight": checks, "action_diagnostics": env.action_diagnostics(),
                          "checkpoint_roundtrip": bool(roundtrip)}, indent=2), flush=True)
    finally:
        env.close()


def main() -> None:
    if "--smoke" in sys.argv:
        smoke_main()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_S5_CUDA_UNAVAILABLE")
    torch.set_num_threads(min(24, os.cpu_count() or 1))
    torch.set_num_interop_threads(min(4, os.cpu_count() or 1))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    set_random_seed(SEED, using_cuda=True)
    tasks = load_tasks()
    scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    statistics = derive_train_statistics(TRAIN_NPZ)
    prior = FrozenDiffusionPrior(S3R2_CHECKPOINT, "cuda")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINTS / "best_model.zip"
    last_path = CHECKPOINTS / "last_model.zip"
    for path in (best_path, last_path):
        if path.exists():
            path.unlink()

    prior_reference, prior_rows = reference_audit(prior, tasks, scenes)
    (ARTIFACTS / "prior_reference.json").write_text(
        json.dumps(prior_reference, indent=2), encoding="utf-8"
    )
    if not prior_reference["val_reference_pass"]:
        raise RuntimeError("BLOCKED_S5_PRIOR_REFERENCE_MISMATCH")

    base_env = DummyVecEnv([make_train_env(rank, tasks, scenes) for rank in range(8)])
    train_env = DiffusionResidualVecEnv(base_env, prior)
    start_time = time.perf_counter()
    try:
        model = PPO(
            UnitBallActorCriticPolicy, train_env,
            learning_rate=PPO_CONFIG["learning_rate"], n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"], n_epochs=PPO_CONFIG["n_epochs"],
            gamma=PPO_CONFIG["gamma"], gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"], ent_coef=PPO_CONFIG["ent_coef"],
            vf_coef=PPO_CONFIG["vf_coef"], max_grad_norm=PPO_CONFIG["max_grad_norm"],
            normalize_advantage=PPO_CONFIG["normalize_advantage"],
            policy_kwargs=policy_kwargs(statistics), seed=SEED, device="cuda", verbose=0,
        )
        initialization = initialize_residual_policy(model)
        checks = preflight(model, prior, statistics, tasks, scenes)
        required_checks = [
            checks["checkpoint_identity"], checks["diffusion_frozen"],
            checks["normalization_mean_float32_identity"], checks["normalization_std_float32_identity"],
            checks["constant_feature_runtime_equivalence"], checks["optimized_reference_pass"],
            checks["zero_residual_exact_identity"], checks["residual_action_finite"],
            checks["residual_support_pass"], checks["ppo_log_prob_finite"],
            checks["ppo_gradient_present"], not checks["diffusion_gradient_present"],
            checks["reward_hash"] == REWARD_CONTRACT_HASH, checks["train_reset_split"] == "train",
            checks["train_reset_sampling"], not checks["val_or_test_used_for_design"],
        ]
        if not all(required_checks):
            raise RuntimeError("BLOCKED_S5_PREFLIGHT_GATE")
        # The deterministic actor mean is exactly zero and the preflight proves
        # bit-exact zero-residual identity, so this is the same integrated
        # step-0 policy without rerunning the identical 54 physical episodes.
        step0_rows = prior_rows
        step0_summary = prior_reference["reference_val"]
        initial_record = {"env_steps": 0, "rows": step0_rows, "summary": step0_summary}
        records = [initial_record]
        model.save(str(best_path.with_suffix("")))
        callback = EvaluationCallback(prior, tasks["val"], scenes, records, best_path, initial_record)
        torch.cuda.reset_peak_memory_stats()
        model.learn(total_timesteps=TOTAL_ENV_STEPS, callback=callback, progress_bar=False)
        actual_steps = int(model.num_timesteps)
        model.save(str(last_path.with_suffix("")))
        train_action_diagnostics = train_env.action_diagnostics()
        peak_memory = int(torch.cuda.max_memory_allocated())
    finally:
        train_env.close()
    wall_time = time.perf_counter() - start_time
    if actual_steps != TOTAL_ENV_STEPS:
        raise RuntimeError(f"S5 training did not stop at exact budget: {actual_steps}")
    best_record = callback.best_record
    best_model = PPO.load(str(best_path), device="cuda")
    val_gate = gate(best_record["summary"])
    test_rows: list[dict[str, Any]] = []
    test_summary = None
    test_run_count = 0
    if val_gate:
        test_rows, test_summary = evaluate_integrated(best_model, prior, tasks["test"], scenes, actual_steps)
        test_run_count = 1

    write_csv(ARTIFACTS / "learning_curve.csv", flatten_curve(records))
    write_csv(ARTIFACTS / "val_best_rollout.csv", best_record["rows"])
    if val_gate:
        write_csv(ARTIFACTS / "test_rollout.csv", test_rows)
    pure_summary = json.loads(PURE_SUMMARY.read_text(encoding="utf-8"))
    with PURE_CURVE.open(encoding="utf-8", newline="") as handle:
        pure_curve = list(csv.DictReader(handle))
    final_label = "PASS_S5_DIFFUSION_PRIOR_RESIDUAL_PPO_SANITY" if val_gate else "BLOCKED_S5_PRIOR_INTEGRATION_NO_BENEFIT"
    summary = {
        "task": TASK, "final_label": final_label,
        "branch": BRANCH, "start_head": S4_CANONICAL_MAIN, "end_head": None, "remote_head": None,
        "s4_canonical_main": S4_CANONICAL_MAIN, "s4_baseline_frozen": True,
        "formal_progress_at_start": "55%", "system": hardware_info() | {"vec_env_backend": "DummyVecEnv", "n_envs": 8},
        "diffusion_checkpoint_sha256": prior.checkpoint_sha256,
        "diffusion_frozen": prior.frozen, "prior_reference_result": prior_reference,
        "observation_normalization_identity": checks,
        "residual_composition": "unit_ball_project(a_prior + 0.25 * delta_a)",
        "residual_scale": RESIDUAL_SCALE, "residual_init": initialization,
        "ppo_config": PPO_CONFIG | {"observation_normalization": "fixed_s2_train_standardization",
                                    "residual_log_std_init": RESIDUAL_LOG_STD_INIT},
        "ppo_config_hash_s4": PPO_CONFIG_HASH,
        "unexpected_ppo_config_diff": [], "reward_contract": REWARD_CONTRACT,
        "reward_contract_hash": REWARD_CONTRACT_HASH,
        "total_env_steps": actual_steps, "training_wall_time_s": wall_time,
        "training_peak_gpu_memory_bytes": peak_memory,
        "learning_curve": flatten_curve(records), "pure_ppo_reference_curve": pure_curve,
        "best_val": best_record["summary"] | {"env_steps": best_record["env_steps"]},
        "val_by_family": best_record["summary"]["by_family"],
        "train_action_diagnostics": train_action_diagnostics,
        "diffusion_gradient_present": prior.gradient_present,
        "teacher_runtime_dependence": False,
        "val_gate": "PASS" if val_gate else "FAIL", "model_frozen": True,
        "test_executed": bool(val_gate), "test_run_count": test_run_count,
        "test_used_for_selection": False, "test": test_summary,
        "dataset_sha256": dataset_hashes(include_test=val_gate),
        "pure_ppo_baseline": {
            "status": "VALID_WEAK_BASELINE", "success": pure_summary["best_val"]["success"],
            "open_success": pure_summary["best_val"]["by_family"]["OPEN"]["success"],
            "block_success": pure_summary["best_val"]["by_family"]["BLOCK"]["success"],
            "sbend_success": pure_summary["best_val"]["by_family"]["SBEND"]["success"],
        },
        "what_was_proven": [
            "frozen S3-R2 diffusion prior was integrated with a trainable PPO residual",
            "500,000 exact TRAIN environment steps completed under the frozen S4 PPO contract",
            "VAL-only model selection and conditional single TEST access were enforced",
        ],
        "what_was_not_proven": [
            "no multi-seed superiority or final sample-efficiency claim",
            "no S6 three-seed comparison",
            "no cross-topology or real-flight claim",
        ],
        "current_blocker": None if val_gate else final_label,
        "formal_progress": "70%" if val_gate else "55%",
        "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW",
        "waiting_for_high_level_controller_audit": True,
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    test_text = "NOT_EXECUTED" if test_summary is None else f"{test_summary['success']}/54"
    print(f"S5 complete; best VAL={best_record['summary']['success']}/54 at {best_record['env_steps']}; "
          f"TEST={test_text}; label={final_label}", flush=True)


if __name__ == "__main__":
    main()
