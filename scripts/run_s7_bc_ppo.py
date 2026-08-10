"""Run one preregistered BC-prior+PPO seed with deferred S2-VAL evaluation."""
from __future__ import annotations

import argparse
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
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)] + ([str(EXTERNAL)] if EXTERNAL.exists() else [])

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper

from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.ppo.bc import FrozenBCPrior, sha256_file
from quadrotor_diffusion_ppo.ppo.contract import PPO_CONFIG, PPO_CONFIG_HASH, REWARD_CONTRACT, REWARD_CONTRACT_HASH, TOTAL_ENV_STEPS
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.normalization import derive_train_statistics
from quadrotor_diffusion_ppo.ppo.residual import RESIDUAL_LOG_STD_INIT, RESIDUAL_SCALE, compose_residual_action
from quadrotor_diffusion_ppo.ppo.runtime import capture_rng_state
from quadrotor_diffusion_ppo.ppo.unit_ball import UnitBallActorCriticPolicy
from scripts.run_s4r2_ppo import aggregate, load_tasks, make_train_env, policy_kwargs, selection_key
from scripts.run_s5_residual_ppo import initialize_residual_policy
from scripts.run_s6_core import (ExactCheckpointCallback, ResourceSampler, base_row, curve_row,
                                 hardware, rng_equal, sha256, stable_hash, write_csv, write_json)


TASK = "S7-R0-MATCHED-BC-ABLATION-AND-FRESH-TOPOLOGY-GENERALIZATION-V1"
SEEDS = (20260812, 20260813, 20260814)
EVAL_STEPS = tuple(range(0, 500_001, 50_000))
BC_CHECKPOINT = ROOT / "checkpoints" / "s7" / "bc" / "best.pt"
TRAIN_NPZ = ROOT / "artifacts" / "s2" / "dataset" / "train.npz"
VAL_NPZ = ROOT / "artifacts" / "s2" / "dataset" / "val.npz"


class BCResidualVecEnv(VecEnvWrapper):
    def __init__(self, venv: VecEnv, prior: FrozenBCPrior):
        super().__init__(venv); self.prior = prior; self.current_observations = None
        self.total_actions = self.projection_count = self.support_violation_count = 0
        self.prior_norm_sum = self.residual_norm_sum = self.scaled_norm_sum = self.final_norm_sum = 0.0
        self.pending = None

    def reset(self) -> np.ndarray:
        values = self.venv.reset(); self.current_observations = np.asarray(values, dtype=np.float32); return values

    def step_async(self, actions: np.ndarray) -> None:
        if self.current_observations is None: raise RuntimeError("BC residual env used before reset")
        residual = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, 3)
        prior = self.prior.predict(self.current_observations)
        executed, projected = compose_residual_action(prior, residual, RESIDUAL_SCALE)
        prior_norm = np.linalg.norm(prior, axis=1); residual_norm = np.linalg.norm(residual, axis=1)
        final_norm = np.linalg.norm(executed, axis=1)
        self.total_actions += self.num_envs; self.projection_count += int(projected.sum())
        self.support_violation_count += int((final_norm > 1.0 + 1e-6).sum())
        self.prior_norm_sum += float(prior_norm.sum()); self.residual_norm_sum += float(residual_norm.sum())
        self.scaled_norm_sum += float((RESIDUAL_SCALE * residual_norm).sum()); self.final_norm_sum += float(final_norm.sum())
        self.pending = (prior, residual, executed, projected); self.venv.step_async(executed)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        prior, residual, executed, projected = self.pending
        for i, info in enumerate(infos):
            info.update({"a_prior": prior[i].copy(), "delta_a": residual[i].copy(),
                         "a_exec": executed[i].copy(), "residual_projection": bool(projected[i])})
        self.current_observations = np.asarray(observations, dtype=np.float32); self.pending = None
        return observations, rewards, dones, infos

    def action_diagnostics(self) -> dict[str, Any]:
        count = max(1, self.total_actions)
        return {"count": self.total_actions, "prior_norm_mean": self.prior_norm_sum / count,
                "residual_norm_mean": self.residual_norm_sum / count,
                "scaled_residual_norm_mean": self.scaled_norm_sum / count,
                "final_action_norm_mean": self.final_norm_sum / count,
                "projection_count": self.projection_count, "projection_fraction": self.projection_count / count,
                "support_violation_count": self.support_violation_count,
                "support_violation_fraction": self.support_violation_count / count}


def evaluate(model: PPO, prior: FrozenBCPrior, tasks: list[TaskEndpoint], scenes: dict[str, Any],
             env_steps: int, workers: int = 16):
    policy_lock = threading.Lock(); prior_lock = threading.Lock()
    def one(task: TaskEndpoint):
        env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)
        total_return = 0.0; info = {}; projection_count = support = 0
        prior_sum = residual_sum = scaled_sum = final_sum = 0.0
        try:
            observation, _ = env.reset(seed=task.candidate_seed); terminated = truncated = False
            while not (terminated or truncated):
                with policy_lock: residual, _ = model.predict(observation, deterministic=True)
                residual = np.asarray(residual, dtype=np.float32).reshape(1, 3)
                with prior_lock: prior_action = prior.predict(np.asarray(observation).reshape(1, 34))
                executed, projected = compose_residual_action(prior_action, residual, RESIDUAL_SCALE)
                prior_sum += float(np.linalg.norm(prior_action)); rn = float(np.linalg.norm(residual))
                residual_sum += rn; scaled_sum += RESIDUAL_SCALE * rn
                fn = float(np.linalg.norm(executed)); final_sum += fn
                projection_count += int(projected[0]); support += int(fn > 1.0 + 1e-6)
                observation, reward, terminated, truncated, info = env.step(executed[0]); total_return += reward
            diagnostics = env.episode_diagnostics()
        finally: env.close()
        row = base_row(task, env_steps, info, total_return, diagnostics); count = max(1, row["episode_steps"])
        row.update({"policy_action_support_violation_count": support, "prior_norm_mean": prior_sum / count,
                    "residual_norm_mean": residual_sum / count, "scaled_residual_norm_mean": scaled_sum / count,
                    "final_action_norm_mean": final_sum / count, "projection_count": projection_count,
                    "projection_fraction": projection_count / count})
        return row
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="s7-bc-val") as executor:
        rows = list(executor.map(one, tasks))
    families = {name: aggregate([r for r in rows if r["family"] == name]) for name in ("OPEN", "BLOCK", "SBEND")}
    summary = aggregate(rows, families=families); total = max(1, sum(r["episode_steps"] for r in rows))
    for key in ("prior_norm_mean", "residual_norm_mean", "scaled_residual_norm_mean", "final_action_norm_mean"):
        summary[key] = sum(r[key] * r["episode_steps"] for r in rows) / total
    summary["projection_count"] = sum(r["projection_count"] for r in rows)
    summary["projection_fraction"] = summary["projection_count"] / total
    return rows, summary


def train(seed: int, run_dir: Path, checkpoint_dir: Path):
    output = run_dir / "training_manifest.json"
    if output.exists(): return json.loads(output.read_text(encoding="utf-8"))
    if any(checkpoint_dir.glob("checkpoint_*.zip")): raise RuntimeError("partial S7 checkpoint set requires classification")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=True)
    tasks = load_tasks(); scenes = {s.scene_id: s for s in load_all_scenes()}; stats = derive_train_statistics(TRAIN_NPZ)
    prior = FrozenBCPrior(BC_CHECKPOINT, "cuda")
    base = DummyVecEnv([make_train_env(rank, tasks, scenes) for rank in range(8)])
    env = BCResidualVecEnv(base, prior); callback = ExactCheckpointCallback(checkpoint_dir)
    sampler = ResourceSampler(); started = time.perf_counter()
    try:
        model = PPO(UnitBallActorCriticPolicy, env, learning_rate=PPO_CONFIG["learning_rate"],
                    n_steps=PPO_CONFIG["n_steps"], batch_size=PPO_CONFIG["batch_size"], n_epochs=PPO_CONFIG["n_epochs"],
                    gamma=PPO_CONFIG["gamma"], gae_lambda=PPO_CONFIG["gae_lambda"], clip_range=PPO_CONFIG["clip_range"],
                    ent_coef=PPO_CONFIG["ent_coef"], vf_coef=PPO_CONFIG["vf_coef"], max_grad_norm=PPO_CONFIG["max_grad_norm"],
                    normalize_advantage=PPO_CONFIG["normalize_advantage"], policy_kwargs=policy_kwargs(stats),
                    seed=seed, device="cuda", verbose=0)
        initialization = initialize_residual_policy(model); torch.cuda.reset_peak_memory_stats()
        with sampler: model.learn(total_timesteps=TOTAL_ENV_STEPS, callback=callback, progress_bar=False)
        actual_steps = int(model.num_timesteps); diagnostics = env.action_diagnostics()
        peak = int(torch.cuda.max_memory_allocated())
    finally: env.close()
    wall = time.perf_counter() - started
    if actual_steps != TOTAL_ENV_STEPS or [x["env_steps"] for x in callback.saved] != list(EVAL_STEPS):
        raise RuntimeError("incomplete S7 BC+PPO training")
    config = dict(PPO_CONFIG); config.update({"seed": seed, "method": "bc_ppo", "residual_scale": RESIDUAL_SCALE,
        "residual_log_std_init": RESIDUAL_LOG_STD_INIT, "bc_checkpoint_sha256": prior.checkpoint_sha256,
        "evaluation": "deferred_S2_VAL_only_0_50k_to_500k", "test_accessed": False, "fresh_holdout_accessed": False})
    manifest = {"task": TASK, "method": "bc_ppo", "seed": seed, "status": "TRAINING_COMPLETE_VAL_NOT_YET_FROZEN",
        "scientific_config": config, "scientific_config_sha256": stable_hash(config), "base_ppo_config_hash": PPO_CONFIG_HASH,
        "reward_contract": REWARD_CONTRACT, "reward_contract_hash": REWARD_CONTRACT_HASH,
        "dataset_sha256": {"train.npz": sha256(TRAIN_NPZ), "val.npz": sha256(VAL_NPZ)}, "test_accessed": False,
        "fresh_holdout_accessed": False, "n_envs": 8, "vec_env_backend": "DummyVecEnv", "actual_env_steps": actual_steps,
        "checkpoint_schedule": callback.saved, "checkpoint_serialization_rng_unchanged": True,
        "training_wall_time_s": wall, "env_steps_per_s": actual_steps / wall, "resources": sampler.result(),
        "peak_gpu_memory_bytes": peak, "hardware": hardware(), "prior_frozen": prior.frozen,
        "residual_initialization": initialization, "train_action_diagnostics": diagnostics}
    write_json(output, manifest); return manifest


def evaluate_run(seed: int, run_dir: Path, checkpoint_dir: Path):
    output = run_dir / "evaluation_manifest.json"
    if output.exists(): return json.loads(output.read_text(encoding="utf-8"))
    training = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
    tasks = load_tasks(); scenes = {s.scene_id: s for s in load_all_scenes()}; prior = FrozenBCPrior(BC_CHECKPOINT, "cuda")
    records = []; all_rows = []; walls = []; sampler = ResourceSampler(); total_started = time.perf_counter()
    with sampler:
        for checkpoint in training["checkpoint_schedule"]:
            step = int(checkpoint["env_steps"]); path = Path(checkpoint["path"])
            if not path.exists() or sha256(path) != checkpoint["sha256"]: raise RuntimeError("S7 checkpoint identity failure")
            model = PPO.load(str(path), device="cuda"); started = time.perf_counter()
            rows, summary = evaluate(model, prior, tasks["val"], scenes, step); wall = time.perf_counter() - started
            records.append({"env_steps": step, "summary": summary}); all_rows.extend(rows)
            walls.append({"env_steps": step, "wall_time_s": wall})
            print(f"VAL method=bc_ppo seed={seed} step={step} success={summary['success']}/54 unsafe={summary['unsafe']} wall={wall:.2f}s", flush=True)
    best = max(records, key=selection_key); write_csv(run_dir / "val_rollouts.csv", all_rows)
    write_csv(run_dir / "curve.csv", [curve_row(x["env_steps"], x["summary"]) for x in records])
    shutil.copy2(checkpoint_dir / f"checkpoint_{best['env_steps']:06d}.zip", checkpoint_dir / "best_model.zip")
    shutil.copy2(checkpoint_dir / "checkpoint_500000.zip", checkpoint_dir / "last_model.zip")
    manifest = {"task": TASK, "method": "bc_ppo", "seed": seed, "status": "POSTHOC_VAL_COMPLETE",
        "evaluation_steps": list(EVAL_STEPS), "checkpoint_count": len(records),
        "evaluation_wall_time_s": time.perf_counter() - total_started, "per_checkpoint_wall_time": walls,
        "resources": sampler.result(), "best_step": best["env_steps"], "best_val": best["summary"],
        "final_500k_val": records[-1]["summary"], "test_accessed": False, "fresh_holdout_accessed": False,
        "selection_rule": "success desc, unsafe asc, mean_return desc, earlier step",
        "raw_rollouts_sha256": sha256(run_dir / "val_rollouts.csv"), "curve_sha256": sha256(run_dir / "curve.csv"),
        "best_checkpoint_sha256": sha256(checkpoint_dir / "best_model.zip"),
        "last_checkpoint_sha256": sha256(checkpoint_dir / "last_model.zip")}
    write_json(output, manifest); return manifest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all"); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("BLOCKED_S7_CUDA_UNAVAILABLE")
    torch.set_num_threads(min(24, os.cpu_count() or 1)); torch.set_num_interop_threads(min(4, os.cpu_count() or 1))
    run_dir = ROOT / "artifacts" / "s7" / "runs" / f"bc_ppo_seed_{args.seed}"
    checkpoint_dir = ROOT / "checkpoints" / "s7" / f"bc_ppo_seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("train", "all"): print(json.dumps(train(args.seed, run_dir, checkpoint_dir)["status"]))
    if args.phase in ("evaluate", "all"): print(json.dumps(evaluate_run(args.seed, run_dir, checkpoint_dir)["status"]))


if __name__ == "__main__": main()
