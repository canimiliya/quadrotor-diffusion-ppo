"""S6-P0 runtime profiling and scientific-equivalence audit."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
if EXTERNAL.exists():
    sys.path.insert(0, str(EXTERNAL))

import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.evaluation.runtime import DiffusionInferenceScheduler, RuntimeMode
from quadrotor_diffusion_ppo.ppo.contract import N_ENVS, PPO_CONFIG, PPO_CONFIG_HASH, REWARD_CONTRACT_HASH, SEED
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.normalization import derive_train_statistics
from quadrotor_diffusion_ppo.ppo.residual import FrozenDiffusionPrior, stable_prior_seed
from quadrotor_diffusion_ppo.ppo.runtime import capture_rng_state
from quadrotor_diffusion_ppo.ppo.unit_ball import UnitBallActorCriticPolicy
from scripts.run_s4r2_ppo import aggregate, load_tasks, policy_kwargs
from scripts.run_s5_residual_ppo import DiffusionResidualVecEnv, initialize_residual_policy, make_train_env


TASK = "S6-P0-HIGH-THROUGHPUT-EXPERIMENT-RUNTIME-OPTIMIZATION-V1"
BRANCH = "agent/s6p0-runtime-optimization-v1"
START_HEAD = "435e9d58ba41f627195354a333b3f5051d451b4c"
CHECKPOINT = ROOT / "checkpoints" / "s3r2" / "best.pt"
TRAIN_NPZ = ROOT / "artifacts" / "s2" / "dataset" / "train.npz"
ARCHIVAL_VAL = ROOT / "artifacts" / "s3r2" / "val_rollout.csv"
ARTIFACTS = ROOT / "artifacts" / "s6p0"
WORKER_COUNTS = (8, 12, 16, 20)
ACTION_TOLERANCE = 2.0e-6


class ResourceSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.cpu: list[list[float]] = []
        self.gpu: list[float] = []
        self.memory_mib: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.cpu.append(psutil.cpu_percent(interval=None, percpu=True))
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=2,
                ).stdout.strip().split(",")
                self.gpu.append(float(result[0])); self.memory_mib.append(float(result[1]))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        psutil.cpu_percent(interval=None, percpu=True)
        self._thread.start(); return self

    def __exit__(self, *_args):
        self._stop.set(); self._thread.join()

    def result(self) -> dict[str, Any]:
        cpu = np.asarray(self.cpu, dtype=np.float64)
        per_core = cpu.mean(axis=0).tolist() if cpu.size else []
        return {
            "samples": len(self.cpu),
            "cpu_total_mean_percent": float(cpu.mean()) if cpu.size else None,
            "cpu_per_logical_mean_percent": per_core,
            "cpu_active_logical_processors_mean": float(np.sum(np.asarray(per_core) > 10.0)) if per_core else 0.0,
            "gpu_utilization_mean_percent": float(np.mean(self.gpu)) if self.gpu else None,
            "gpu_utilization_max_percent": float(np.max(self.gpu)) if self.gpu else None,
            "gpu_memory_mean_mib": float(np.mean(self.memory_mib)) if self.memory_mib else None,
            "gpu_memory_max_mib": float(np.max(self.memory_mib)) if self.memory_mib else None,
        }


def timed(call: Callable[[], Any], repeats: int = 1) -> tuple[float, Any, dict[str, Any]]:
    durations = []; output = None; resources = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        with ResourceSampler() as sampler:
            started = time.perf_counter(); output = call(); torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        resources = sampler.result()
    return min(durations), output, resources or {}


def hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(), "logical_processors": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "ppo_config_hash": PPO_CONFIG_HASH, "reward_contract_hash": REWARD_CONTRACT_HASH,
        "n_envs_frozen": N_ENVS,
    }


def profile_diffusion(prior: FrozenDiffusionPrior) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with np.load(ROOT / "artifacts" / "s2" / "dataset" / "val.npz", allow_pickle=False) as data:
        observations = np.asarray(data["observations"][:64], dtype=np.float32)
    seeds = [stable_prior_seed(f"S6P0_MICRO_{index}") + index for index in range(len(observations))]
    for function in (prior.predict_original_reference, prior.predict_reference, prior.predict_batched):
        function(observations[:2], seeds[:2])
    original_s, original, original_resources = timed(
        lambda: prior.predict_original_reference(observations, seeds), repeats=2)
    optimized_s, optimized, optimized_resources = timed(
        lambda: prior.predict_reference(observations, seeds), repeats=3)
    batched_s, batched, batched_resources = timed(
        lambda: prior.predict_batched(observations, seeds), repeats=3)
    fast_s, fast, fast_resources = timed(
        lambda: prior.predict_fast(observations, seeds), repeats=3)

    condition_s, conditions, _ = timed(lambda: prior._conditions(observations), repeats=5)
    noise_s, initial_noise, _ = timed(
        lambda: [prior._initial_noise(seed) for seed in seeds], repeats=3)
    denoise_s, _, _ = timed(
        lambda: [prior._ddim_from_initial(condition.unsqueeze(0), noise.clone())[0]
                 for condition, noise in zip(conditions, initial_noise)], repeats=3)
    baseline = {
        "task": TASK, "runtime_path": "original_reference", "hardware": hardware(),
        "sample_count": len(observations), "wall_time_s": original_s,
        "diffusion_actions_per_s": len(observations) / original_s,
        "wall_time_breakdown_s": {
            "observation_preprocessing": condition_s,
            "initial_noise_generation": noise_s,
            "ten_step_ddim_and_transfers": max(0.0, original_s - condition_s - noise_s),
        },
        "resources": original_resources,
        "historical_integrated_500k_wall_time_s": 5792.2036968,
    }
    optimized_profile = {
        "task": TASK, "runtime_path": "optimized_reference", "hardware": hardware(),
        "sample_count": len(observations), "wall_time_s": optimized_s,
        "diffusion_actions_per_s": len(observations) / optimized_s,
        "exact_reference_speedup": original_s / optimized_s,
        "wall_time_breakdown_s": {
            "observation_preprocessing": condition_s,
            "initial_noise_generation": noise_s,
            "ten_step_ddim": denoise_s,
            "other_and_transfers": max(0.0, optimized_s - condition_s - noise_s - denoise_s),
        },
        "resources": optimized_resources,
    }
    equivalence = {
        "action_tolerance": ACTION_TOLERANCE,
        "optimized_reference_max_abs_error": float(np.max(np.abs(original - optimized))),
        "optimized_reference_bit_exact": bool(np.array_equal(original, optimized)),
        "optimized_reference_action_gate": bool(np.max(np.abs(original - optimized)) <= ACTION_TOLERANCE),
        "batched_max_abs_error": float(np.max(np.abs(original - batched))),
        "batched_action_gate": bool(np.max(np.abs(original - batched)) <= ACTION_TOLERANCE),
        "batched_wall_time_s": batched_s,
        "batched_actions_per_s": len(observations) / batched_s,
        "batched_speedup": original_s / batched_s,
        "batched_resources": batched_resources,
        "cuda_graph_reference_max_abs_error": float(np.max(np.abs(original - fast))),
        "cuda_graph_reference_bit_exact": bool(np.array_equal(original, fast)),
        "cuda_graph_reference_action_gate": bool(np.max(np.abs(original - fast)) <= ACTION_TOLERANCE),
        "cuda_graph_reference_wall_time_s": fast_s,
        "cuda_graph_reference_actions_per_s": len(observations) / fast_s,
        "cuda_graph_reference_speedup": original_s / fast_s,
        "cuda_graph_reference_resources": fast_resources,
    }
    return baseline, optimized_profile, equivalence


def intermediate_audit(prior: FrozenDiffusionPrior) -> dict[str, Any]:
    with np.load(ROOT / "artifacts" / "s2" / "dataset" / "val.npz", allow_pickle=False) as data:
        observations = np.asarray(data["observations"][[0, 17, 101, 503, 1007, 2003, 4001, 8009]], dtype=np.float32)
    seeds = [stable_prior_seed(f"S6P0_TRACE_{index}") + index for index in range(len(observations))]
    reference = [prior.trace_reference(observation, seed) for observation, seed in zip(observations, seeds)]
    batched = prior.trace_batched(observations, seeds)
    maxima = {key: 0.0 for key in ("x_t", "raw_x0", "bounded_x0", "epsilon", "next_sample")}
    for item_index, trace in enumerate(reference):
        for step_index, record in enumerate(trace):
            for key in maxima:
                maxima[key] = max(maxima[key], float(np.max(np.abs(
                    np.asarray(record[key]) - np.asarray(batched[step_index][key][item_index:item_index + 1])
                ))))
    return {"samples": len(observations), "steps": 10, "max_abs_error_by_layer": maxima,
            "all_layers_within_tolerance": all(value <= ACTION_TOLERANCE for value in maxima.values())}


def evaluate_prior(prior: FrozenDiffusionPrior, tasks: list[TaskEndpoint], scenes: dict[str, Any],
                   mode: RuntimeMode, workers: int, *, audit_batched: bool = False
                   ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scheduler = DiffusionInferenceScheduler(
        prior, mode, max_batch_size=workers, audit_unverified_batched=audit_batched
    )

    def one(task: TaskEndpoint) -> dict[str, Any]:
        env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)
        seed = stable_prior_seed(task.task_id); total_return = 0.0; last_info: dict[str, Any] = {}
        try:
            observation, _ = env.reset(seed=seed); terminated = truncated = False
            while not (terminated or truncated):
                action = scheduler.predict(observation, seed + env.episode_steps)
                observation, reward, terminated, truncated, last_info = env.step(action)
                total_return += float(reward)
            diagnostics = env.episode_diagnostics()
        finally:
            env.close()
        collision = bool(last_info.get("collision", False)); ground = bool(last_info.get("ground_contact", False))
        nonfinite = bool(last_info.get("nonfinite", False))
        return {
            "task_id": task.task_id, "family": task.scene_id.split("_")[0],
            "success": int(bool(last_info.get("success", False))), "collision": int(collision),
            "ground_contact": int(ground), "unsafe": int(collision or ground or nonfinite),
            "timeout": int(bool(last_info.get("timeout", False))), "nonfinite": int(nonfinite),
            "mean_return": total_return, "episode_steps": int(last_info.get("episode_steps", 0)),
            "action_clip_count": int(diagnostics["action_clip_count"]),
            "policy_action_support_violation_count": 0,
        }

    try:
        with ResourceSampler() as sampler:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="s6p0-eval") as executor:
                rows = list(executor.map(one, tasks))
            wall = time.perf_counter() - started
    finally:
        scheduler.close()
    families = {family: aggregate([row for row in rows if row["family"] == family])
                for family in ("OPEN", "BLOCK", "SBEND")}
    summary = aggregate(rows, families=families)
    summary.update({"wall_time_s": wall, "tasks_per_min": len(tasks) * 60.0 / wall,
                    "workers": workers, "runtime_mode": mode.value,
                    "inference_path": "unverified_batched_audit" if audit_batched else mode.value})
    return rows, summary, scheduler.diagnostics() | sampler.result()


def outcome_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with ARCHIVAL_VAL.open(encoding="utf-8-sig", newline="") as handle:
        expected = {row["task_id"]: row for row in csv.DictReader(handle)}
    mismatches = []
    for row in rows:
        frozen = expected[row["task_id"]]
        for key in ("success", "collision", "ground_contact", "timeout", "nonfinite"):
            value = int(str(frozen[key]).lower() in ("1", "true"))
            if int(row[key]) != value:
                mismatches.append({"task_id": row["task_id"], "field": key,
                                   "expected": value, "actual": int(row[key])})
    return {"outcome_mismatch_count": len(mismatches), "outcome_mismatches": mismatches,
            "outcome_identity": not mismatches}


def train_smoke(prior: FrozenDiffusionPrior, tasks, scenes,
                selected_runtime_mode: str | None = None) -> dict[str, Any]:
    statistics = derive_train_statistics(TRAIN_NPZ)
    common = dict(
        learning_rate=PPO_CONFIG["learning_rate"], n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"], n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"], gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"], ent_coef=PPO_CONFIG["ent_coef"],
        vf_coef=PPO_CONFIG["vf_coef"], max_grad_norm=PPO_CONFIG["max_grad_norm"],
        normalize_advantage=PPO_CONFIG["normalize_advantage"],
        policy_kwargs=policy_kwargs(statistics), seed=SEED, device="cuda", verbose=0,
    )
    results = {}
    names = ("pure_ppo", "diffusion_integrated_historical_batched",
             "diffusion_integrated_reference", "diffusion_integrated_fast")
    if selected_runtime_mode is not None:
        names = (f"diffusion_integrated_{selected_runtime_mode}",)
    for name in names:
        base = DummyVecEnv([make_train_env(rank, tasks, scenes) for rank in range(N_ENVS)])
        runtime_mode = name.removeprefix("diffusion_integrated_")
        env = base if name == "pure_ppo" else DiffusionResidualVecEnv(base, prior, runtime_mode=runtime_mode)
        try:
            model = PPO(UnitBallActorCriticPolicy, env, **common)
            if name != "pure_ppo":
                initialize_residual_policy(model)
            wall, _, resources = timed(lambda: model.learn(total_timesteps=8192, progress_bar=False))
            before = capture_rng_state()
            with tempfile.TemporaryDirectory(prefix="s6p0-checkpoint-") as directory:
                model.save(str(Path(directory) / "model"))
            after = capture_rng_state()
            rng_unchanged = (
                before["python"] == after["python"]
                and np.array_equal(before["numpy"][1], after["numpy"][1])
                and before["numpy"][:1] == after["numpy"][:1]
                and before["numpy"][2:] == after["numpy"][2:]
                and torch.equal(before["torch_cpu"], after["torch_cpu"])
                and all(torch.equal(left, right) for left, right in zip(
                    before["torch_cuda"], after["torch_cuda"]
                ))
            )
            if not rng_unchanged:
                raise RuntimeError("real PPO checkpoint serialization changed RNG state")
            results[name] = {"env_steps": int(model.num_timesteps), "wall_time_s": wall,
                             "env_steps_per_s": model.num_timesteps / wall, "resources": resources,
                             "checkpoint_serialization_rng_unchanged": True}
        finally:
            env.close()
    return results


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("micro", "val", "train", "runtime-smoke", "all"), default="all")
    parser.add_argument("--runtime-mode", choices=("reference", "fast"), default="fast",
                        help="Runtime mode exposed to future formal runners; audits still execute both modes.")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("S6P0 requires CUDA")
    torch.set_num_threads(min(24, os.cpu_count() or 1)); torch.set_num_interop_threads(4)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    prior = FrozenDiffusionPrior(CHECKPOINT, "cuda")

    if args.phase in ("micro", "all"):
        baseline, optimized, equivalence = profile_diffusion(prior)
        equivalence["intermediate_audit"] = intermediate_audit(prior)
        write_json(ARTIFACTS / "profile_baseline.json", baseline)
        write_json(ARTIFACTS / "profile_optimized.json", optimized)
        write_json(ARTIFACTS / "equivalence_audit.json", equivalence)
        print(json.dumps({"baseline": baseline, "optimized": optimized, "equivalence": equivalence}, indent=2), flush=True)

    if args.phase in ("val", "train", "runtime-smoke", "all"):
        tasks = load_tasks(); scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    if args.phase in ("val", "all"):
        subset = []
        for family in ("OPEN", "BLOCK", "SBEND"):
            subset.extend([task for task in tasks["val"] if task.scene_id.startswith(family)][:4])
        benchmarks = []
        for workers in WORKER_COUNTS:
            _, summary, resources = evaluate_prior(prior, subset, scenes, RuntimeMode.REFERENCE, workers)
            benchmarks.append(summary | resources | {"scope": "12-task-balanced-reference"})
            print(f"REFERENCE workers={workers} tasks/min={summary['tasks_per_min']:.3f}", flush=True)
        selected_workers = max(benchmarks, key=lambda row: row["tasks_per_min"])["workers"]
        ref_rows, ref_summary, ref_resources = evaluate_prior(
            prior, tasks["val"], scenes, RuntimeMode.REFERENCE, int(selected_workers))
        fast_rows, fast_summary, fast_resources = evaluate_prior(
            prior, tasks["val"], scenes, RuntimeMode.FAST, int(selected_workers))
        batched_rows, batched_summary, batched_resources = evaluate_prior(
            prior, tasks["val"], scenes, RuntimeMode.FAST, int(selected_workers), audit_batched=True)
        reference_outcomes = outcome_audit(ref_rows); fast_outcomes = outcome_audit(fast_rows)
        batched_outcomes = outcome_audit(batched_rows)
        with (ARTIFACTS / "runtime_benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in benchmarks for key in row if not isinstance(row[key], (list, dict))})
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows([{key: value for key, value in row.items() if key in fields} for row in benchmarks])
        equivalence_path = ARTIFACTS / "equivalence_audit.json"
        equivalence = json.loads(equivalence_path.read_text(encoding="utf-8")) if equivalence_path.exists() else {}
        equivalence.update({
            "reference_54_task": ref_summary | reference_outcomes,
            "fast_cuda_graph_54_task": fast_summary | fast_outcomes,
            "fast_cuda_graph_diffusion_inference": "VERIFIED_EQUIVALENT" if fast_outcomes["outcome_identity"] else "REJECTED",
            "batched_ddim_54_task": batched_summary | batched_outcomes,
            "fast_batched_diffusion_inference": "VERIFIED_EQUIVALENT" if batched_outcomes["outcome_identity"] else "REJECTED",
            "selected_evaluation_workers": int(selected_workers),
            "reference_resources": ref_resources, "fast_resources": fast_resources,
            "batched_audit_resources": batched_resources,
            "test_partition_accessed": False,
        })
        write_json(equivalence_path, equivalence)
        optimized_path = ARTIFACTS / "profile_optimized.json"
        optimized = json.loads(optimized_path.read_text(encoding="utf-8")) if optimized_path.exists() else {}
        optimized.update({"evaluation_worker_benchmarks": benchmarks, "selected_evaluation_workers": selected_workers,
                          "reference_54_task": ref_summary, "fast_54_task": fast_summary,
                          "batched_audit_54_task": batched_summary})
        write_json(optimized_path, optimized)
        print(json.dumps({"reference": ref_summary | reference_outcomes,
                          "fast": fast_summary | fast_outcomes,
                          "batched_audit": batched_summary | batched_outcomes}, indent=2), flush=True)
    if args.phase in ("train", "all"):
        result = train_smoke(prior, tasks, scenes)
        optimized_path = ARTIFACTS / "profile_optimized.json"
        optimized = json.loads(optimized_path.read_text(encoding="utf-8")) if optimized_path.exists() else {}
        optimized["train_smoke"] = result
        write_json(optimized_path, optimized)
        print(json.dumps({"train_smoke": result}, indent=2), flush=True)
    if args.phase == "runtime-smoke":
        result = train_smoke(prior, tasks, scenes, selected_runtime_mode=args.runtime_mode)
        print(json.dumps({"runtime_mode": args.runtime_mode, "train_smoke": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
