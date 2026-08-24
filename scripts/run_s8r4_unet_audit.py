"""S8-R4 single-seed Conditional 1-D U-Net architecture audit.

Only the denoiser backbone changes relative to the frozen S8-R3 contract.
This script never loads TEST, never runs PPO, and trains one continuous U-Net
trajectory for 30 effective passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from quadrotor_diffusion_ppo.paths import configure_external_imports
configure_external_imports()

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from quadrotor_diffusion_ppo.diffusion.model import unit_ball_squash
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from quadrotor_diffusion_ppo.diffusion.unet1d import ConditionalUnet1D
from scripts.run_s8r3_training_sufficiency import (
    DATA_ROOT, OUT as S8R3_OUT, CKPT_ROOT as S8R3_CKPT, HORIZON, ACTION_DIM,
    OBSERVATION_DIM, DIFFUSION_STEPS, DDIM_STEPS, DDIM_ETA, BATCH_SIZE,
    MAX_PASSES, CHECKPOINT_PASSES, FAMILIES, build_windows, read_val_tasks,
    generate_map, make_scene, parse_vec, stable_seed, SPEED_LIMIT, RAY_RANGE,
    MAX_EPISODE_STEPS, rollout_checkpoint, summarize_rows, sha256_file,
)
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint

TASK = "S8-R4-CONDITIONAL-1D-UNET-DIFFUSION-ARCHITECTURE-AUDIT-V1"
START_HEAD = "d7cfd36e8e508f7c68bfd463d203be6e10391d18"
TRAIN_SHA = "395a5c2ab1eb6cb3f3925fa3ecbe160d151833ef95cc5bc3b9187fbc497de3c5"
VAL_SHA = "0fc50886504d36890444b4c09c2990962cc5be6551cb0e54c008df0d26f5246e"
NORMALIZATION_SHA = "e05b9a424cbb782dc1eddfcfd69fbf4e849806d3995bbab48e4a60193c76b6e0"
TRAINING_SEED = 20260820
# Effective batch is frozen at 512.  This RTX 5060 Ti fits the full batch, so
# no microbatch fallback is needed; keeping accumulation=1 is the same
# optimization semantics and avoids the severe small-batch kernel penalty.
MICROBATCH = 512
ACCUMULATION = 1
OUT = ROOT / "artifacts" / "s8r4"
CKPT_ROOT = ROOT / "checkpoints" / "s8r4" / "unet"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def set_deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def validate_identity() -> dict:
    # TEST is deliberately not opened or hashed.  TRAIN/VAL identity is enough
    # to establish the permitted S8-R2 data source without touching TEST.
    train_path, val_path = DATA_ROOT / "train.npz", DATA_ROOT / "val.npz"
    train_hash, val_hash = sha256_file(train_path), sha256_file(val_path)
    if train_hash != TRAIN_SHA or val_hash != VAL_SHA:
        raise RuntimeError(f"BLOCKED_S8R4_DATASET_HASH: train={train_hash} val={val_hash}")
    with np.load(train_path, allow_pickle=False) as train, np.load(val_path, allow_pickle=False) as val:
        if train["observations"].shape[1:] != (34,) or train["actions"].shape[1:] != (3,):
            raise RuntimeError("BLOCKED_S8R4_TRAIN_SHAPE")
        if val["observations"].shape[1:] != (34,) or val["actions"].shape[1:] != (3,):
            raise RuntimeError("BLOCKED_S8R4_VAL_SHAPE")
        train_episodes, val_episodes = len(train["episode_lengths"]), len(val["episode_lengths"])
        train_windows = int(np.maximum(0, train["episode_lengths"].astype(np.int64) - 16 + 1).sum())
        val_windows = int(np.maximum(0, val["episode_lengths"].astype(np.int64) - 16 + 1).sum())
        return {"train_sha256": train_hash, "val_sha256": val_hash, "train_episodes": train_episodes,
                "val_episodes": val_episodes, "train_windows": train_windows, "val_windows": val_windows,
                "test_accessed": False, "test_values_loaded": False}


def load_norm() -> tuple[np.ndarray, np.ndarray]:
    path = S8R3_OUT / "train_obs_normalization.npz"
    if sha256_file(path) != NORMALIZATION_SHA:
        raise RuntimeError("BLOCKED_S8R4_NORMALIZATION_IDENTITY")
    with np.load(path, allow_pickle=False) as n:
        return n["mean"].astype(np.float32), n["std"].astype(np.float32)


def diffusion_val_loss(model: ConditionalUnet1D, schedule: DiffusionSchedule, obs: torch.Tensor,
                       actions: torch.Tensor, device: torch.device) -> float:
    model.eval(); total = 0.0; generator = torch.Generator(device=device).manual_seed(TRAINING_SEED + 991)
    with torch.no_grad():
        for start in range(0, len(obs), BATCH_SIZE):
            target = actions[start:start + BATCH_SIZE]
            ts = torch.randint(0, DIFFUSION_STEPS, (len(target),), generator=generator, device=device)
            noise = torch.randn(target.shape, generator=generator, device=device)
            pred = model(schedule.add_noise(target, noise, ts), obs[start:start + BATCH_SIZE], ts)
            total += float(torch.nn.functional.mse_loss(pred, target).item()) * len(target)
    return total / max(1, len(obs))


def save_checkpoint(model: torch.nn.Module, path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "model_state": model.state_dict(), "model_frozen": True}, path)
    return sha256_file(path)


def train_unet(train_obs: torch.Tensor, train_actions: torch.Tensor, val_obs: torch.Tensor,
               val_actions: torch.Tensor, mean: np.ndarray, std: np.ndarray,
               updates_per_pass: int, device: torch.device, config: dict) -> tuple[list[dict], dict]:
    model = ConditionalUnet1D(**config).to(device)
    params = sum(p.numel() for p in model.parameters())
    schedule = DiffusionSchedule(DIFFUSION_STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)
    total_updates = updates_per_pass * MAX_PASSES
    curve: list[dict] = []; checkpoints: dict[str, dict] = {}
    next_event = max(1, math.ceil(updates_per_pass / 2)); started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for update in range(1, total_updates + 1):
        model.train(); indices = torch.randint(0, len(train_obs), (BATCH_SIZE,), generator=generator, device=device)
        target = train_actions[indices]; obs = train_obs[indices]
        loss_value = 0.0
        for mb_start in range(0, BATCH_SIZE, MICROBATCH):
            mb_target = target[mb_start:mb_start + MICROBATCH]; mb_obs = obs[mb_start:mb_start + MICROBATCH]
            ts = torch.randint(0, DIFFUSION_STEPS, (len(mb_target),), generator=generator, device=device)
            noise = torch.randn(mb_target.shape, generator=generator, device=device)
            pred = model(schedule.add_noise(mb_target, noise, ts), mb_obs, ts)
            loss = torch.nn.functional.mse_loss(pred, mb_target) / ACCUMULATION
            if not torch.isfinite(loss): raise RuntimeError("BLOCKED_S8R4_NONFINITE_LOSS")
            loss.backward(); loss_value += float(loss.item()) * ACCUMULATION
        grad = float(clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(grad) or not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("BLOCKED_S8R4_NONFINITE_GRAD_OR_WEIGHT")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if update >= next_event or update == total_updates:
            pass_value = min(MAX_PASSES, update / updates_per_pass)
            val_loss = diffusion_val_loss(model, schedule, val_obs, val_actions, device)
            row = {"model": "UNET", "update": update, "effective_pass": pass_value,
                   "train_loss": loss_value / ACCUMULATION, "val_offline_loss": val_loss,
                   "gradient_norm": grad, "learning_rate": 3e-4,
                   "wall_time_s": time.perf_counter() - started,
                   "gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                   "microbatch": MICROBATCH, "gradient_accumulation": ACCUMULATION,
                   "effective_batch": BATCH_SIZE}
            curve.append(row); next_event += max(1, math.ceil(updates_per_pass / 2))
        for p in CHECKPOINT_PASSES:
            if update == p * updates_per_pass:
                payload = {"task": TASK, "model_config": config, "architecture": "ConditionalUnet1D 256-512-1024",
                           "parameter_count": params, "observation_mean": mean, "observation_std": std,
                           "normalization_sha256": NORMALIZATION_SHA, "train_seed": TRAINING_SEED,
                           "effective_pass": p, "updates": update, "horizon": HORIZON, "action_dim": ACTION_DIM,
                           "action_support": "unit_ball_squash", "prediction_target": "x0",
                           "diffusion_steps": DIFFUSION_STEPS, "ddim_steps": DDIM_STEPS, "ddim_eta": DDIM_ETA,
                           "beta_schedule": "cosine", "optimizer": "AdamW", "learning_rate": 3e-4,
                           "weight_decay": 1e-4, "batch_size": BATCH_SIZE, "gradient_clip": 1.0,
                           "microbatch": MICROBATCH, "gradient_accumulation": ACCUMULATION,
                           "runtime": "UNET_RUNTIME=EAGER_EXACT", "ema": False}
                path = CKPT_ROOT / f"pass_{p:02d}.pt"; checkpoints[str(p)] = {"path": str(path), "sha256": save_checkpoint(model, path, payload)}
        if update % updates_per_pass == 0: print(f"UNET pass={update // updates_per_pass}/{MAX_PASSES} loss={loss_value / ACCUMULATION:.6f}", flush=True)
    return curve, {"parameter_count": params, "checkpoints": checkpoints,
                   "training_wall_time_s": time.perf_counter() - started, "all_finite": True,
                   "microbatch": MICROBATCH, "gradient_accumulation": ACCUMULATION}


class FrozenUnetSampler:
    def __init__(self, checkpoint: Path, device: torch.device):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        required = {"prediction_target": "x0", "action_support": "unit_ball_squash", "diffusion_steps": 100,
                    "ddim_steps": 10, "ddim_eta": 0.0, "model_frozen": True, "runtime": "UNET_RUNTIME=EAGER_EXACT"}
        if any(payload.get(k) != v for k, v in required.items()): raise RuntimeError("BLOCKED_S8R4_CONFIG_IDENTITY")
        self.model = ConditionalUnet1D(**payload["model_config"]).to(device)
        self.model.load_state_dict(payload["model_state"]); self.model.eval().requires_grad_(False)
        self.schedule = DiffusionSchedule(100).to(device); self.mean = torch.as_tensor(payload["observation_mean"], device=device)
        self.std = torch.clamp(torch.as_tensor(payload["observation_std"], device=device), min=1e-6); self.device = device
        self.timesteps = tuple(int(v) for v in torch.linspace(99, 0, 10, device=device).round().long().cpu().tolist())

    @torch.inference_mode()
    def predict(self, observation: np.ndarray, seed: int) -> np.ndarray:
        obs = (torch.as_tensor(observation, dtype=torch.float32, device=self.device) - self.mean) / self.std
        gen = torch.Generator(device=self.device).manual_seed(int(seed)); sample = torch.randn((1, 16, 3), generator=gen, device=self.device)
        for i, t in enumerate(self.timesteps):
            tb = torch.full((1,), t, device=self.device, dtype=torch.long)
            x0 = self.model(sample, obs.reshape(1, 34), tb)
            alpha = self.schedule.alpha_bars[t]; eps = (sample - alpha.sqrt() * x0) / (1.0 - alpha).sqrt()
            if i == len(self.timesteps) - 1: sample = x0
            else:
                an = self.schedule.alpha_bars[self.timesteps[i + 1]]; sample = an.sqrt() * x0 + (1.0 - an).sqrt() * eps
        return sample[0, 0].cpu().numpy().astype(np.float32)


def rollout_unet(checkpoint: Path, mean: np.ndarray, std: np.ndarray, tasks: list[dict], device: torch.device) -> list[dict]:
    sampler = FrozenUnetSampler(checkpoint, device); rows: list[dict] = []
    for idx, item in enumerate(tasks):
        family = item["family"]; map_idx = int(item["map_id"].rsplit("_", 1)[1]); family_idx = FAMILIES.index(family)
        scene = make_scene(generate_map(family_idx, map_idx), parse_vec(item["start"]), parse_vec(item["goal"]), int(item["task_slot"]))
        task = TaskEndpoint(scene.scene_id, item["task_id"], stable_seed(item["task_id"]), scene.start, scene.goal, "VAL")
        env = PurePPONavigationEnv(scene, task, speed_limit=SPEED_LIMIT, ray_range=RAY_RANGE, gui=False)
        observation, _ = env.reset(seed=stable_seed(item["task_id"])); total_return = 0.0; info = {}; finite = True; started = time.perf_counter()
        try:
            for _ in range(MAX_EPISODE_STEPS):
                if not np.isfinite(observation).all(): finite = False; break
                action = sampler.predict(observation, stable_seed(item["task_id"]))
                if not np.isfinite(action).all(): finite = False; break
                observation, reward, terminated, truncated, info = env.step(action); total_return += reward
                if terminated or truncated: break
        except Exception as exc:
            info = {"error": str(exc), "nonfinite": True}; finite = False
        finally: env.close()
        collision = bool(info.get("collision", False)); ground = bool(info.get("ground_contact", False))
        rows.append({"model": "UNET", "checkpoint": checkpoint.stem, "task_id": item["task_id"], "family": family,
                     "success": bool(info.get("success", False)) and finite, "collision": collision, "ground": ground,
                     "unsafe": collision or ground, "timeout": bool(info.get("timeout", False)),
                     "nonfinite": bool(info.get("nonfinite", False) or not finite), "return": float(total_return),
                     "steps": int(info.get("episode_steps", 0)), "action_clip_count": int(info.get("action_clip_count", 0)),
                     "wall_time_s": time.perf_counter() - started})
        if (idx + 1) % 100 == 0: print(f"VAL UNET {checkpoint.stem}: {idx + 1}/{len(tasks)}", flush=True)
    return rows


def nearest_pass_rows(curve_rows: list[dict], model: str) -> dict[int, dict]:
    rows = [r for r in curve_rows if r["model"] == model]
    for r in rows: r["effective_pass"] = float(r["effective_pass"]); r["val_offline_loss"] = float(r["val_offline_loss"])
    return {p: min(rows, key=lambda r: abs(r["effective_pass"] - p)) for p in CHECKPOINT_PASSES}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    if os.environ.get("S8R4_ALLOW_TEST"): raise RuntimeError("BLOCKED_S8R4_TEST_UNLOCK_ATTEMPT")
    if not torch.cuda.is_available(): raise RuntimeError("BLOCKED_S8R4_CUDA_UNAVAILABLE")
    set_deterministic(TRAINING_SEED); device = torch.device("cuda"); identity = validate_identity(); mean, std = load_norm()
    config = {"observation_dim": 34, "horizon": 16, "action_dim": 3, "condition_dim": 256, "bounded_output": True}
    OUT.mkdir(parents=True, exist_ok=True); write_json(OUT / "unet_config.json", {"task": TASK, "model": config,
        "channels": [256, 512, 1024], "global_condition_dim": 512, "groups": 8, "kernel_size": 5,
        "normalization_sha256": NORMALIZATION_SHA, "training_seed": TRAINING_SEED, "runtime": "UNET_RUNTIME=EAGER_EXACT", "ema": False})
    cfg_sha = sha256_file(OUT / "unet_config.json"); model = ConditionalUnet1D(**config); params = sum(p.numel() for p in model.parameters())
    arch = {"model": "ConditionalUnet1D", "parameter_count": params, "mlp_parameter_count": 621360,
            "bc_parameter_count": 592688, "architecture_capacity_audit": True, "config_sha256": cfg_sha,
            "input_shape": ["B", 16, 3], "output_shape": ["B", 16, 3], "condition_shape": ["B", 512], "test_accessed": False}
    write_json(OUT / "architecture_summary.json", arch)
    if args.smoke:
        m = ConditionalUnet1D(**config).to(device)
        x = torch.randn((MICROBATCH, 16, 3), device=device)
        o = torch.randn((MICROBATCH, 34), device=device)
        t = torch.zeros(MICROBATCH, dtype=torch.long, device=device)
        y = m(x, o, t); y.mean().backward()
        print(json.dumps({"parameter_count": params, "smoke": "PASS"})); return
    train, val = (np.load(DATA_ROOT / f"{split}.npz", allow_pickle=False) for split in ("train", "val"))
    train_data = {k: train[k] for k in train.files}; val_data = {k: val[k] for k in val.files}; train.close(); val.close()
    train_obs_np, train_actions_np = build_windows(train_data); val_obs_np, val_actions_np = build_windows(val_data)
    # Normalize in place so the large 4.7M-window array is not duplicated.
    train_obs_np -= mean; train_obs_np /= std; val_obs_np -= mean; val_obs_np /= std
    train_obs = torch.from_numpy(train_obs_np).to(device); val_obs = torch.from_numpy(val_obs_np).to(device)
    train_actions = torch.from_numpy(train_actions_np).to(device); val_actions = torch.from_numpy(val_actions_np).to(device)
    updates = math.ceil(identity["train_windows"] / BATCH_SIZE)
    if updates != 9329 or identity["train_windows"] != 4776168 or identity["val_windows"] != 598153: raise RuntimeError("BLOCKED_S8R4_WINDOW_IDENTITY")
    # Keep the single continuous trajectory in one process.  No BC/MLP retraining.
    curve, train_summary = train_unet(train_obs, train_actions, val_obs, val_actions, mean, std, updates, device, config)
    # Include the frozen S8-R3 MLP reference rows in the combined audit curve;
    # these rows are copied evidence, never retrained here.
    mlp_curve = list(csv.DictReader((S8R3_OUT / "training_curve.csv").open(encoding="utf-8", newline="")))
    write_csv(OUT / "training_curve.csv", curve + [r for r in mlp_curve if r["model"] == "DIFFUSION"])
    curves = mlp_curve
    mlp_rows = nearest_pass_rows(curves, "DIFFUSION"); unet_rows = nearest_pass_rows(curve, "UNET")
    mlp_selected = min(mlp_rows, key=lambda p: mlp_rows[p]["val_offline_loss"]); unet_selected = min(unet_rows, key=lambda p: unet_rows[p]["val_offline_loss"])
    selection = {"selection_metric": "VAL offline diffusion loss", "candidate_passes": list(CHECKPOINT_PASSES),
                 "mlp": {"selected_pass": mlp_selected, "loss": mlp_rows[mlp_selected]["val_offline_loss"], "source_effective_pass": mlp_rows[mlp_selected]["effective_pass"]},
                 "unet": {"selected_pass": unet_selected, "loss": unet_rows[unet_selected]["val_offline_loss"], "source_effective_pass": unet_rows[unet_selected]["effective_pass"]},
                 "frozen_before_closed_loop": True, "selected_by_closed_loop_success": False}
    selection["sha256"] = canonical_sha(selection); write_json(OUT / "checkpoint_selection.json", selection)
    # Closed-loop is run by the task-sharded evaluator after this training script.
    write_json(OUT / "summary.json", {"task": TASK, "status": "TRAINING_COMPLETE_AWAITING_VAL", "start_head": START_HEAD,
        "dataset": identity, "normalization_sha256": NORMALIZATION_SHA, "training_seed": TRAINING_SEED,
        "effective_passes": MAX_PASSES, "updates_per_pass": updates, "unet": train_summary,
        "unet_parameter_count": params, "unet_config_sha256": cfg_sha, "checkpoint_selection": selection,
        "test_accessed": False, "test_run_count": 0, "ppo_run_count": 0})
    print(json.dumps({"status": "TRAINING_COMPLETE_AWAITING_VAL", "parameter_count": params, "selection": selection}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
