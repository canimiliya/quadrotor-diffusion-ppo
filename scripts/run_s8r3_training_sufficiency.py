"""S8-R3 training-sufficiency audit on the frozen S8-R2 10K dataset.

This script intentionally owns the complete diagnostic: exact H=16 window
counting, TRAIN-only normalization, continuous 30-effective-pass training for
matched BC and the existing ConditionalDiffusionMLP, frozen-checkpoint VAL
rollouts, and evidence tables.  TEST is never loaded or opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from quadrotor_diffusion_ppo.paths import configure_external_imports
configure_external_imports()

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from quadrotor_diffusion_ppo.bc.model import MatchedSequenceBC
from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from scripts.generate_s8r2_10k_dataset import FAMILIES, generate_map, make_scene


TASK = "S8-R3-10K-BC-VS-MLP-DIFFUSION-TRAINING-SUFFICIENCY-AUDIT-V1"
START_HEAD = "a8805cf62258a86948a3496267aaebbfad486425"
TRAIN_SHA = "395a5c2ab1eb6cb3f3925fa3ecbe160d151833ef95cc5bc3b9187fbc497de3c5"
VAL_SHA = "0fc50886504d36890444b4c09c2990962cc5be6551cb0e54c008df0d26f5246e"
TEST_SHA = "b4eee85496acb4d1e7b02e4f4bb44715c18c355014e77a79f8499a5fffa3a74e"
TRAINING_SEED = 20260820
BATCH_SIZE = 512
MAX_PASSES = 30
HORIZON = 16
ACTION_DIM = 3
OBSERVATION_DIM = 34
DIFFUSION_STEPS = 100
DDIM_STEPS = 10
DDIM_ETA = 0.0
SPEED_LIMIT = 0.801
RAY_RANGE = 3.0
MAX_EPISODE_STEPS = 960
GOAL_TOLERANCE = 0.30
EVAL_BASE_SEED = 20260811
CHECKPOINT_PASSES = (1, 5, 10, 20, 30)
MODEL_CONFIG_BC = {
    "observation_dim": 34, "horizon": 16, "action_dim": 3,
    "observation_width": 128, "width": 256, "residual_blocks": 4,
}
MODEL_CONFIG_DIFF = {**MODEL_CONFIG_BC, "time_dim": 64, "bounded_output": True}
OPTIMIZER_CONFIG = {
    "loss": "SmoothL1",
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": BATCH_SIZE,
    "gradient_clip": 1.0,
}
DIFFUSION_OPTIMIZER_CONFIG = {
    **OPTIMIZER_CONFIG,
    "loss": "MSE_x0",
    "prediction_target": "x0",
    "diffusion_steps": DIFFUSION_STEPS,
    "beta_schedule": "cosine",
    "ddim_steps": DDIM_STEPS,
    "ddim_eta": DDIM_ETA,
    "action_support": "unit_ball_squash",
}
DATA_ROOT = ROOT / "artifacts" / "s8r2_10k"
OUT = ROOT / "artifacts" / "s8r3"
CKPT_ROOT = ROOT / "checkpoints" / "s8r3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def window_count(data: dict[str, np.ndarray]) -> int:
    lengths = np.asarray(data["episode_lengths"], dtype=np.int64)
    return int(np.maximum(0, lengths - HORIZON + 1).sum())


def build_windows(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    observations = data["observations"]
    actions = data["actions"]
    obs_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for offset, length in zip(data["episode_offsets"], data["episode_lengths"]):
        offset, length = int(offset), int(length)
        n = max(0, length - HORIZON + 1)
        if n:
            obs_parts.append(observations[offset:offset + n])
            # sliding_window_view preserves episode boundaries and ordering,
            # while avoiding millions of Python-level slice/stack operations.
            view = np.lib.stride_tricks.sliding_window_view(
                actions[offset:offset + length], HORIZON, axis=0
            )
            action_parts.append(np.moveaxis(view, -1, 1))
    if not obs_parts:
        raise RuntimeError("no H16 windows")
    return np.concatenate(obs_parts).astype(np.float32, copy=False), np.concatenate(action_parts).astype(np.float32, copy=False)


def validate_dataset_identity() -> dict:
    train_path, val_path, test_path = (DATA_ROOT / f"{s}.npz" for s in ("train", "val", "test"))
    hashes = {"train": sha256_file(train_path), "val": sha256_file(val_path), "test": sha256_file(test_path)}
    if hashes != {"train": TRAIN_SHA, "val": VAL_SHA, "test": TEST_SHA}:
        raise RuntimeError(f"BLOCKED_S8R3_DATASET_HASH:{hashes}")
    train, val = load_npz(train_path), load_npz(val_path)
    # TEST is hashed only; its values are deliberately not loaded.
    if any(data["observations"].shape[1:] != (34,) or data["actions"].shape[1:] != (3,) for data in (train, val)):
        raise RuntimeError("BLOCKED_S8R3_CONTRACT_SHAPE")
    return {
        "train_sha256": hashes["train"], "val_sha256": hashes["val"], "test_sha256": hashes["test"],
        "test_accessed": False, "test_values_loaded": False,
        "train_episodes": int(len(train["episode_lengths"])), "val_episodes": int(len(val["episode_lengths"])),
        "train_transitions": int(len(train["actions"])), "val_transitions": int(len(val["actions"])),
        "train_windows": window_count(train), "val_windows": window_count(val),
    }


def train_normalization(train: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mean = train["observations"].astype(np.float64).mean(axis=0).astype(np.float32)
    std = train["observations"].astype(np.float64).std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "train_obs_normalization.npz", mean=mean, std=std,
                        source_split=np.asarray(["TRAIN"], dtype="U5"), observation_dim=np.asarray([34], dtype=np.int32))
    return mean, std


def normalize_to_device(values: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(((values - mean) / std).astype(np.float32, copy=False)).to(device)


def smooth_l1_validation(model, obs: torch.Tensor, actions: torch.Tensor) -> float:
    model.eval(); total = 0.0
    with torch.no_grad():
        for start in range(0, len(obs), BATCH_SIZE):
            pred = model(obs[start:start + BATCH_SIZE])
            value = torch.nn.functional.smooth_l1_loss(pred, actions[start:start + BATCH_SIZE])
            total += float(value.item()) * len(pred)
    return total / max(1, len(obs))


def diffusion_validation(model, schedule, obs: torch.Tensor, actions: torch.Tensor) -> float:
    model.eval(); total = 0.0
    generator = torch.Generator(device=obs.device).manual_seed(TRAINING_SEED + 991)
    with torch.no_grad():
        for start in range(0, len(obs), BATCH_SIZE):
            target = actions[start:start + BATCH_SIZE]
            ts = torch.randint(0, DIFFUSION_STEPS, (len(target),), generator=generator, device=obs.device)
            noise = torch.randn(target.shape, generator=generator, device=obs.device)
            noisy = schedule.add_noise(target, noise, ts)
            pred = model(noisy, obs[start:start + BATCH_SIZE], ts)
            value = torch.nn.functional.mse_loss(pred, target)
            total += float(value.item()) * len(target)
    return total / max(1, len(obs))


def save_model_checkpoint(model: torch.nn.Module, path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "model_state": model.state_dict(), "model_frozen": True}, path)
    return sha256_file(path)


def train_bc(train_obs: torch.Tensor, train_actions: torch.Tensor, val_obs: torch.Tensor,
             val_actions: torch.Tensor, mean: np.ndarray, std: np.ndarray, updates_per_pass: int,
             norm_sha: str) -> tuple[list[dict], dict]:
    model = MatchedSequenceBC(**MODEL_CONFIG_BC).to(train_obs.device)
    params = sum(p.numel() for p in model.parameters())
    if params != 592688:
        raise RuntimeError(f"BLOCKED_S8R3_BC_ARCHITECTURE:{params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=train_obs.device).manual_seed(TRAINING_SEED)
    curve: list[dict] = []
    checkpoints: dict[str, dict] = {}
    next_event = max(1, math.ceil(updates_per_pass / 2))
    started = time.perf_counter()
    total_updates = updates_per_pass * MAX_PASSES
    for update in range(1, total_updates + 1):
        model.train()
        indices = torch.randint(0, len(train_obs), (BATCH_SIZE,), generator=generator, device=train_obs.device)
        pred = model(train_obs[indices])
        loss = torch.nn.functional.smooth_l1_loss(pred, train_actions[indices])
        if not torch.isfinite(loss):
            raise RuntimeError("BLOCKED_S8R3_BC_NONFINITE_LOSS")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = float(clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(grad) or not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("BLOCKED_S8R3_BC_NONFINITE_GRAD_OR_WEIGHT")
        optimizer.step()
        if update >= next_event or update == total_updates:
            pass_value = min(MAX_PASSES, update / updates_per_pass)
            row = {"model": "BC", "update": update, "effective_pass": pass_value,
                   "train_loss": float(loss.item()), "val_offline_loss": smooth_l1_validation(model, val_obs, val_actions),
                   "gradient_norm": grad, "learning_rate": 3e-4,
                   "wall_time_s": time.perf_counter() - started,
                   "gpu_memory_bytes": int(torch.cuda.max_memory_allocated(train_obs.device))}
            if not math.isfinite(row["val_offline_loss"]):
                raise RuntimeError("BLOCKED_S8R3_BC_NONFINITE_VAL_LOSS")
            curve.append(row)
            next_event += max(1, math.ceil(updates_per_pass / 2))
        for p in CHECKPOINT_PASSES:
            target_update = p * updates_per_pass
            if update == target_update:
                payload = {"task": TASK, "model_config": MODEL_CONFIG_BC, "architecture": "34->128->128->256x4 residual->8",
                           "parameter_count": params, "observation_mean": mean, "observation_std": std,
                           "normalization_sha256": norm_sha, "train_seed": TRAINING_SEED,
                           "effective_pass": p, "updates": update, "horizon": 16, "action_dim": 3,
                           "action_support": "per_step_unit_ball_squash", "deployment": "receding_horizon_first_action",
                           **OPTIMIZER_CONFIG}
                path = CKPT_ROOT / "bc" / f"pass_{p:02d}.pt"
                checkpoints[str(p)] = {"path": str(path), "sha256": save_model_checkpoint(model, path, payload)}
        if update % updates_per_pass == 0:
            print(f"BC pass={update // updates_per_pass}/{MAX_PASSES} loss={loss.item():.6f}", flush=True)
    return curve, {"model": "BC", "parameter_count": params, "checkpoints": checkpoints,
                   "training_wall_time_s": time.perf_counter() - started, "all_finite": True}


def train_diffusion(train_obs: torch.Tensor, train_actions: torch.Tensor, val_obs: torch.Tensor,
                    val_actions: torch.Tensor, mean: np.ndarray, std: np.ndarray, updates_per_pass: int,
                    norm_sha: str) -> tuple[list[dict], dict]:
    model = ConditionalDiffusionMLP(**MODEL_CONFIG_DIFF).to(train_obs.device)
    params = sum(p.numel() for p in model.parameters())
    if params != 621360:
        raise RuntimeError(f"BLOCKED_S8R3_DIFFUSION_ARCHITECTURE:{params}")
    schedule = DiffusionSchedule(DIFFUSION_STEPS).to(train_obs.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=train_obs.device).manual_seed(TRAINING_SEED)
    curve: list[dict] = []; checkpoints: dict[str, dict] = {}
    next_event = max(1, math.ceil(updates_per_pass / 2))
    started = time.perf_counter(); total_updates = updates_per_pass * MAX_PASSES
    for update in range(1, total_updates + 1):
        model.train()
        indices = torch.randint(0, len(train_obs), (BATCH_SIZE,), generator=generator, device=train_obs.device)
        target = train_actions[indices]
        ts = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), generator=generator, device=train_obs.device)
        noise = torch.randn(target.shape, generator=generator, device=train_obs.device)
        noisy = schedule.add_noise(target, noise, ts)
        pred = model(noisy, train_obs[indices], ts)
        loss = torch.nn.functional.mse_loss(pred, target)
        if not torch.isfinite(loss):
            raise RuntimeError("BLOCKED_S8R3_DIFFUSION_NONFINITE_LOSS")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = float(clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(grad) or not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("BLOCKED_S8R3_DIFFUSION_NONFINITE_GRAD_OR_WEIGHT")
        optimizer.step()
        if update >= next_event or update == total_updates:
            pass_value = min(MAX_PASSES, update / updates_per_pass)
            row = {"model": "DIFFUSION", "update": update, "effective_pass": pass_value,
                   "train_loss": float(loss.item()), "val_offline_loss": diffusion_validation(model, schedule, val_obs, val_actions),
                   "gradient_norm": grad, "learning_rate": 3e-4,
                   "wall_time_s": time.perf_counter() - started,
                   "gpu_memory_bytes": int(torch.cuda.max_memory_allocated(train_obs.device))}
            if not math.isfinite(row["val_offline_loss"]):
                raise RuntimeError("BLOCKED_S8R3_DIFFUSION_NONFINITE_VAL_LOSS")
            curve.append(row)
            next_event += max(1, math.ceil(updates_per_pass / 2))
        for p in CHECKPOINT_PASSES:
            target_update = p * updates_per_pass
            if update == target_update:
                payload = {"task": TASK, "model_config": MODEL_CONFIG_DIFF, "architecture": "34->128->128 + time64 + 48 -> 256x4 residual -> 48",
                           "parameter_count": params, "observation_mean": mean, "observation_std": std,
                           "normalization_sha256": norm_sha, "train_seed": TRAINING_SEED,
                           "effective_pass": p, "updates": update, "horizon": 16, "action_dim": 3,
                           **DIFFUSION_OPTIMIZER_CONFIG}
                path = CKPT_ROOT / "diffusion" / f"pass_{p:02d}.pt"
                checkpoints[str(p)] = {"path": str(path), "sha256": save_model_checkpoint(model, path, payload)}
        if update % updates_per_pass == 0:
            print(f"DIFFUSION pass={update // updates_per_pass}/{MAX_PASSES} loss={loss.item():.6f}", flush=True)
    return curve, {"model": "DIFFUSION", "parameter_count": params, "checkpoints": checkpoints,
                   "training_wall_time_s": time.perf_counter() - started, "all_finite": True}


def stable_seed(task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int((EVAL_BASE_SEED + int.from_bytes(digest[:8], "little")) % (2**63 - 1))


def read_val_tasks() -> list[dict]:
    rows = []
    with (DATA_ROOT / "task_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val" or row["accepted"].lower() != "true":
                continue
            rows.append(row)
    if len(rows) != 1000:
        raise RuntimeError(f"BLOCKED_S8R3_VAL_TASK_COUNT:{len(rows)}")
    return sorted(rows, key=lambda row: row["task_id"])


def parse_vec(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


class FrozenDiffusionSampler:
    def __init__(self, path: Path, device: torch.device):
        payload = torch.load(path, map_location=device, weights_only=False)
        required = {"prediction_target": "x0", "action_support": "unit_ball_squash", "diffusion_steps": 100,
                    "ddim_steps": 10, "ddim_eta": 0.0, "model_frozen": True}
        if any(payload.get(k) != v for k, v in required.items()):
            raise RuntimeError("BLOCKED_S8R3_DIFFUSION_CONFIG_IDENTITY")
        self.model = ConditionalDiffusionMLP(**payload["model_config"]).to(device)
        self.model.load_state_dict(payload["model_state"]); self.model.eval().requires_grad_(False)
        self.schedule = DiffusionSchedule(100).to(device)
        self.mean = torch.as_tensor(payload["observation_mean"], device=device)
        self.std = torch.clamp(torch.as_tensor(payload["observation_std"], device=device), min=1e-6)
        self.device = device
        self.timesteps = tuple(int(v) for v in torch.linspace(99, 0, 10, device=device).round().long().cpu().tolist())
        self.tbatch = tuple(torch.full((1,), t, device=device, dtype=torch.long) for t in self.timesteps)
        self.alphas = tuple(self.schedule.alpha_bars[t] for t in self.timesteps)
        self.graph = None; self.gcond = self.ginit = self.gout = None

    @torch.inference_mode()
    def _sample(self, condition: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        sample = initial
        for i, (tb, alpha) in enumerate(zip(self.tbatch, self.alphas)):
            x0 = self.model.forward(sample, condition, tb)  # bounded x0 contract
            eps = (sample - alpha.sqrt() * x0) / (1.0 - alpha).sqrt()
            if i == len(self.timesteps) - 1:
                sample = x0
            else:
                an = self.alphas[i + 1]
                sample = an.sqrt() * x0 + (1.0 - an).sqrt() * eps
        return sample

    @torch.inference_mode()
    def _capture(self) -> None:
        self.gcond = torch.empty((1, 34), dtype=torch.float32, device=self.device)
        self.ginit = torch.empty((1, 16, 3), dtype=torch.float32, device=self.device)
        stream = torch.cuda.Stream(device=self.device); stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3): self._sample(self.gcond, self.ginit)
        torch.cuda.current_stream(self.device).wait_stream(stream); torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph): self.gout = self._sample(self.gcond, self.ginit)

    @torch.inference_mode()
    def predict(self, observation: np.ndarray, seed: int) -> np.ndarray:
        cond = (torch.as_tensor(observation, dtype=torch.float32, device=self.device) - self.mean) / self.std
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        initial = torch.randn((1, 16, 3), generator=gen, device=self.device)
        if self.device.type == "cuda":
            if self.graph is None: self._capture()
            self.gcond.copy_(cond.reshape(1, 34)); self.ginit.copy_(initial); self.graph.replay()
            return self.gout[0, 0].clone().cpu().numpy().astype(np.float32)
        return self._sample(cond.reshape(1, 34), initial)[0, 0].cpu().numpy().astype(np.float32)


def summarize_rows(rows: list[dict], *, include_family: bool = True) -> dict:
    total = max(1, len(rows))
    result = {"count": len(rows), "success": sum(int(r["success"]) for r in rows),
              "success_rate": sum(int(r["success"]) for r in rows) / total,
              "collision": sum(int(r["collision"]) for r in rows),
              "ground": sum(int(r["ground"]) for r in rows),
              "unsafe": sum(int(r["unsafe"]) for r in rows),
              "timeout": sum(int(r["timeout"]) for r in rows),
              "nonfinite": sum(int(r["nonfinite"]) for r in rows),
              "mean_return": float(np.mean([r["return"] for r in rows])) if rows else 0.0,
              "mean_steps": float(np.mean([r["steps"] for r in rows])) if rows else 0.0}
    if include_family:
        result["by_family"] = {}
        for family in FAMILIES:
            result["by_family"][family] = summarize_rows(
                [r for r in rows if r["family"] == family], include_family=False
            ) if any(r["family"] == family for r in rows) else {"count": 0}
    return result


def rollout_checkpoint(kind: str, checkpoint: Path, mean: np.ndarray, std: np.ndarray,
                       val_tasks: list[dict], device: torch.device) -> list[dict]:
    if kind == "BC":
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = MatchedSequenceBC(**payload["model_config"]).to(device); model.load_state_dict(payload["model_state"])
        model.eval().requires_grad_(False)
        def predict(obs, seed):
            del seed
            with torch.inference_mode():
                x = (torch.as_tensor(obs, dtype=torch.float32, device=device) - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)
                return model(x.reshape(1, 34))[0, 0].cpu().numpy().astype(np.float32)
    else:
        model = FrozenDiffusionSampler(checkpoint, device)
        predict = model.predict
    rows: list[dict] = []
    for idx, item in enumerate(val_tasks):
        family = item["family"]; map_idx = int(item["map_id"].rsplit("_", 1)[1]); family_idx = FAMILIES.index(family)
        scene = make_scene(generate_map(family_idx, map_idx), parse_vec(item["start"]), parse_vec(item["goal"]), int(item["task_slot"]))
        task = TaskEndpoint(scene.scene_id, item["task_id"], stable_seed(item["task_id"]), scene.start, scene.goal, "VAL")
        env = PurePPONavigationEnv(scene, task, speed_limit=SPEED_LIMIT, ray_range=RAY_RANGE, gui=False)
        observation, _ = env.reset(seed=stable_seed(item["task_id"]))
        total_return = 0.0; info: dict = {}; finite = True
        started = time.perf_counter()
        try:
            for step in range(MAX_EPISODE_STEPS):
                if not np.isfinite(observation).all(): finite = False; break
                action = predict(observation, stable_seed(item["task_id"]) + step)
                if not np.isfinite(action).all(): finite = False; break
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += reward
                if terminated or truncated: break
        except Exception as exc:
            info = {"error": str(exc), "nonfinite": True}; finite = False
        finally:
            env.close()
        collision = bool(info.get("collision", False)); ground = bool(info.get("ground_contact", False)); success = bool(info.get("success", False)) and finite
        row = {"model": kind, "checkpoint": checkpoint.stem, "task_id": item["task_id"], "family": family,
               "success": success, "collision": collision, "ground": ground, "unsafe": collision or ground,
               "timeout": bool(info.get("timeout", False)), "nonfinite": bool(info.get("nonfinite", False) or not finite),
               "return": float(total_return), "steps": int(info.get("episode_steps", 0)),
               "action_clip_count": int(info.get("action_clip_count", 0)), "wall_time_s": time.perf_counter() - started}
        rows.append(row)
        if (idx + 1) % 100 == 0: print(f"VAL {kind} {checkpoint.stem}: {idx + 1}/{len(val_tasks)}", flush=True)
    return rows


def classify(curves: dict, metrics: dict) -> dict:
    out = {}
    for kind in ("BC", "DIFFUSION"):
        vals = {int(p): float(metrics[kind][str(p)]["success_rate"]) for p in CHECKPOINT_PASSES}
        out[kind] = {"success_by_pass": vals,
                     "delta_1_to_5": vals[5] - vals[1], "delta_5_to_10": vals[10] - vals[5],
                     "delta_10_to_20": vals[20] - vals[10], "delta_20_to_30": vals[30] - vals[20],
                     "undertraining": bool(vals[30] - vals[1] >= 0.10 and vals[30] > vals[1]),
                     "saturated": bool(max(vals[p] for p in (10, 20, 30)) - min(vals[p] for p in (10, 20, 30)) < 0.05)}
    d = out["DIFFUSION"]
    if d["undertraining"]:
        label = "PASS_S8R3_DIFFUSION_UNDERTRAINING_CONFIRMED"
        classification = "UNDERTRAINING_SUPPORTED"
    elif d["saturated"]:
        label = "PASS_S8R3_MLP_DIFFUSION_SATURATED_BC_STILL_STRONGER"
        classification = "TRAINING_SATURATED"
    else:
        label = "PASS_S8R3_TRAINING_SUFFICIENCY_AUDIT_COMPLETE"
        classification = "INCONCLUSIVE_WITHOUT_UNDERTRAINING_GATE"
    out["overall"] = {"diffusion_classification": classification, "final_label": label,
                       "bc_undertraining": out["BC"]["undertraining"]}
    return out


def run() -> dict:
    if not torch.cuda.is_available(): raise RuntimeError("BLOCKED_S8R3_CUDA_UNAVAILABLE")
    if os.environ.get("S8R3_ALLOW_TEST", ""):
        raise RuntimeError("BLOCKED_S8R3_TEST_UNLOCK_ATTEMPT")
    set_deterministic(TRAINING_SEED)
    device = torch.device("cuda")
    identity = validate_dataset_identity()
    train, val = load_npz(DATA_ROOT / "train.npz"), load_npz(DATA_ROOT / "val.npz")
    mean, std = train_normalization(train)
    norm_sha = sha256_file(OUT / "train_obs_normalization.npz")
    write_json(OUT / "diffusion_frozen_config.json", {"task": TASK, "model": MODEL_CONFIG_DIFF, **DIFFUSION_OPTIMIZER_CONFIG,
                                                         "training_seed": TRAINING_SEED, "normalization_sha256": norm_sha,
                                                         "runtime": "FAST_CUDA_GRAPH_EXACT_BATCH_ONE", "test_accessed": False})
    diff_cfg_sha = sha256_file(OUT / "diffusion_frozen_config.json")
    bc_cfg_sha = canonical_sha({"model": MODEL_CONFIG_BC, **OPTIMIZER_CONFIG, "seed": TRAINING_SEED,
                                "normalization_sha256": norm_sha, "horizon": HORIZON, "action_dim": ACTION_DIM,
                                "deployment": "receding_horizon_first_action"})
    window_stats = {"train_episodes": identity["train_episodes"], "train_transitions": identity["train_transitions"],
                    "train_h16_windows": identity["train_windows"], "val_episodes": identity["val_episodes"],
                    "val_transitions": identity["val_transitions"], "val_h16_windows": identity["val_windows"],
                    "batch_size": BATCH_SIZE, "updates_per_effective_pass": math.ceil(identity["train_windows"] / BATCH_SIZE),
                    "max_effective_passes": MAX_PASSES, "training_seed": TRAINING_SEED,
                    "test_accessed": False}
    write_json(OUT / "window_statistics.json", window_stats)
    train_obs_np, train_actions_np = build_windows(train); val_obs_np, val_actions_np = build_windows(val)
    train_obs = normalize_to_device(train_obs_np, mean, std, device); val_obs = normalize_to_device(val_obs_np, mean, std, device)
    train_actions = torch.from_numpy(train_actions_np).to(device); val_actions = torch.from_numpy(val_actions_np).to(device)
    updates = window_stats["updates_per_effective_pass"]
    bc_curve, bc_summary = train_bc(train_obs, train_actions, val_obs, val_actions, mean, std, updates, norm_sha)
    diff_curve, diff_summary = train_diffusion(train_obs, train_actions, val_obs, val_actions, mean, std, updates, norm_sha)
    curves = bc_curve + diff_curve
    write_csv(OUT / "training_curve.csv", curves)
    val_tasks = read_val_tasks()
    # Initial checkpoint is deliberately a labelled untrained state; no policy rollout is required for pass 0.
    metrics_rows: list[dict] = []; metrics_by_model: dict = {"BC": {}, "DIFFUSION": {}}
    for kind, summary in (("BC", bc_summary), ("DIFFUSION", diff_summary)):
        metrics_by_model[kind]["0"] = {"status": "INITIAL_UNTRAINED", "count": 0, "success_rate": None}
        for p in CHECKPOINT_PASSES:
            rows = rollout_checkpoint(kind, Path(summary["checkpoints"][str(p)]["path"]), mean, std, val_tasks, device)
            write_csv(OUT / f"{kind.lower()}_pass_{p:02d}_val_rollouts.csv", rows)
            result = summarize_rows(rows); result["model"] = kind; result["pass"] = p
            metrics_by_model[kind][str(p)] = result
            metrics_rows.append(result)
    write_csv(OUT / "closed_loop_metrics.csv", metrics_rows)
    family_rows = []
    for row in metrics_rows:
        for family, values in row["by_family"].items():
            family_rows.append({"model": row["model"], "pass": row["pass"], "family": family, **values})
    write_csv(OUT / "family_metrics.csv", family_rows)
    classification = classify({"BC": bc_curve, "DIFFUSION": diff_curve}, metrics_by_model)
    summary = {"task": TASK, "final_label": classification["overall"]["final_label"], "start_head": START_HEAD,
               "branch": "agent/s8r3-training-sufficiency-audit-v1", "dataset": identity, "window_statistics": window_stats,
               "normalization_sha256": norm_sha, "bc_config_sha256": bc_cfg_sha, "diffusion_config_sha256": diff_cfg_sha,
               "training_seed": TRAINING_SEED, "max_effective_passes": MAX_PASSES,
               "bc": bc_summary, "diffusion": diff_summary, "closed_loop": metrics_by_model,
               "classification": classification, "test_accessed": False, "ppo_run_count": 0,
               "test_run_count": 0, "test_values_opened": False, "storage": {"npz_submitted": False, "checkpoints_submitted": False},
               "formal_progress": "95%", "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW",
               "gpu": torch.cuda.get_device_name(0), "pytorch": torch.__version__, "cuda": torch.version.cuda or "none",
               "regression": None, "independent_verification": None}
    write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("all", "train", "val"), default="all")
    args = parser.parse_args()
    if args.phase != "all":
        raise RuntimeError("S8R3 requires one continuous all-phase run; phase splitting is forbidden")
    print(json.dumps(run(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
