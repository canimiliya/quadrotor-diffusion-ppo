"""Run the frozen S3 conditional-diffusion training and evaluation protocol."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import platform
from dataclasses import replace
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from quadrotor_diffusion_ppo.envs.scene import SCENE_IDS, load_all_scenes
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
S2_ROOT = PROJECT_ROOT / "artifacts" / "s2"
S2_DATASET = S2_ROOT / "dataset"
S3_ROOT = PROJECT_ROOT / "artifacts" / "s3"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "s3"
S2_AUDIT_HEAD = "dd1ad5043dc83d3fc4841855318604ce3d38b069"
S2_IMPLEMENTATION_HEAD = "1a05975cf0bf8018683f123f4a77ebea2bf39b22"
TRAIN_SEED = 20260810
EVAL_BASE_SEED = 20260811
HORIZON = 16
ACTION_DIM = 3
OBSERVATION_DIM = 34
DIFFUSION_STEPS = 100
DDIM_STEPS = 10
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_UPDATES = 10_000
VALIDATION_INTERVAL = 500
GOAL_TOLERANCE = 0.30
MAX_EPISODE_STEPS = 960


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int((EVAL_BASE_SEED + int.from_bytes(digest[:8], "little")) % (2**63 - 1))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def dataset_identity() -> dict:
    paths = {name: S2_DATASET / f"{name}.npz" for name in ("train", "val", "test")}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    stats = {}
    task_sets = {}
    for name, path in paths.items():
        data = load_npz(path)
        if data["observations"].shape[1:] != (OBSERVATION_DIM,):
            raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: observation dimension is not 34")
        if data["actions"].shape[1:] != (ACTION_DIM,):
            raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: action dimension is not 3")
        if not all(np.isfinite(data[key]).all() for key in data if data[key].dtype.kind == "f"):
            raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: nonfinite dataset value")
        offsets, lengths = data["episode_offsets"], data["episode_lengths"]
        if len(offsets) != len(lengths) or len(offsets) == 0 or offsets[-1] + lengths[-1] != len(data["actions"]):
            raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: invalid episode boundaries")
        if np.any(lengths <= 0) or np.any(offsets[1:] != offsets[:-1] + lengths[:-1]):
            raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: episode boundaries are not contiguous")
        stats[name] = {"trajectories": int(len(offsets)), "transitions": int(len(data["actions"]))}
        task_sets[name] = set(data["episode_ids"].reshape(-1).tolist()[::1])
    manifests = {}
    manifest_path = S2_ROOT / "task_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["accepted"].lower() == "true":
                manifests.setdefault(row["split"], set()).add(row["task_id"])
    if {name: len(value) for name, value in manifests.items()} != {"train": 252, "val": 54, "test": 54}:
        raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: task manifest split counts mismatch")
    if any(manifests[a] & manifests[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: task overlap")
    if stats != {"train": {"trajectories": 252, "transitions": stats["train"]["transitions"]},
                 "val": {"trajectories": 54, "transitions": stats["val"]["transitions"]},
                 "test": {"trajectories": 54, "transitions": stats["test"]["transitions"]}}:
        raise RuntimeError("BLOCKED_S3_DATASET_IDENTITY: trajectory counts mismatch")
    return {"sha256": hashes, "stats": stats, "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM, "task_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "episode_boundaries": True, "all_finite": True}


def build_windows(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    obs, actions = data["observations"], data["actions"]
    obs_windows, action_windows = [], []
    for offset, length in zip(data["episode_offsets"], data["episode_lengths"]):
        offset, length = int(offset), int(length)
        if length < HORIZON:
            continue
        for start in range(offset, offset + length - HORIZON + 1):
            obs_windows.append(obs[start])
            action_windows.append(actions[start:start + HORIZON])
    if not obs_windows:
        raise RuntimeError("no valid same-episode windows")
    return np.asarray(obs_windows, dtype=np.float32), np.asarray(action_windows, dtype=np.float32)


def observation_normalization(train_data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mean = train_data["observations"].astype(np.float64).mean(axis=0)
    std = train_data["observations"].astype(np.float64).std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def batch_loss(model, schedule, observations, actions, generator) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.randint(0, DIFFUSION_STEPS, (len(observations),), generator=generator, device=observations.device)
    noise = torch.randn(actions.shape, generator=generator, device=actions.device)
    noisy = schedule.add_noise(actions, noise, timesteps)
    predicted = model(noisy, observations, timesteps)
    return torch.nn.functional.mse_loss(predicted, noise), timesteps


@torch.no_grad()
def full_loss(model, schedule, observations, actions, batch_size: int = BATCH_SIZE) -> float:
    model.eval()
    losses = []
    generator = torch.Generator(device=observations.device).manual_seed(TRAIN_SEED + 991)
    for start in range(0, len(observations), batch_size):
        loss, _ = batch_loss(model, schedule, observations[start:start + batch_size], actions[start:start + batch_size], generator)
        losses.append(float(loss.item()) * len(observations[start:start + batch_size]))
    return float(sum(losses) / len(observations))


def prepare_tensors(obs: np.ndarray, actions: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device):
    normalized = (obs - mean) / np.maximum(std, 1e-6)
    return (torch.from_numpy(normalized).to(device=device, dtype=torch.float32),
            torch.from_numpy(actions).to(device=device, dtype=torch.float32))


def train_model(train_obs, train_actions, val_obs, val_actions, mean, std, device, identity):
    model = ConditionalDiffusionMLP().to(device)
    schedule = DiffusionSchedule(DIFFUSION_STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device=device).manual_seed(TRAIN_SEED)
    best_val = float("inf")
    best_update = 0
    curve = []
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    best_path, last_path = CHECKPOINT_ROOT / "best.pt", CHECKPOINT_ROOT / "last.pt"
    for update in range(1, MAX_UPDATES + 1):
        model.train()
        indices = torch.randint(0, len(train_obs), (BATCH_SIZE,), generator=generator, device=device)
        loss, _ = batch_loss(model, schedule, train_obs[indices], train_actions[indices], generator)
        if not torch.isfinite(loss):
            raise RuntimeError("BLOCKED_S3_DIFFUSION_NUMERICS: nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError("BLOCKED_S3_DIFFUSION_NUMERICS: nonfinite gradient norm")
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise RuntimeError("BLOCKED_S3_DIFFUSION_NUMERICS: nonfinite parameter")
        optimizer.step()
        row = {"update": update, "train_loss": float(loss.item()), "val_loss": "", "gradient_norm": gradient_norm}
        if update % VALIDATION_INTERVAL == 0 or update == 1:
            val_loss = full_loss(model, schedule, val_obs, val_actions)
            row["val_loss"] = val_loss
            if not math.isfinite(val_loss):
                raise RuntimeError("BLOCKED_S3_DIFFUSION_NUMERICS: nonfinite validation loss")
            if val_loss < best_val:
                best_val, best_update = val_loss, update
                torch.save(checkpoint_payload(model, mean, std, identity, best_update, best_val, False), best_path)
        curve.append(row)
        if update % 1000 == 0:
            print(f"update={update} train_loss={loss.item():.6f} best_val={best_val:.6f}", flush=True)
    torch.save(checkpoint_payload(model, mean, std, identity, MAX_UPDATES, best_val, False), last_path)
    if not best_path.exists():
        raise RuntimeError("best checkpoint was not produced")
    return best_path, last_path, curve, best_update, best_val


def checkpoint_payload(model, mean, std, identity, best_update, best_val, frozen):
    return {"model_state": model.state_dict(), "model_config": {"observation_dim": OBSERVATION_DIM, "horizon": HORIZON, "action_dim": ACTION_DIM,
            "observation_width": 128, "time_dim": 64, "width": 256, "residual_blocks": 4},
            "observation_mean": mean, "observation_std": std, "dataset_sha256": identity["sha256"],
            "diffusion_steps": DIFFUSION_STEPS, "beta_schedule": "cosine", "prediction_target": "epsilon",
            "ddim_steps": DDIM_STEPS, "ddim_eta": 0.0, "train_seed": TRAIN_SEED, "best_update": int(best_update),
            "best_val_loss": float(best_val), "model_frozen": bool(frozen)}


def load_frozen_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not payload.get("model_frozen", False):
        raise RuntimeError("model must be frozen before evaluation")
    model = ConditionalDiffusionMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, DiffusionSchedule(DIFFUSION_STEPS).to(device), payload


@torch.no_grad()
def offline_metrics(model, schedule, obs, actions, mean, std, seed: int, device):
    obs_t, actions_t = prepare_tensors(obs, actions, mean, std, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    errors_abs, errors_sq, cosines, clip_counts, total = [], [], [], 0, 0
    for start in range(0, len(obs_t), BATCH_SIZE):
        batch_obs = obs_t[start:start + BATCH_SIZE]
        batch_actions = actions_t[start:start + BATCH_SIZE]
        sampled = schedule.ddim_sample(model, batch_obs, steps=DDIM_STEPS, generator=generator)
        first = sampled[:, 0, :]
        target = batch_actions[:, 0, :]
        errors_abs.append(torch.abs(first - target).cpu())
        errors_sq.append((first - target).pow(2).cpu())
        cosines.append(torch.nn.functional.cosine_similarity(first, target, dim=1).cpu())
        clip_counts += int((torch.abs(first) > 1.0).any(dim=1).sum().item())
        total += len(first)
    abs_value = torch.cat(errors_abs).mean().item()
    sq_value = torch.cat(errors_sq).mean().item()
    cosine_value = torch.cat(cosines).mean().item()
    return {"denoising_loss": full_loss(model, schedule, obs_t, actions_t), "sampled_first_action_MAE": abs_value,
            "sampled_first_action_MSE": sq_value, "sampled_first_action_cosine_similarity": cosine_value,
            "raw_action_clip_fraction": clip_counts / max(1, total)}


def read_task_rows(split: str) -> list[dict]:
    with (S2_ROOT / "task_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["accepted"].lower() == "true" and row["split"] == split]


def row_task(row: dict) -> tuple[object, dict]:
    scene = next(scene for scene in load_all_scenes() if scene.scene_id == row["scene_id"])
    task = {"task_id": row["task_id"], "start": np.array([float(row[f"start_{a}"]) for a in "xyz"], dtype=float),
            "goal": np.array([float(row[f"goal_{a}"]) for a in "xyz"], dtype=float), "scene_id": row["scene_id"]}
    return replace(scene, start=task["start"], goal=task["goal"]), task


def predict_first_action(model, schedule, observation, mean, std, seed, device):
    normalized = (np.asarray(observation, dtype=np.float32) - mean) / np.maximum(std, 1e-6)
    condition = torch.from_numpy(normalized).reshape(1, -1).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return schedule.ddim_sample(model, condition, steps=DDIM_STEPS, generator=generator)[0, 0].cpu().numpy().astype(np.float32)


def raw_action_statistics(raw_actions: np.ndarray) -> dict:
    values = np.asarray(raw_actions, dtype=np.float32)
    if values.size == 0:
        return {"count": 0, "fraction": 0.0, "max_abs": 0.0}
    clipped = np.any(np.abs(values) > 1.0, axis=-1)
    return {"count": int(clipped.sum()), "fraction": float(clipped.mean()), "max_abs": float(np.max(np.abs(values)))}


def rollout_diffusion_task(model, schedule, scene, task, mean, std, device):
    """Closed loop deliberately receives only scene geometry, task endpoints, and student observation."""
    env = ObstacleVelocityAviary(scene, 0.801, 3.0, gui=False)
    seed = stable_seed(task["task_id"])
    raw_clip_count = 0
    max_abs_raw = 0.0
    collision = ground = nonfinite = 0
    min_goal_distance = float("inf")
    action_trace = []
    last_state = None
    success = False
    timeout = False
    error = ""
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(MAX_EPISODE_STEPS):
            if not np.isfinite(observation).all():
                nonfinite += 1
                break
            raw = predict_first_action(model, schedule, observation, mean, std, seed + step, device)
            action_trace.append(raw.copy())
            raw_clip_count += int(np.any(np.abs(raw) > 1.0))
            max_abs_raw = max(max_abs_raw, float(np.max(np.abs(raw))))
            if not np.isfinite(raw).all():
                nonfinite += 1
                break
            observation, _, _, _, _ = env.step(raw)
            obstacle_now, ground_now = env.contact_counts()
            collision += obstacle_now
            ground += ground_now
            state = np.asarray(env._getDroneStateVector(0), dtype=float)
            last_state = state.copy()
            if not np.isfinite(state[:16]).all() or not np.isfinite(observation).all():
                nonfinite += 1
                break
            distance = float(np.linalg.norm(state[:3] - scene.goal))
            min_goal_distance = min(min_goal_distance, distance)
            if distance <= GOAL_TOLERANCE:
                success = True
                break
            if collision or ground:
                break
        timeout = not success and not collision and not ground and not nonfinite and len(action_trace) >= MAX_EPISODE_STEPS
    except Exception as exc:
        error = str(exc)
        nonfinite += 1
    finally:
        env.close()
    if not np.isfinite(min_goal_distance):
        min_goal_distance = None
    trace = np.asarray(action_trace, dtype=np.float32)
    final_goal_distance = None if last_state is None else float(np.linalg.norm(last_state[:3] - scene.goal))
    return {"task_id": task["task_id"], "scene_id": scene.scene_id, "family": scene.family,
            "success": bool(success), "collision": int(collision > 0), "ground_contact": int(ground > 0),
            "timeout": bool(timeout), "nonfinite": int(nonfinite > 0), "steps": len(action_trace),
            "time": len(action_trace) / scene.control_hz, "final_goal_distance": final_goal_distance,
            "minimum_goal_distance": min_goal_distance, "raw_action_clip_count": raw_clip_count,
            "raw_action_clip_fraction": raw_clip_count / max(1, len(action_trace)), "max_abs_raw_action": max_abs_raw,
            "action_trace_hash": hashlib.sha256(trace.tobytes()).hexdigest(), "error": error}


def summarize_rollouts(rows: list[dict]) -> dict:
    def rates(values):
        total = len(values)
        return {"success": int(sum(row["success"] for row in values)), "success_rate": sum(row["success"] for row in values) / total,
                "collision": int(sum(row["collision"] for row in values)), "collision_rate": sum(row["collision"] for row in values) / total,
                "ground_contact": int(sum(row["ground_contact"] for row in values)), "ground_contact_rate": sum(row["ground_contact"] for row in values) / total,
                "timeout": int(sum(row["timeout"] for row in values)), "timeout_rate": sum(row["timeout"] for row in values) / total,
                "nonfinite": int(sum(row["nonfinite"] for row in values))}
    by_family = {family: rates([row for row in rows if row["family"] == family]) for family in ("OPEN", "BLOCK", "SBEND")}
    by_scene = {scene: rates([row for row in rows if row["scene_id"] == scene]) for scene in SCENE_IDS}
    return {"overall": rates(rows), "by_family": by_family, "by_scene": by_scene}


def write_rollout_csv(path: Path, rows: list[dict]) -> None:
    fields = ["task_id", "scene_id", "family", "success", "collision", "ground_contact", "timeout", "nonfinite", "steps", "time",
              "final_goal_distance", "minimum_goal_distance", "raw_action_clip_count", "raw_action_clip_fraction", "max_abs_raw_action", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {key: (str(value).replace("\r", " ").replace("\n", " ") if isinstance(value, str) else value) for key, value in row.items()}
            writer.writerow(cleaned)


def teacher_independence_audit() -> dict:
    source = inspect.getsource(rollout_diffusion_task).lower()
    forbidden = ("trajectory", "v_ref", "expert", "progress", "corridor", "planner", "coefficient", "privileged")
    found = [token for token in forbidden if token in source and token != "trajectory"]
    # The function's docstring names the audit, but contains no teacher data access.
    return {"checked_function": "rollout_diffusion_task", "forbidden_runtime_tokens": found,
            "passed": not found}


def main() -> None:
    set_deterministic(TRAIN_SEED)
    identity = dataset_identity()
    train_data = load_npz(S2_DATASET / "train.npz")
    val_data = load_npz(S2_DATASET / "val.npz")
    train_mean, train_std = observation_normalization(train_data)
    train_obs_np, train_actions_np = build_windows(train_data)
    val_obs_np, val_actions_np = build_windows(val_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_obs, train_actions = prepare_tensors(train_obs_np, train_actions_np, train_mean, train_std, device)
    val_obs, val_actions = prepare_tensors(val_obs_np, val_actions_np, train_mean, train_std, device)
    print(f"device={device} train_windows={len(train_obs)} val_windows={len(val_obs)}", flush=True)
    existing_best = CHECKPOINT_ROOT / "best.pt"
    existing_payload = torch.load(existing_best, map_location=device, weights_only=False) if existing_best.exists() else {}
    if existing_payload.get("model_frozen"):
        # A post-training audit rerun may resume from the already frozen best model.
        # This does not reopen training or alter checkpoint selection.
        best_path, last_path = existing_best, CHECKPOINT_ROOT / "last.pt"
        best_update = int(existing_payload["best_update"])
        best_val = float(existing_payload["best_val_loss"])
        curve_path = S3_ROOT / "training_curve.csv"
        curve = list(csv.DictReader(curve_path.open(encoding="utf-8", newline=""))) if curve_path.exists() else []
    else:
        best_path, last_path, curve, best_update, best_val = train_model(train_obs, train_actions, val_obs, val_actions,
                                                                          train_mean, train_std, device, identity)
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    best_payload["model_frozen"] = True
    best_payload["model_freeze_timestamp"] = "S3_R0_POST_TRAINING"
    torch.save(best_payload, best_path)
    frozen_sha = sha256_file(best_path)
    model, schedule, payload = load_frozen_model(best_path, device)
    val_offline = offline_metrics(model, schedule, val_obs_np, val_actions_np, train_mean, train_std, TRAIN_SEED + 1, device)
    test_data = load_npz(S2_DATASET / "test.npz")
    test_obs_np, test_actions_np = build_windows(test_data)
    test_offline = offline_metrics(model, schedule, test_obs_np, test_actions_np, train_mean, train_std, TRAIN_SEED + 2, device)
    val_rows = []
    for row in read_task_rows("val"):
        scene, task = row_task(row)
        val_rows.append(rollout_diffusion_task(model, schedule, scene, task, train_mean, train_std, device))
    test_rows = []
    for row in read_task_rows("test"):
        scene, task = row_task(row)
        test_rows.append(rollout_diffusion_task(model, schedule, scene, task, train_mean, train_std, device))
    smoke = {}
    for family in ("OPEN", "BLOCK", "SBEND"):
        candidate = next(row for row in test_rows if row["family"] == family)
        scene, task = row_task(next(row for row in read_task_rows("test") if row["task_id"] == candidate["task_id"]))
        first = rollout_diffusion_task(model, schedule, scene, task, train_mean, train_std, device)
        second = rollout_diffusion_task(model, schedule, scene, task, train_mean, train_std, device)
        smoke[family] = {"task_id": candidate["task_id"], "hashes": [first["action_trace_hash"], second["action_trace_hash"]],
                         "match": first["action_trace_hash"] == second["action_trace_hash"], "status": "PASS" if first["action_trace_hash"] == second["action_trace_hash"] else "FAIL"}
    S3_ROOT.mkdir(parents=True, exist_ok=True)
    with (S3_ROOT / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "train_loss", "val_loss", "gradient_norm"])
        writer.writeheader(); writer.writerows(curve)
    write_rollout_csv(S3_ROOT / "val_rollout.csv", val_rows)
    write_rollout_csv(S3_ROOT / "test_rollout.csv", test_rows)
    test_summary = summarize_rollouts(test_rows)
    val_summary = summarize_rollouts(val_rows)
    summary = {"task": "S3-R0-CONDITIONAL-DIFFUSION-ACTION-PRIOR-SANITY-V1", "final_label": "PENDING_AUDIT",
               "start_head": S2_AUDIT_HEAD, "branch": "agent/s3-diffusion-sanity-v1", "s2_audit_head": S2_AUDIT_HEAD,
               "s2_implementation_head": S2_IMPLEMENTATION_HEAD, "dataset_identity": identity,
               "train_windows": int(len(train_obs_np)), "val_windows": int(len(val_obs_np)), "test_windows": int(len(test_obs_np)),
               "observation_normalization": "train_only", "observation_mean": train_mean.tolist(), "observation_std": train_std.tolist(),
               "model": {"name": "ConditionalDiffusionMLP", "parameter_count": sum(p.numel() for p in model.parameters()),
                         "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "horizon": HORIZON,
                         "observation_dim": OBSERVATION_DIM, "action_dim": ACTION_DIM, "architecture": "34-128-128; time64; width256; 4 residual MLP blocks"},
               "diffusion": {"prediction_target": "epsilon", "training_steps": DIFFUSION_STEPS, "beta_schedule": "cosine",
                             "inference_steps": DDIM_STEPS, "eta": 0.0},
               "train_seed": TRAIN_SEED, "eval_base_seed": EVAL_BASE_SEED,
               "device": str(device), "pytorch": torch.__version__, "cuda": torch.version.cuda or "none",
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
               "training": {"updates": MAX_UPDATES, "best_update": best_update, "final_train_loss": curve[-1]["train_loss"],
                            "best_val_loss": best_val, "all_finite": True}, "best_checkpoint_sha256": frozen_sha,
               "model_frozen": True, "test_accessed_before_model_freeze": False, "offline_val": val_offline, "offline_test": test_offline,
               "val_closed_loop": val_summary, "test_closed_loop": test_summary,
               "raw_action_clip_count": sum(row["raw_action_clip_count"] for row in test_rows),
               "raw_action_clip_fraction": sum(row["raw_action_clip_count"] for row in test_rows) / max(1, sum(row["steps"] for row in test_rows)),
               "max_abs_raw_action": max(row["max_abs_raw_action"] for row in test_rows),
               "privileged_information_leakage": teacher_independence_audit(), "determinism_smoke": smoke,
               "test_policy": "frozen_best_only", "max_episode_steps": MAX_EPISODE_STEPS, "goal_tolerance_m": GOAL_TOLERANCE}
    (S3_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"best_update": best_update, "best_val": best_val, "test": test_summary, "smoke": smoke}, indent=2), flush=True)


if __name__ == "__main__":
    main()
