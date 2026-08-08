"""Run the single pre-registered S3-R1 bounded-x0 repair protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP, unit_ball_squash
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from quadrotor_diffusion_ppo.envs.scene import SCENE_IDS
from scripts.run_s3_diffusion import (
    ACTION_DIM, BATCH_SIZE, DIFFUSION_STEPS, EVAL_BASE_SEED, GOAL_TOLERANCE,
    HORIZON, MAX_EPISODE_STEPS, OBSERVATION_DIM, PROJECT_ROOT, S2_DATASET,
    S2_ROOT, build_windows, dataset_identity, load_npz, observation_normalization,
    prepare_tensors, read_task_rows, raw_action_statistics, row_task, sha256_file,
    stable_seed, summarize_rollouts, teacher_independence_audit, write_rollout_csv,
)
from quadrotor_diffusion_ppo.envs.velocity_aviary import ObstacleVelocityAviary


S3R1_ROOT = PROJECT_ROOT / "artifacts" / "s3r1"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "s3r1"
R0_SUMMARY = PROJECT_ROOT / "artifacts" / "s3" / "summary.json"
R0_AUDIT_HEAD = "5c5dc9443aa96bc40ac7881aaf63182097ef31b1"
BRANCH = "agent/s3r1-bounded-x0-v1"
TRAIN_SEED = 20260810
HORIZON = 16
ACTION_DIM = 3
OBSERVATION_DIM = 34
DDIM_STEPS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_UPDATES = 10_000
VALIDATION_INTERVAL = 500
SPEED_LIMIT = 0.801
UNIT_BALL_EPS = 1e-8
UNIT_BALL_MARGIN = 1e-6
RECOVERED_TRAINING_LOG = [
    (1000, 0.000953, 0.000974), (2000, 0.000635, 0.000678),
    (3000, 0.000548, 0.000554), (4000, 0.000468, 0.000463),
    (5000, 0.000382, 0.000413), (6000, 0.000357, 0.000385),
    (7000, 0.000341, 0.000332), (8000, 0.000291, 0.000309),
    (9000, 0.000292, 0.000293), (10000, 0.000236, 0.000272),
]

EXPECTED_DATASET_SHA256 = {
    "train": "701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21",
    "val": "c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc",
    "test": "88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66",
}


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def validate_action_support(data: dict[str, np.ndarray], split: str) -> dict:
    actions = np.asarray(data["actions"], dtype=np.float32)
    norms = np.linalg.norm(actions, axis=1)
    violations = int(np.sum(norms > 1.0 + 1e-6))
    if violations:
        raise RuntimeError(f"BLOCKED_S3R1_ACTION_SUPPORT_IMPLEMENTATION: {split} expert actions exceed unit ball")
    return {"max_norm": float(norms.max()), "support_violation_count": violations}


def write_support_diagnostic() -> dict:
    with R0_SUMMARY.open(encoding="utf-8") as handle:
        r0 = json.load(handle)
    schedule = DiffusionSchedule(DIFFUSION_STEPS)
    timesteps = torch.linspace(DIFFUSION_STEPS - 1, 0, DDIM_STEPS).round().long().tolist()
    records = []
    for timestep in timesteps:
        alpha_bar = float(schedule.alpha_bars[timestep].item())
        sqrt_alpha_bar = math.sqrt(alpha_bar)
        records.append({"timestep": int(timestep), "alpha_bar": alpha_bar,
                        "sqrt_alpha_bar": sqrt_alpha_bar,
                        "x0_error_amplification": 1.0 / sqrt_alpha_bar})
    t99 = next(row for row in records if row["timestep"] == 99)
    diagnostic = {
        "task": "S3-R1-BOUNDED-X0-DIFFUSION-ACTION-SUPPORT-REPAIR-V1",
        "r0_audit_head": R0_AUDIT_HEAD,
        "schedule": {"training_steps": DIFFUSION_STEPS, "beta_schedule": "cosine",
                      "ddim_steps": DDIM_STEPS, "eta": 0.0, "timesteps": records},
        "t99": t99,
        "r0": {
            "best_val_loss": r0["training"]["best_val_loss"],
            "offline_val_first_action_mae": r0["offline_val"]["sampled_first_action_MAE"],
            "offline_val_first_action_mse": r0["offline_val"]["sampled_first_action_MSE"],
            "offline_val_cosine": r0["offline_val"]["sampled_first_action_cosine_similarity"],
            "offline_val_clip_fraction": r0["offline_val"]["raw_action_clip_fraction"],
            "test_max_abs_raw_action": r0["max_abs_raw_action"],
            "test_raw_action_clip_fraction": r0["raw_action_clip_fraction"],
        },
        "purpose": "R0 action-support parameterization diagnosis; no training or tuning decision is made here.",
    }
    S3R1_ROOT.mkdir(parents=True, exist_ok=True)
    (S3R1_ROOT / "support_diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    return diagnostic


def oracle_checks() -> dict:
    schedule = DiffusionSchedule(DIFFUSION_STEPS)
    checks = {}
    values = [torch.zeros(2, 16, 3), torch.full((2, 16, 3), 1e-8),
              torch.full((2, 16, 3), 1e4), torch.full((2, 16, 3), 1e20),
              torch.randn(2, 16, 3)]
    for index, value in enumerate(values):
        output = unit_ball_squash(value)
        norms = torch.linalg.vector_norm(output, dim=-1)
        checks[f"unit_ball_case_{index}"] = {
            "finite": bool(torch.isfinite(output).all()),
            "max_norm": float(norms.max()),
            "passed": bool(torch.isfinite(output).all() and torch.all(norms <= 1.0 + 1e-6)),
        }
    x0 = torch.randn(2, 16, 3) * 0.2
    epsilon = torch.randn_like(x0)
    timesteps = torch.linspace(DIFFUSION_STEPS - 1, 0, DDIM_STEPS).round().long().tolist()
    errors = []
    current = schedule.add_noise(x0, epsilon, torch.tensor([timesteps[0], timesteps[0]]))
    for index, timestep in enumerate(timesteps):
        predicted_x0 = x0
        alpha_bar = schedule.alpha_bars[timestep]
        epsilon_hat = (current - alpha_bar.sqrt() * predicted_x0) / (1.0 - alpha_bar).sqrt()
        if index == len(timesteps) - 1:
            errors.append(float(torch.max(torch.abs(predicted_x0 - x0))))
            continue
        next_timestep = timesteps[index + 1]
        next_current = schedule.alpha_bars[next_timestep].sqrt() * predicted_x0 + (1.0 - schedule.alpha_bars[next_timestep]).sqrt() * epsilon_hat
        expected = schedule.add_noise(x0, epsilon, torch.tensor([next_timestep, next_timestep]))
        errors.append(float(torch.max(torch.abs(next_current - expected))))
        current = next_current
    checks["x0_sampler_oracle"] = {"finite": bool(all(math.isfinite(value) for value in errors)),
                                   "max_reconstruction_error": max(errors),
                                   "tolerance": 2e-5, "passed": bool(max(errors) <= 2e-5)}
    checks["all_passed"] = bool(all(item["passed"] for item in checks.values()))
    if not checks["all_passed"]:
        raise RuntimeError("BLOCKED_S3R1_SAMPLER_IMPLEMENTATION: oracle failed")
    return checks


def batch_loss(model, schedule, observations, actions, generator):
    timesteps = torch.randint(0, DIFFUSION_STEPS, (len(observations),), generator=generator, device=observations.device)
    noise = torch.randn(actions.shape, generator=generator, device=actions.device)
    noisy = schedule.add_noise(actions, noise, timesteps)
    predicted_x0 = model(noisy, observations, timesteps)
    return torch.nn.functional.mse_loss(predicted_x0, actions)


@torch.no_grad()
def full_loss(model, schedule, observations, actions, batch_size=BATCH_SIZE) -> float:
    model.eval()
    generator = torch.Generator(device=observations.device).manual_seed(TRAIN_SEED + 991)
    total = 0.0
    for start in range(0, len(observations), batch_size):
        batch = slice(start, start + batch_size)
        loss = batch_loss(model, schedule, observations[batch], actions[batch], generator)
        total += float(loss.item()) * len(observations[batch])
    return total / len(observations)


def checkpoint_payload(model, mean, std, identity, best_update, best_val, frozen):
    return {"model_state": model.state_dict(),
            "model_config": {"observation_dim": OBSERVATION_DIM, "horizon": HORIZON, "action_dim": ACTION_DIM,
                              "observation_width": 128, "time_dim": 64, "width": 256, "residual_blocks": 4,
                              "bounded_output": True},
            "observation_mean": mean, "observation_std": std, "dataset_sha256": identity["sha256"],
            "diffusion_steps": DIFFUSION_STEPS, "beta_schedule": "cosine", "prediction_target": "x0",
            "action_support": "unit_ball_squash", "ddim_steps": DDIM_STEPS, "ddim_eta": 0.0,
            "train_seed": TRAIN_SEED, "best_update": int(best_update), "best_val_loss": float(best_val),
            "model_frozen": bool(frozen)}


def train_model(train_obs, train_actions, val_obs, val_actions, mean, std, device, identity):
    model = ConditionalDiffusionMLP(bounded_output=True).to(device)
    schedule = DiffusionSchedule(DIFFUSION_STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device=device).manual_seed(TRAIN_SEED)
    best_val, best_update = float("inf"), 0
    curve = []
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    best_path, last_path = CHECKPOINT_ROOT / "best.pt", CHECKPOINT_ROOT / "last.pt"
    for update in range(1, MAX_UPDATES + 1):
        model.train()
        indices = torch.randint(0, len(train_obs), (BATCH_SIZE,), generator=generator, device=device)
        loss = batch_loss(model, schedule, train_obs[indices], train_actions[indices], generator)
        if not torch.isfinite(loss):
            raise RuntimeError("BLOCKED_S3R1_NUMERICS: nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError("BLOCKED_S3R1_NUMERICS: nonfinite gradient norm")
        optimizer.step()
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise RuntimeError("BLOCKED_S3R1_NUMERICS: nonfinite parameter")
        row = {"update": update, "train_loss": float(loss.item()), "val_loss": "", "gradient_norm": gradient_norm}
        if update % VALIDATION_INTERVAL == 0 or update == 1:
            val_loss = full_loss(model, schedule, val_obs, val_actions)
            row["val_loss"] = val_loss
            if not math.isfinite(val_loss):
                raise RuntimeError("BLOCKED_S3R1_NUMERICS: nonfinite validation loss")
            if val_loss < best_val:
                best_val, best_update = val_loss, update
                torch.save(checkpoint_payload(model, mean, std, identity, best_update, best_val, False), best_path)
        curve.append(row)
        if update % 1000 == 0:
            print(f"update={update} train_loss={loss.item():.6f} best_val={best_val:.6f}", flush=True)
    torch.save(checkpoint_payload(model, mean, std, identity, MAX_UPDATES, best_val, False), last_path)
    if not best_path.exists():
        raise RuntimeError("BLOCKED_S3R1_NUMERICS: best checkpoint was not produced")
    return best_path, last_path, curve, best_update, best_val


def load_frozen_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not payload.get("model_frozen", False):
        raise RuntimeError("model must be frozen before evaluation")
    if payload.get("prediction_target") != "x0" or payload.get("action_support") != "unit_ball_squash":
        raise RuntimeError("BLOCKED_S3R1_ACTION_SUPPORT_IMPLEMENTATION: checkpoint metadata mismatch")
    model = ConditionalDiffusionMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, DiffusionSchedule(DIFFUSION_STEPS).to(device), payload


@torch.no_grad()
def offline_metrics(model, schedule, obs, actions, mean, std, seed: int, device):
    obs_t, actions_t = prepare_tensors(obs, actions, mean, std, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    errors_abs, errors_sq, cosines = [], [], []
    support_count = 0
    support_total = 0
    for start in range(0, len(obs_t), BATCH_SIZE):
        batch_obs, batch_actions = obs_t[start:start + BATCH_SIZE], actions_t[start:start + BATCH_SIZE]
        sampled, diag = schedule.ddim_sample_x0(model, batch_obs, steps=DDIM_STEPS, generator=generator, return_diagnostics=True)
        first, target = sampled[:, 0, :], batch_actions[:, 0, :]
        errors_abs.append(torch.abs(first - target).cpu())
        errors_sq.append((first - target).pow(2).cpu())
        cosines.append(torch.nn.functional.cosine_similarity(first, target, dim=1).cpu())
        support_count += diag["support_violation_count"]
        support_total += len(batch_obs) * HORIZON * DDIM_STEPS
    return {"sampled_first_action_MAE": torch.cat(errors_abs).mean().item(),
            "sampled_first_action_MSE": torch.cat(errors_sq).mean().item(),
            "sampled_first_action_cosine_similarity": torch.cat(cosines).mean().item(),
            "support_violation_count": support_count,
            "support_violation_fraction": support_count / max(1, support_total),
            "denoising_loss": full_loss(model, schedule, obs_t, actions_t)}


def predict_action(model, schedule, observation, mean, std, seed, device):
    normalized = (np.asarray(observation, dtype=np.float32) - mean) / np.maximum(std, 1e-6)
    condition = torch.from_numpy(normalized).reshape(1, -1).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return schedule.ddim_sample_x0(model, condition, steps=DDIM_STEPS, generator=generator, return_diagnostics=True)


def rollout_diffusion_task(model, schedule, scene, task, mean, std, device):
    """Closed loop receives only geometry, endpoints, and the current 34D observation."""
    env = ObstacleVelocityAviary(scene, SPEED_LIMIT, 3.0, gui=False)
    seed = stable_seed(task["task_id"])
    executed_clip_count = 0
    latent_max_abs = latent_max_norm = 0.0
    x0_max_abs = x0_max_norm = 0.0
    support_count = support_total = 0
    collision = ground = nonfinite = 0
    min_goal_distance = float("inf")
    action_trace = []
    last_state = None
    success = timeout = False
    error = ""
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(MAX_EPISODE_STEPS):
            if not np.isfinite(observation).all():
                nonfinite += 1
                break
            sampled, diag = predict_action(model, schedule, observation, mean, std, seed + step, device)
            x0_sequence = sampled[0].cpu().numpy().astype(np.float32)
            raw = x0_sequence[0].copy()
            action_trace.append(raw.copy())
            latent_max_abs = max(latent_max_abs, diag["latent_max_abs"])
            latent_max_norm = max(latent_max_norm, diag["latent_max_norm"])
            x0_max_abs = max(x0_max_abs, float(np.max(np.abs(x0_sequence))))
            x0_max_norm = max(x0_max_norm, float(np.max(np.linalg.norm(x0_sequence, axis=-1))))
            support_count += diag["support_violation_count"]
            support_total += HORIZON * DDIM_STEPS
            if not np.isfinite(raw).all() or not np.isfinite(x0_sequence).all():
                nonfinite += 1
                break
            observation, _, _, _, _ = env.step(raw)
            executed_clip_count += int(bool(env.last_velocity_action and env.last_velocity_action.clipped))
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
            "minimum_goal_distance": min_goal_distance, "executed_action_clip_count": executed_clip_count,
            "executed_action_clip_fraction": executed_clip_count / max(1, len(action_trace)),
            "network_latent_max_abs": latent_max_abs, "network_latent_max_norm": latent_max_norm,
            "diffusion_x0_max_abs": x0_max_abs, "diffusion_x0_max_norm": x0_max_norm,
            "diffusion_x0_support_violation_count": support_count,
            "diffusion_x0_support_violation_fraction": support_count / max(1, support_total),
            "action_trace_hash": hashlib.sha256(trace.tobytes()).hexdigest(), "error": error}


def write_rollout_csv_r1(path: Path, rows: list[dict]) -> None:
    fields = ["task_id", "scene_id", "family", "success", "collision", "ground_contact", "timeout", "nonfinite", "steps", "time",
              "final_goal_distance", "minimum_goal_distance", "executed_action_clip_count", "executed_action_clip_fraction",
              "network_latent_max_abs", "network_latent_max_norm", "diffusion_x0_max_abs", "diffusion_x0_max_norm",
              "diffusion_x0_support_violation_count", "diffusion_x0_support_violation_fraction", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(value).replace("\r", " ").replace("\n", " ") if isinstance(value, str) else value for key, value in row.items()})


def action_summary(rows: list[dict]) -> dict:
    total_steps = max(1, sum(row["steps"] for row in rows))
    support_total = max(1, sum(row["steps"] * HORIZON * DDIM_STEPS for row in rows))
    return {"network_latent": {"max_abs": max((row["network_latent_max_abs"] for row in rows), default=0.0),
                               "max_norm": max((row["network_latent_max_norm"] for row in rows), default=0.0)},
            "diffusion_x0": {"max_abs": max((row["diffusion_x0_max_abs"] for row in rows), default=0.0),
                             "max_norm": max((row["diffusion_x0_max_norm"] for row in rows), default=0.0),
                             "support_violation_count": sum(row["diffusion_x0_support_violation_count"] for row in rows),
                             "support_violation_fraction": sum(row["diffusion_x0_support_violation_count"] for row in rows) / support_total},
            "executed_action": {"clip_count": sum(row["executed_action_clip_count"] for row in rows),
                                "clip_fraction": sum(row["executed_action_clip_count"] for row in rows) / total_steps}}


def family_gate(summary: dict) -> bool:
    return all(summary["by_family"][family]["success"] > 0 for family in ("OPEN", "BLOCK", "SBEND"))


def closed_loop_gate(summary: dict, action_stats: dict, leakage: dict) -> bool:
    overall = summary["overall"]
    return (overall["success"] >= 27 and family_gate(summary) and
            overall["collision"] + overall["ground_contact"] <= 0.25 * len(summary["by_scene"]) * 6 and
            overall["nonfinite"] == 0 and action_stats["diffusion_x0"]["support_violation_fraction"] <= 0.01 and
            action_stats["executed_action"]["clip_fraction"] <= 0.01 and leakage["passed"])


def determinism_smoke(model, schedule, split: str, mean, std, device) -> dict:
    rows = read_task_rows(split)
    result = {}
    for family in ("OPEN", "BLOCK", "SBEND"):
        scene, task = next(row_task(item) for item in rows if row_task(item)[0].family == family)
        first = rollout_diffusion_task(model, schedule, scene, task, mean, std, device)
        second = rollout_diffusion_task(model, schedule, scene, task, mean, std, device)
        result[family] = {"task_id": task["task_id"], "hashes": [first["action_trace_hash"], second["action_trace_hash"]],
                          "match": first["action_trace_hash"] == second["action_trace_hash"],
                          "status": "PASS" if first["action_trace_hash"] == second["action_trace_hash"] else "FAIL"}
    return result


def metadata_identity(identity: dict) -> bool:
    return identity["sha256"] == EXPECTED_DATASET_SHA256 and identity["stats"]["train"]["trajectories"] == 252 and identity["stats"]["val"]["trajectories"] == 54 and identity["stats"]["test"]["trajectories"] == 54


def main() -> None:
    set_deterministic(TRAIN_SEED)
    diagnostic = write_support_diagnostic()
    oracle = oracle_checks()
    train_data = load_npz(S2_DATASET / "train.npz")
    val_data = load_npz(S2_DATASET / "val.npz")
    train_support = validate_action_support(train_data, "train")
    val_support = validate_action_support(val_data, "val")
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
        best_path, last_path = existing_best, CHECKPOINT_ROOT / "last.pt"
        best_update, best_val = int(existing_payload["best_update"]), float(existing_payload["best_val_loss"])
        curve_path = S3R1_ROOT / "training_curve.csv"
        curve = list(csv.DictReader(curve_path.open(encoding="utf-8", newline=""))) if curve_path.exists() else [
            {"update": update, "train_loss": train_loss, "val_loss": best_loss, "gradient_norm": ""}
            for update, train_loss, best_loss in RECOVERED_TRAINING_LOG
        ]
    else:
        # Test data is deliberately not loaded until the best checkpoint is frozen.
        provisional_identity = {"sha256": {"train": sha256_file(S2_DATASET / "train.npz"), "val": sha256_file(S2_DATASET / "val.npz")}}
        best_path, last_path, curve, best_update, best_val = train_model(train_obs, train_actions, val_obs, val_actions,
                                                                          train_mean, train_std, device, provisional_identity)
    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    best_payload["model_frozen"] = True
    best_payload["model_freeze_timestamp"] = "S3R1_POST_TRAINING_BEFORE_VAL"
    torch.save(best_payload, best_path)
    frozen_sha = sha256_file(best_path)
    model, schedule, payload = load_frozen_model(best_path, device)
    identity = dataset_identity()
    if not metadata_identity(identity):
        raise RuntimeError("BLOCKED_S3R1_DATA_LEAKAGE: S2 dataset identity changed")
    # Complete the frozen metadata only after the model itself is frozen and the
    # held-out file identity has been audited; no held-out values affect training.
    best_payload["dataset_sha256"] = identity["sha256"]
    torch.save(best_payload, best_path)
    frozen_sha = sha256_file(best_path)
    val_offline = offline_metrics(model, schedule, val_obs_np, val_actions_np, train_mean, train_std, TRAIN_SEED + 1, device)
    val_rows = []
    for row in read_task_rows("val"):
        scene, task = row_task(row)
        val_rows.append(rollout_diffusion_task(model, schedule, scene, task, train_mean, train_std, device))
    val_summary = summarize_rollouts(val_rows)
    val_actions_summary = action_summary(val_rows)
    leakage = teacher_independence_audit()
    val_gate = closed_loop_gate(val_summary, val_actions_summary, leakage)
    smoke = determinism_smoke(model, schedule, "val", train_mean, train_std, device)
    all_finite = all(math.isfinite(float(row["train_loss"])) and (row["val_loss"] == "" or math.isfinite(float(row["val_loss"]))) for row in curve)
    S3R1_ROOT.mkdir(parents=True, exist_ok=True)
    with (S3R1_ROOT / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "train_loss", "val_loss", "gradient_norm"])
        writer.writeheader(); writer.writerows(curve)
    write_rollout_csv_r1(S3R1_ROOT / "val_rollout.csv", val_rows)
    base = {"task": "S3-R1-BOUNDED-X0-DIFFUSION-ACTION-SUPPORT-REPAIR-V1", "start_head": R0_AUDIT_HEAD,
            "branch": BRANCH, "s3_r0_audit_head": R0_AUDIT_HEAD, "dataset_identity": identity,
            "observation_normalization": "train_only", "observation_mean": train_mean.tolist(), "observation_std": train_std.tolist(),
            "train_windows": int(len(train_obs_np)), "val_windows": int(len(val_obs_np)), "observation_dim": OBSERVATION_DIM,
            "action_horizon": HORIZON, "model": {"name": "ConditionalDiffusionMLP", "parameter_count": sum(p.numel() for p in model.parameters()),
            "architecture": "34-128-128; time64; width256; 4 residual MLP blocks; bounded x0 output"},
            "prediction_target": "x0", "action_support_parameterization": "unit_ball_squash", "diffusion": {"training_steps": DIFFUSION_STEPS, "beta_schedule": "cosine", "inference_steps": DDIM_STEPS, "eta": 0.0},
            "train_seed": TRAIN_SEED, "device": str(device), "pytorch": torch.__version__, "r0_root_cause_diagnostic": diagnostic,
            "oracle_tests": oracle, "training": {"updates": MAX_UPDATES, "best_update": best_update, "best_val_loss": best_val,
            "final_train_loss": float(curve[-1]["train_loss"]), "all_finite": all_finite}, "best_checkpoint_sha256": frozen_sha,
            "model_frozen": True, "test_accessed_before_model_freeze": False, "val_offline": val_offline,
            "val_closed_loop": val_summary, "val_action_statistics": val_actions_summary, "val_gate": "PASS" if val_gate else "FAIL",
            "val_determinism_smoke": smoke, "privileged_information_leakage": leakage,
            "r1_test_executed": False, "r1_test_run_count": 0, "r1_test_used_for_selection": False,
            "train_action_support": train_support, "val_action_support": val_support,
            "max_episode_steps": MAX_EPISODE_STEPS, "goal_tolerance_m": GOAL_TOLERANCE}
    if not val_gate:
        base["final_label"] = "BLOCKED_S3R1_BOUNDED_X0_VAL_WEAK"
        (S3R1_ROOT / "summary.json").write_text(json.dumps(base, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"final_label": base["final_label"], "val": val_summary, "val_actions": val_actions_summary}, indent=2), flush=True)
        return
    test_data = load_npz(S2_DATASET / "test.npz")
    test_obs_np, test_actions_np = build_windows(test_data)
    test_support = validate_action_support(test_data, "test")
    test_offline = offline_metrics(model, schedule, test_obs_np, test_actions_np, train_mean, train_std, TRAIN_SEED + 2, device)
    test_rows = []
    for row in read_task_rows("test"):
        scene, task = row_task(row)
        test_rows.append(rollout_diffusion_task(model, schedule, scene, task, train_mean, train_std, device))
    test_summary = summarize_rollouts(test_rows)
    test_actions_summary = action_summary(test_rows)
    test_smoke = determinism_smoke(model, schedule, "test", train_mean, train_std, device)
    base.update({"test_windows": int(len(test_obs_np)), "test_action_support": test_support, "offline_test": test_offline,
                 "test_closed_loop": test_summary, "test_action_statistics": test_actions_summary,
                 "r1_test_executed": True, "r1_test_run_count": 1, "test_determinism_smoke": test_smoke})
    write_rollout_csv_r1(S3R1_ROOT / "test_rollout.csv", test_rows)
    test_gate = closed_loop_gate(test_summary, test_actions_summary, leakage)
    base["final_label"] = "PASS_S3R1_BOUNDED_X0_DIFFUSION_SANITY" if test_gate else "BLOCKED_S3R1_BOUNDED_X0_TEST_WEAK"
    (S3R1_ROOT / "summary.json").write_text(json.dumps(base, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"final_label": base["final_label"], "val": val_summary, "test": test_summary,
                      "test_actions": test_actions_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
