"""Train and freeze the preregistered matched H16 BC, then evaluate on S2 VAL."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)] + ([str(EXTERNAL)] if EXTERNAL.exists() else [])

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from quadrotor_diffusion_ppo.bc.model import MatchedSequenceBC
from quadrotor_diffusion_ppo.ppo.bc import FrozenBCPrior, sha256_file
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv
from quadrotor_diffusion_ppo.ppo.residual import stable_prior_seed
from scripts.run_s3_diffusion import build_windows, load_npz, observation_normalization
from scripts.run_s4r2_ppo import aggregate, load_tasks
from quadrotor_diffusion_ppo.envs.scene import load_all_scenes


TASK = "S7-R0-MATCHED-BC-ABLATION-AND-FRESH-TOPOLOGY-GENERALIZATION-V1"
TRAIN_SEED = 20260810
BATCH_SIZE = 512
MAX_UPDATES = 10_000
VALIDATION_INTERVAL = 500
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
HORIZON = 16
MODEL_CONFIG = {"observation_dim": 34, "horizon": 16, "action_dim": 3,
                "observation_width": 128, "width": 256, "residual_blocks": 4}
S2 = ROOT / "artifacts" / "s2" / "dataset"
RECOVERY = ROOT / "artifacts" / "s3r2" / "recovery_dataset" / "recovery_train.npz"
OUT = ROOT / "artifacts" / "s7"
CHECKPOINTS = ROOT / "checkpoints" / "s7" / "bc"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def set_deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validation_loss(model, observations, actions) -> float:
    model.eval(); total = 0.0
    for start in range(0, len(observations), BATCH_SIZE):
        prediction = model(observations[start:start + BATCH_SIZE])
        loss = torch.nn.functional.smooth_l1_loss(prediction, actions[start:start + BATCH_SIZE])
        total += float(loss.item()) * len(prediction)
    return total / len(observations)


def checkpoint_payload(model, mean, std, identities, best_update, best_val, frozen):
    return {
        "task": TASK, "model_state": model.state_dict(), "model_config": MODEL_CONFIG,
        "model_frozen": bool(frozen), "horizon": 16, "action_dim": 3,
        "action_support": "per_step_unit_ball_squash",
        "deployment": "receding_horizon_first_action",
        "observation_mean": mean, "observation_std": std,
        "dataset_identity": identities, "train_seed": TRAIN_SEED,
        "loss": "SmoothL1", "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "max_updates": MAX_UPDATES, "validation_interval": VALIDATION_INTERVAL,
        "best_update": int(best_update), "best_val_loss": float(best_val),
        "test_accessed": False, "fresh_holdout_accessed": False,
    }


def train() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_S7_CUDA_UNAVAILABLE")
    set_deterministic(TRAIN_SEED)
    device = torch.device("cuda")
    original = load_npz(S2 / "train.npz"); recovery = load_npz(RECOVERY); val = load_npz(S2 / "val.npz")
    original_obs, original_actions = build_windows(original)
    recovery_obs, recovery_actions = build_windows(recovery)
    val_obs, val_actions = build_windows(val)
    mean, std = observation_normalization(original)
    train_obs = np.concatenate((original_obs, recovery_obs), axis=0)
    train_actions = np.concatenate((original_actions, recovery_actions), axis=0)
    identities = {
        "s2_train_sha256": sha256_file(S2 / "train.npz"),
        "s2_val_sha256": sha256_file(S2 / "val.npz"),
        "s3r2_recovery_train_sha256": sha256_file(RECOVERY),
        "original_train_windows": len(original_obs), "recovery_train_windows": len(recovery_obs),
        "total_train_windows": len(train_obs), "val_windows": len(val_obs),
        "normalization_source": "S2_TRAIN_ONLY", "test_accessed": False,
    }
    train_obs_t = torch.as_tensor((train_obs - mean) / np.maximum(std, 1e-6), device=device)
    train_act_t = torch.as_tensor(train_actions, device=device)
    val_obs_t = torch.as_tensor((val_obs - mean) / np.maximum(std, 1e-6), device=device)
    val_act_t = torch.as_tensor(val_actions, device=device)
    model = MatchedSequenceBC(**MODEL_CONFIG).to(device)
    diffusion_params = 621_360
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 592_688:
        raise RuntimeError("BLOCKED_S7_BC_PARAMETER_COUNT")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device=device).manual_seed(TRAIN_SEED)
    best_val = float("inf"); best_update = 0; curve = []
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for update in range(1, MAX_UPDATES + 1):
        model.train()
        indices = torch.randint(0, len(train_obs_t), (BATCH_SIZE,), generator=generator, device=device)
        prediction = model(train_obs_t[indices])
        loss = torch.nn.functional.smooth_l1_loss(prediction, train_act_t[indices])
        if not torch.isfinite(loss): raise RuntimeError("BLOCKED_S7_BC_NONFINITE_LOSS")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_norm = float(clip_grad_norm_(model.parameters(), 1.0).item()); optimizer.step()
        row = {"update": update, "train_loss": float(loss.item()), "val_loss": "",
               "gradient_norm": gradient_norm}
        if update == 1 or update % VALIDATION_INTERVAL == 0:
            score = validation_loss(model, val_obs_t, val_act_t); row["val_loss"] = score
            if score < best_val:
                best_val, best_update = score, update
                torch.save(checkpoint_payload(model, mean, std, identities, best_update, best_val, False),
                           CHECKPOINTS / "best.pt")
        curve.append(row)
        if update % 1000 == 0:
            print(f"BC update={update} train={loss.item():.7f} best_val={best_val:.7f}", flush=True)
    wall = time.perf_counter() - started
    payload = torch.load(CHECKPOINTS / "best.pt", map_location="cpu", weights_only=False)
    payload["model_frozen"] = True
    torch.save(payload, CHECKPOINTS / "best.pt")
    torch.save(checkpoint_payload(model, mean, std, identities, MAX_UPDATES, best_val, True),
               CHECKPOINTS / "last.pt")
    write_csv(OUT / "bc_training_curve.csv", curve)
    result = {"task": TASK, "status": "BC_TRAINING_FROZEN", "architecture_frozen_before_training": True,
              "architecture": MODEL_CONFIG, "parameter_count": parameter_count,
              "diffusion_parameter_count": diffusion_params,
              "parameter_ratio": parameter_count / diffusion_params,
              "training_contract": {"loss": "SmoothL1", "optimizer": "AdamW", "lr": LEARNING_RATE,
                                    "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                                    "updates": MAX_UPDATES, "validation_interval": VALIDATION_INTERVAL,
                                    "gradient_clip": 1.0, "seed": TRAIN_SEED},
              "dataset_identity": identities, "best_update": best_update, "best_val_loss": best_val,
              "best_checkpoint_sha256": sha256_file(CHECKPOINTS / "best.pt"),
              "training_wall_time_s": wall, "test_accessed": False, "fresh_holdout_accessed": False}
    write_json(OUT / "bc_training_summary.json", result)
    return result


def evaluate_bc(prior: FrozenBCPrior, tasks, scenes) -> tuple[list[dict], dict]:
    rows = []
    for task in tasks:
        env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)
        observation, _ = env.reset(seed=stable_prior_seed(task.task_id)); total_return = 0.0
        terminated = truncated = False; info = {}
        while not (terminated or truncated):
            action = prior.predict(np.asarray(observation).reshape(1, 34))[0]
            observation, reward, terminated, truncated, info = env.step(action); total_return += reward
        rows.append({"task_id": task.task_id, "scene_id": task.scene_id,
                     "family": task.scene_id.split("_")[0], "success": bool(info.get("success")),
                     "collision": bool(info.get("collision")), "ground_contact": bool(info.get("ground_contact")),
                     "unsafe": bool(info.get("collision") or info.get("ground_contact")),
                     "timeout": bool(info.get("timeout")), "nonfinite": bool(info.get("nonfinite")),
                     "return": total_return, "episode_steps": int(info.get("episode_steps", 0)),
                     "action_clip_count": int(info.get("action_clip_count", 0))})
        env.close()
    return rows, aggregate(rows)


def evaluate() -> dict:
    prior = FrozenBCPrior(CHECKPOINTS / "best.pt", "cuda")
    tasks = load_tasks(); scenes = {scene.scene_id: scene for scene in load_all_scenes()}
    rows, summary = evaluate_bc(prior, tasks["val"], scenes)
    write_csv(OUT / "bc_only_val_rollouts.csv", rows)
    result = {"task": TASK, "status": "BC_ONLY_VAL_COMPLETE", "checkpoint_sha256": prior.checkpoint_sha256,
              "frozen": prior.frozen, "deployment": "H16_RECEDING_HORIZON_FIRST_ACTION",
              "val": summary, "diffusion_reference_val_success": 37,
              "test_accessed": False, "fresh_holdout_accessed": False}
    write_json(OUT / "bc_only_val_summary.json", result)
    return result


def main() -> None:
    training = train(); result = evaluate()
    print(json.dumps({"training": training, "evaluation": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
