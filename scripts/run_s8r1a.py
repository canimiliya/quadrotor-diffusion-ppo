"""S8-R1A frozen offline expert-data budget ablation.

This script intentionally contains no PPO path.  It creates task-level nested
manifests, trains the frozen S7 matched BC and S3-R2 bounded-x0 Diffusion
architectures on the selected TRAIN episodes, and evaluates the frozen
checkpoints on current VAL and the existing S7 development topology.
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
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT)] + ([str(EXTERNAL)] if EXTERNAL.exists() else [])

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from quadrotor_diffusion_ppo.bc.model import MatchedSequenceBC
from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP, unit_ball_squash
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule
from quadrotor_diffusion_ppo.evaluation.runtime import DiffusionInferenceScheduler, RuntimeMode
from quadrotor_diffusion_ppo.envs.scene import load_all_scenes
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.residual import stable_prior_seed, sha256_file
from scripts.generate_s7_fresh_holdout import canonical_hash, scene_spec


TASK = "S8-R1A-OFFLINE-EXPERT-DATA-BUDGET-BC-VS-DIFFUSION-ABLATION-V1"
START_HEAD = "1b160f5e5ba2d785196ce5ee74e1a10378e6804e"
BRANCH = "agent/s8r1a-offline-data-ablation-v1"
DATA_BUDGET_SEED = 20260817
TRAINING_SEEDS = (20260820, 20260821, 20260822)
FRACTIONS = (25, 50, 100)
HORIZON = 16
ACTION_DIM = 3
OBSERVATION_DIM = 34
DIFFUSION_STEPS = 100
DDIM_STEPS = 10
BATCH_SIZE = 512
MAX_UPDATES = 10_000
VALIDATION_INTERVAL = 500
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
S2_ROOT = ROOT / "artifacts" / "s2"
S2_DATASET = S2_ROOT / "dataset"
RECOVERY_PATH = ROOT / "artifacts" / "s3r2" / "recovery_dataset" / "recovery_train.npz"
OUT = ROOT / "artifacts" / "s8r1a"
CKPT = ROOT / "checkpoints" / "s8r1a"

BC_CONFIG = {
    "observation_dim": 34, "horizon": 16, "action_dim": 3,
    "observation_width": 128, "width": 256, "residual_blocks": 4,
    "parameter_count": 592688,
    "loss": "SmoothL1", "optimizer": "AdamW", "lr": 3e-4,
    "weight_decay": 1e-4, "batch_size": 512, "updates": 10000,
    "gradient_clip": 1.0, "action_support": "per_step_unit_ball_squash",
}
DIFFUSION_CONFIG = {
    "observation_dim": 34, "horizon": 16, "action_dim": 3,
    "observation_width": 128, "time_dim": 64, "width": 256,
    "residual_blocks": 4, "bounded_output": True,
    "prediction_target": "bounded x0", "action_support": "unit_ball_squash",
    "diffusion_steps": 100, "ddim_steps": 10, "ddim_eta": 0.0,
    "beta_schedule": "cosine", "loss": "MSE(x0)", "optimizer": "AdamW",
    "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 512,
    "updates": 10000, "gradient_clip": 1.0,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def load_train_manifest() -> list[dict[str, str]]:
    with (S2_ROOT / "task_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["accepted"].lower() == "true" and row["split"] == "train"]
    if len(rows) != 252:
        raise RuntimeError(f"BLOCKER_TASK_LEVEL_TRAIN_COUNT:{len(rows)}")
    return rows


def dataset_episode_pairs(data: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    return [(int(data["scene_indices"][int(offset)]), int(data["task_indices"][int(offset)]))
            for offset in data["episode_offsets"]]


def filtered_windows(data: dict[str, np.ndarray], selected_pairs: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int, int]:
    obs_windows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    episode_count = 0
    transition_count = 0
    for offset_raw, length_raw, pair in zip(data["episode_offsets"], data["episode_lengths"], dataset_episode_pairs(data)):
        offset, length = int(offset_raw), int(length_raw)
        if pair not in selected_pairs:
            continue
        episode_count += 1
        transition_count += length
        if length < HORIZON:
            continue
        for start in range(offset, offset + length - HORIZON + 1):
            obs_windows.append(data["observations"][start])
            action_windows.append(data["actions"][start:start + HORIZON])
    if not obs_windows:
        raise RuntimeError("BLOCKER_NO_VALID_WINDOWS")
    return (np.asarray(obs_windows, dtype=np.float32),
            np.asarray(action_windows, dtype=np.float32), episode_count, transition_count)


def full_windows(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    obs, actions, _, _ = filtered_windows(data, set(dataset_episode_pairs(data)))
    return obs, actions


def make_manifests() -> dict[int, dict[str, Any]]:
    rows = load_train_manifest()
    scene_to_ids: dict[str, list[str]] = {}
    pair_to_id: dict[tuple[int, int], str] = {}
    scene_index = {scene.scene_id: i for i, scene in enumerate(load_all_scenes())}
    for row in rows:
        scene_to_ids.setdefault(row["scene_id"], []).append(row["task_id"])
        pair_to_id[(scene_index[row["scene_id"]], int(row["candidate_id"]))] = row["task_id"]
    if sorted(len(v) for v in scene_to_ids.values()) != [28] * 9:
        raise RuntimeError("BLOCKER_SCENE_TASK_COUNTS")
    base = load_npz(S2_DATASET / "train.npz")
    recovery = load_npz(RECOVERY_PATH)
    base_pairs = set(dataset_episode_pairs(base)); recovery_pairs = set(dataset_episode_pairs(recovery))
    source_hashes = {"s2_train": sha256_file(S2_DATASET / "train.npz"), "s3r2_recovery_train": sha256_file(RECOVERY_PATH)}
    manifests: dict[int, dict[str, Any]] = {}
    rng = np.random.Generator(np.random.PCG64(DATA_BUDGET_SEED))
    selected_by_fraction: dict[int, set[str]] = {}
    selected_pairs_by_fraction: dict[int, set[tuple[int, int]]] = {}
    for scene_id in sorted(scene_to_ids):
        ids = sorted(scene_to_ids[scene_id])
        permutation = rng.permutation(ids)
        for fraction, count in ((25, 7), (50, 14), (100, 28)):
            selected_by_fraction.setdefault(fraction, set()).update(str(x) for x in permutation[:count])
    for fraction in FRACTIONS:
        selected_ids = selected_by_fraction[fraction]
        selected_pairs = {pair for pair, task_id in pair_to_id.items() if task_id in selected_ids}
        selected_pairs_by_fraction[fraction] = selected_pairs
        base_obs, base_actions, base_episodes, base_transitions = filtered_windows(base, selected_pairs)
        recovery_obs, recovery_actions, recovery_episodes, recovery_transitions = filtered_windows(recovery, selected_pairs)
        scene_selected = {scene: sorted(task for task in ids if task in selected_ids) for scene, ids in sorted(scene_to_ids.items())}
        recovery_ids = sorted(pair_to_id[pair] for pair in recovery_pairs if pair in selected_pairs and pair in pair_to_id)
        payload = {
            "task": TASK, "fraction": fraction, "seed": DATA_BUDGET_SEED,
            "scene_wise_selected_task_ids": scene_selected,
            "selected_task_ids": sorted(selected_ids), "recovery_task_ids": recovery_ids,
            "total_task_count": len(selected_ids), "base_episode_count": base_episodes,
            "recovery_episode_count": recovery_episodes,
            "base_transition_count": base_transitions, "recovery_transition_count": recovery_transitions,
            "base_window_count": len(base_obs), "recovery_window_count": len(recovery_obs),
            "total_window_count": len(base_obs) + len(recovery_obs),
            "source_hashes": source_hashes,
            "normalization_source": "S2_TRAIN_FULL_OBSERVATION_ONLY",
            "subset_nesting": "PASS", "scene_balance": "PASS", "task_level_split": "PASS",
        }
        payload["sha256"] = object_hash(payload)
        write_json(OUT / f"data_budget_{fraction}.json", payload)
        manifests[fraction] = payload
    if not (selected_by_fraction[25] < selected_by_fraction[50] < selected_by_fraction[100]):
        raise RuntimeError("BLOCKER_SUBSET_NESTING")
    return manifests


def load_manifests() -> dict[int, dict[str, Any]]:
    return {fraction: json.loads((OUT / f"data_budget_{fraction}.json").read_text(encoding="utf-8")) for fraction in FRACTIONS}


def set_deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def normalization() -> tuple[np.ndarray, np.ndarray]:
    data = load_npz(S2_DATASET / "train.npz")
    mean = data["observations"].astype(np.float64).mean(axis=0).astype(np.float32)
    std = data["observations"].astype(np.float64).std(axis=0).astype(np.float32)
    return mean, std


def tensors(obs: np.ndarray, actions: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device):
    normalized = (obs - mean) / np.maximum(std, 1e-6)
    return (torch.from_numpy(normalized.astype(np.float32)).to(device),
            torch.from_numpy(actions.astype(np.float32)).to(device))


def bc_payload(model: MatchedSequenceBC, mean: np.ndarray, std: np.ndarray, manifest: dict[str, Any], seed: int,
               best_update: int, best_val: float, frozen: bool) -> dict[str, Any]:
    return {"task": TASK, "model_state": model.state_dict(), "model_config": {k: v for k, v in BC_CONFIG.items() if k in (
        "observation_dim", "horizon", "action_dim", "observation_width", "width", "residual_blocks")},
            "model_frozen": bool(frozen), "horizon": 16, "action_dim": 3,
            "action_support": "per_step_unit_ball_squash", "deployment": "receding_horizon_first_action",
            "observation_mean": mean, "observation_std": std, "dataset_manifest_sha256": manifest["sha256"],
            "train_seed": seed, "loss": "SmoothL1", "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_updates": MAX_UPDATES,
            "validation_interval": VALIDATION_INTERVAL, "best_update": int(best_update),
            "best_val_loss": float(best_val), "test_accessed": False, "s6_test_accessed": False,
            "parameter_count": sum(p.numel() for p in model.parameters())}


def diffusion_payload(model: ConditionalDiffusionMLP, mean: np.ndarray, std: np.ndarray, manifest: dict[str, Any], seed: int,
                      best_update: int, best_val: float, frozen: bool) -> dict[str, Any]:
    return {"task": TASK, "model_state": model.state_dict(), "model_config": {
        k: DIFFUSION_CONFIG[k] for k in ("observation_dim", "horizon", "action_dim", "observation_width", "time_dim", "width", "residual_blocks", "bounded_output")},
        "model_frozen": bool(frozen), "horizon": 16, "action_dim": 3,
        "observation_mean": mean, "observation_std": std, "dataset_manifest_sha256": manifest["sha256"],
        "train_seed": seed, "prediction_target": "x0", "action_support": "unit_ball_squash",
        "diffusion_steps": DIFFUSION_STEPS, "beta_schedule": "cosine", "ddim_steps": DDIM_STEPS,
        "ddim_eta": 0.0, "loss": "MSE(x0)", "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_updates": MAX_UPDATES,
        "validation_interval": VALIDATION_INTERVAL, "best_update": int(best_update),
        "best_val_loss": float(best_val), "test_accessed": False, "s6_test_accessed": False,
        "parameter_count": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def bc_val_loss(model: MatchedSequenceBC, obs: torch.Tensor, actions: torch.Tensor) -> float:
    model.eval(); total = 0.0
    for start in range(0, len(obs), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        loss = torch.nn.functional.smooth_l1_loss(model(obs[batch]), actions[batch])
        total += float(loss.item()) * len(obs[batch])
    return total / len(obs)


@torch.no_grad()
def diffusion_val_loss(model: ConditionalDiffusionMLP, schedule: DiffusionSchedule, obs: torch.Tensor,
                       actions: torch.Tensor, seed: int) -> float:
    model.eval(); generator = torch.Generator(device=obs.device).manual_seed(seed + 991); total = 0.0
    for start in range(0, len(obs), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE); target = actions[batch]
        ts = torch.randint(0, DIFFUSION_STEPS, (len(target),), generator=generator, device=obs.device)
        noise = torch.randn(target.shape, generator=generator, device=obs.device)
        loss = torch.nn.functional.mse_loss(model(schedule.add_noise(target, noise, ts), obs[batch], ts), target)
        total += float(loss.item()) * len(target)
    return total / len(obs)


def train_bc(fraction: int, seed: int, device: torch.device, manifests: dict[int, dict[str, Any]], smoke: bool = False) -> dict[str, Any]:
    manifest = manifests[fraction]
    run = OUT / "runs" / f"bc_{fraction}_seed_{seed}"; checkpoint_dir = CKPT / f"bc_{fraction}_seed_{seed}"; best_path = checkpoint_dir / "best.pt"
    if best_path.exists() and (run / "training_manifest.json").exists(): return json.loads((run / "training_manifest.json").read_text(encoding="utf-8"))
    set_deterministic(seed); mean, std = normalization(); base = load_npz(S2_DATASET / "train.npz"); recovery = load_npz(RECOVERY_PATH); val = load_npz(S2_DATASET / "val.npz")
    selected = set(manifest["selected_task_ids"]); rows = load_train_manifest(); pair_map = {(int(next(i for i,s in enumerate(sorted({r['scene_id'] for r in rows})) if s == r['scene_id'])), int(r['candidate_id'])): r['task_id'] for r in rows}
    # Scene indices are the stable order used by the frozen scene loader.
    scene_index = {scene.scene_id: i for i, scene in enumerate(load_all_scenes())}; pair_map = {(scene_index[r['scene_id']], int(r['candidate_id'])): r['task_id'] for r in rows}
    pairs = {pair for pair, task_id in pair_map.items() if task_id in selected}
    train_obs, train_act, _, _ = filtered_windows(base, pairs); rec_obs, rec_act, _, _ = filtered_windows(recovery, pairs); train_obs = np.concatenate([train_obs, rec_obs]); train_act = np.concatenate([train_act, rec_act])
    val_obs, val_act = full_windows(val); train_t = tensors(train_obs, train_act, mean, std, device); val_t = tensors(val_obs, val_act, mean, std, device)
    model = MatchedSequenceBC(**{k: BC_CONFIG[k] for k in ("observation_dim", "horizon", "action_dim", "observation_width", "width", "residual_blocks")}).to(device)
    if sum(p.numel() for p in model.parameters()) != 592688: raise RuntimeError("BLOCKER_BC_PARAMETER_COUNT")
    if smoke:
        model.train(); gen = torch.Generator(device=device).manual_seed(seed); idx = torch.randint(0, len(train_t[0]), (BATCH_SIZE,), generator=gen, device=device); loss = torch.nn.functional.smooth_l1_loss(model(train_t[0][idx]), train_t[1][idx]);
        if not torch.isfinite(loss): raise RuntimeError("BLOCKER_BC_SMOKE_NONFINITE")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY); gen = torch.Generator(device=device).manual_seed(seed); best_val = float("inf"); best_update = 0; curve = []; checkpoint_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    for update in range(1, MAX_UPDATES + 1):
        model.train(); idx = torch.randint(0, len(train_t[0]), (BATCH_SIZE,), generator=gen, device=device); pred = model(train_t[0][idx]); loss = torch.nn.functional.smooth_l1_loss(pred, train_t[1][idx])
        if not torch.isfinite(loss): raise RuntimeError("BLOCKER_BC_NONFINITE_LOSS")
        optimizer.zero_grad(set_to_none=True); loss.backward(); grad = float(clip_grad_norm_(model.parameters(), GRAD_CLIP).item()); optimizer.step()
        row = {"update": update, "train_loss": float(loss.item()), "val_loss": "", "gradient_norm": grad}
        if update == 1 or update % VALIDATION_INTERVAL == 0:
            score = bc_val_loss(model, val_t[0], val_t[1]); row["val_loss"] = score
            if score < best_val:
                best_val, best_update = score, update; torch.save(bc_payload(model, mean, std, manifest, seed, best_update, best_val, False), best_path)
        curve.append(row)
        if update % 1000 == 0: print(f"BC fraction={fraction} seed={seed} update={update} train={loss.item():.7f} best_val={best_val:.7f}", flush=True)
    last_path = checkpoint_dir / "last.pt"; torch.save(bc_payload(model, mean, std, manifest, seed, MAX_UPDATES, best_val, True), last_path); payload = torch.load(best_path, map_location="cpu", weights_only=False); payload["model_frozen"] = True; torch.save(payload, best_path)
    result = {"task": TASK, "model": "BC", "fraction": fraction, "seed": seed, "status": "TRAINED_FROZEN", "best_update": best_update, "best_val_loss": best_val, "best_checkpoint_sha256": sha256_file(best_path), "last_checkpoint_sha256": sha256_file(last_path), "training_wall_time_s": time.perf_counter() - started, "dataset_manifest_sha256": manifest["sha256"], "test_accessed": False, "s6_test_accessed": False, "config_hash": object_hash(BC_CONFIG), "parameter_count": 592688}
    write_csv(run / "training_curve.csv", curve); write_json(run / "training_manifest.json", result); return result


def train_diffusion(fraction: int, seed: int, device: torch.device, manifests: dict[int, dict[str, Any]], smoke: bool = False) -> dict[str, Any]:
    manifest = manifests[fraction]; run = OUT / "runs" / f"diffusion_{fraction}_seed_{seed}"; checkpoint_dir = CKPT / f"diffusion_{fraction}_seed_{seed}"; best_path = checkpoint_dir / "best.pt"
    if best_path.exists() and (run / "training_manifest.json").exists(): return json.loads((run / "training_manifest.json").read_text(encoding="utf-8"))
    set_deterministic(seed); mean, std = normalization(); base = load_npz(S2_DATASET / "train.npz"); recovery = load_npz(RECOVERY_PATH); val = load_npz(S2_DATASET / "val.npz"); rows = load_train_manifest(); scene_index = {scene.scene_id: i for i, scene in enumerate(load_all_scenes())}; pairs = {(scene_index[r['scene_id']], int(r['candidate_id'])) for r in rows if r['task_id'] in set(manifest['selected_task_ids'])}; train_obs, train_act, _, _ = filtered_windows(base, pairs); rec_obs, rec_act, _, _ = filtered_windows(recovery, pairs); train_obs = np.concatenate([train_obs, rec_obs]); train_act = np.concatenate([train_act, rec_act]); val_obs, val_act = full_windows(val); train_t = tensors(train_obs, train_act, mean, std, device); val_t = tensors(val_obs, val_act, mean, std, device)
    model = ConditionalDiffusionMLP(**{k: DIFFUSION_CONFIG[k] for k in ("observation_dim", "horizon", "action_dim", "observation_width", "time_dim", "width", "residual_blocks", "bounded_output")}).to(device); schedule = DiffusionSchedule(DIFFUSION_STEPS).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY); gen = torch.Generator(device=device).manual_seed(seed); best_val = float("inf"); best_update = 0; curve = []; checkpoint_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    if smoke:
        idx = torch.randint(0, len(train_t[0]), (BATCH_SIZE,), generator=gen, device=device); ts = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), generator=gen, device=device); noise = torch.randn(train_t[1][idx].shape, generator=gen, device=device); out = model(schedule.add_noise(train_t[1][idx], noise, ts), train_t[0][idx], ts); loss = torch.nn.functional.mse_loss(out, train_t[1][idx]);
        if not torch.isfinite(loss) or float(torch.linalg.vector_norm(out, dim=-1).max()) > 1.0 + 1e-5: raise RuntimeError("BLOCKER_DIFFUSION_SMOKE")
    for update in range(1, MAX_UPDATES + 1):
        model.train(); idx = torch.randint(0, len(train_t[0]), (BATCH_SIZE,), generator=gen, device=device); ts = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), generator=gen, device=device); noise = torch.randn(train_t[1][idx].shape, generator=gen, device=device); pred = model(schedule.add_noise(train_t[1][idx], noise, ts), train_t[0][idx], ts); loss = torch.nn.functional.mse_loss(pred, train_t[1][idx])
        if not torch.isfinite(loss): raise RuntimeError("BLOCKER_DIFFUSION_NONFINITE_LOSS")
        optimizer.zero_grad(set_to_none=True); loss.backward(); grad = float(clip_grad_norm_(model.parameters(), GRAD_CLIP).item()); optimizer.step(); row = {"update": update, "train_loss": float(loss.item()), "val_loss": "", "gradient_norm": grad}
        if update == 1 or update % VALIDATION_INTERVAL == 0:
            score = diffusion_val_loss(model, schedule, val_t[0], val_t[1], seed); row["val_loss"] = score
            if score < best_val:
                best_val, best_update = score, update; torch.save(diffusion_payload(model, mean, std, manifest, seed, best_update, best_val, False), best_path)
        curve.append(row)
        if update % 1000 == 0: print(f"Diffusion fraction={fraction} seed={seed} update={update} train={loss.item():.7f} best_val={best_val:.7f}", flush=True)
    last_path = checkpoint_dir / "last.pt"; torch.save(diffusion_payload(model, mean, std, manifest, seed, MAX_UPDATES, best_val, True), last_path); payload = torch.load(best_path, map_location="cpu", weights_only=False); payload["model_frozen"] = True; torch.save(payload, best_path)
    result = {"task": TASK, "model": "Diffusion", "fraction": fraction, "seed": seed, "status": "TRAINED_FROZEN", "best_update": best_update, "best_val_loss": best_val, "best_checkpoint_sha256": sha256_file(best_path), "last_checkpoint_sha256": sha256_file(last_path), "training_wall_time_s": time.perf_counter() - started, "dataset_manifest_sha256": manifest["sha256"], "test_accessed": False, "s6_test_accessed": False, "config_hash": object_hash(DIFFUSION_CONFIG), "parameter_count": sum(p.numel() for p in model.parameters()), "runtime_contract": "FAST_CUDA_GRAPH_EXACT_BATCH_ONE"}
    write_csv(run / "training_curve.csv", curve); write_json(run / "training_manifest.json", result); return result


class BudgetDiffusionPrior:
    """The unchanged S3-R2 bounded-x0 model with the audited exact sampler."""
    def __init__(self, checkpoint: Path, device: str | torch.device = "cuda"):
        self.checkpoint = checkpoint; self.checkpoint_sha256 = sha256_file(checkpoint); self.device = torch.device(device); self.payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        required = {"model_frozen": True, "prediction_target": "x0", "action_support": "unit_ball_squash", "diffusion_steps": 100, "ddim_steps": 10, "ddim_eta": 0.0}
        if any(self.payload.get(k) != v for k, v in required.items()): raise RuntimeError("BLOCKER_DIFFUSION_CHECKPOINT_METADATA")
        self.model = ConditionalDiffusionMLP(**self.payload["model_config"]).to(self.device); self.model.load_state_dict(self.payload["model_state"]); self.model.eval().requires_grad_(False); self.schedule = DiffusionSchedule(100).to(self.device); self.mean = torch.as_tensor(self.payload["observation_mean"], dtype=torch.float32, device=self.device); self.scale = torch.clamp(torch.as_tensor(self.payload["observation_std"], dtype=torch.float32, device=self.device), min=1e-6)
        ts = torch.linspace(99, 0, 10, device=self.device).round().long(); self._ts = tuple(int(x) for x in ts.cpu().tolist()); self._tb = tuple(torch.full((1,), x, dtype=torch.long, device=self.device) for x in self._ts); self._alpha = tuple(self.schedule.alpha_bars[x] for x in self._ts); self._graph = None; self._graph_condition = None; self._graph_initial = None; self._graph_output = None
    def _conditions(self, observations: np.ndarray) -> torch.Tensor:
        value = torch.as_tensor(observations, dtype=torch.float32, device=self.device); value = value.unsqueeze(0) if value.ndim == 1 else value; result = (value - self.mean) / self.scale
        if not torch.isfinite(result).all(): raise FloatingPointError("nonfinite normalized diffusion observation")
        return result
    def _initial_noise(self, seed: int) -> torch.Tensor:
        return torch.randn((1, self.model.horizon, self.model.action_dim), device=self.device, generator=torch.Generator(device=self.device).manual_seed(int(seed)))
    def _ddim(self, condition: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        for i, (t, tb, alpha) in enumerate(zip(self._ts, self._tb, self._alpha)):
            x0 = unit_ball_squash(self.model.predict_raw(sample, condition, tb)); epsilon = (sample - alpha.sqrt() * x0) / (1.0 - alpha).sqrt()
            if i == len(self._ts) - 1: sample = x0
            else:
                nxt = self._alpha[i + 1]; sample = nxt.sqrt() * x0 + (1.0 - nxt).sqrt() * epsilon
        return sample
    @torch.inference_mode()
    def predict_reference(self, observations: np.ndarray, seeds: list[int]) -> np.ndarray:
        conditions = self._conditions(observations); return torch.stack([self._ddim(conditions[i:i+1], self._initial_noise(seed))[0, 0] for i, seed in enumerate(seeds)]).cpu().numpy().astype(np.float32)
    @torch.inference_mode()
    def _capture(self) -> None:
        self._graph_condition = torch.empty((1, 34), dtype=torch.float32, device=self.device); self._graph_initial = torch.empty((1, 16, 3), dtype=torch.float32, device=self.device); stream = torch.cuda.Stream(device=self.device); stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3): self._ddim(self._graph_condition, self._graph_initial)
        torch.cuda.current_stream(self.device).wait_stream(stream); torch.cuda.synchronize(self.device); self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph): self._graph_output = self._ddim(self._graph_condition, self._graph_initial)
    @torch.inference_mode()
    def predict_fast(self, observations: np.ndarray, seeds: list[int]) -> np.ndarray:
        if self.device.type != "cuda": return self.predict_reference(observations, seeds)
        conditions = self._conditions(observations)
        if self._graph is None: self._capture()
        actions = []
        for i, seed in enumerate(seeds): self._graph_condition.copy_(conditions[i:i+1]); self._graph_initial.copy_(self._initial_noise(seed)); self._graph.replay(); actions.append(self._graph_output[0, 0].clone())
        return torch.stack(actions).cpu().numpy().astype(np.float32)


class BCPolicy:
    def __init__(self, checkpoint: Path, device: torch.device):
        self.payload = torch.load(checkpoint, map_location=device, weights_only=False); self.device = device; self.model = MatchedSequenceBC(**self.payload["model_config"]).to(device); self.model.load_state_dict(self.payload["model_state"]); self.model.eval().requires_grad_(False); self.mean = torch.as_tensor(self.payload["observation_mean"], dtype=torch.float32, device=device); self.scale = torch.clamp(torch.as_tensor(self.payload["observation_std"], dtype=torch.float32, device=device), min=1e-6); self.checkpoint_sha256 = sha256_file(checkpoint)
    @torch.inference_mode()
    def predict(self, observation: np.ndarray) -> np.ndarray:
        value = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, 34); return self.model((value - self.mean) / self.scale)[0, 0].cpu().numpy().astype(np.float32)


def current_val_tasks() -> tuple[list[TaskEndpoint], dict[str, Any]]:
    tasks = []
    with (S2_ROOT / "task_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["accepted"].lower() != "true" or row["split"] != "val": continue
            tasks.append(TaskEndpoint(row["scene_id"], row["task_id"], int(row["candidate_seed"]), np.asarray([float(row[f"start_{a}"]) for a in "xyz"]), np.asarray([float(row[f"goal_{a}"]) for a in "xyz"]), "VAL"))
    return tasks, {scene.scene_id: scene for scene in load_all_scenes()}


def fresh_tasks() -> tuple[list[TaskEndpoint], dict[str, Any]]:
    root = OUT.parent / "s7" / "fresh_holdout"; frozen = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8")); topologies = frozen["topologies"]; scenes = {d["scene_id"]: scene_spec(d) for d in topologies}; tasks = []
    with (root / "task_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle): tasks.append(TaskEndpoint(row["scene_id"], row["task_id"], int(row["candidate_seed"]), np.asarray([float(row[f"start_{a}"]) for a in "xyz"]), np.asarray([float(row[f"goal_{a}"]) for a in "xyz"]), "S7_FRESH_DEVELOPMENT"))
    if len(tasks) != 54 or frozen.get("used_for_training_or_selection") is not False: raise RuntimeError("BLOCKER_FRESH_MANIFEST")
    return tasks, scenes


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows)); result = {"tasks": len(rows), "success": int(sum(r["success"] for r in rows)), "success_rate": float(sum(r["success"] for r in rows) / n), "collision": int(sum(r["collision"] for r in rows)), "ground_contact": int(sum(r["ground_contact"] for r in rows)), "unsafe": int(sum(r["unsafe"] for r in rows)), "timeout": int(sum(r["timeout"] for r in rows)), "mean_return": float(np.mean([r["mean_return"] for r in rows])) if rows else 0.0, "by_family": {}}
    for family in ("OPEN", "BLOCK", "SBEND"):
        family_rows = [r for r in rows if r["family"] == family]; fn = max(1, len(family_rows)); result["by_family"][family] = {"tasks": len(family_rows), "success": int(sum(r["success"] for r in family_rows)), "success_rate": float(sum(r["success"] for r in family_rows) / fn)}
    return result


def evaluate_policy(model_name: str, fraction: int, seed: int, split: str, tasks: list[TaskEndpoint], scenes: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    checkpoint = CKPT / f"{model_name.lower()}_{fraction}_seed_{seed}" / "best.pt"; scheduler = None; bc = None
    if model_name == "BC": bc = BCPolicy(checkpoint, device)
    else:
        prior = BudgetDiffusionPrior(checkpoint, device); scheduler = DiffusionInferenceScheduler(prior, RuntimeMode.FAST, max_batch_size=1)
    rows = []
    try:
        for task in tasks:
            env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False); observation, _ = env.reset(seed=stable_prior_seed(task.task_id)); total = 0.0; terminated = truncated = False; info = {}; support = 0
            while not (terminated or truncated):
                action = bc.predict(observation) if bc is not None else scheduler.predict(observation, stable_prior_seed(task.task_id) + env.episode_steps)
                action = np.asarray(action, dtype=np.float32).reshape(3); support += int(np.linalg.norm(action) > 1.0 + 1e-6); observation, reward, terminated, truncated, info = env.step(action); total += float(reward)
            rows.append({"model": model_name, "fraction": fraction, "seed": seed, "split": split, "task_id": task.task_id, "family": task.scene_id.split("_")[0], "success": int(bool(info.get("success"))), "collision": int(bool(info.get("collision"))), "ground_contact": int(bool(info.get("ground_contact"))), "unsafe": int(bool(info.get("collision") or info.get("ground_contact") or info.get("nonfinite"))), "timeout": int(bool(info.get("timeout"))), "mean_return": float(total), "episode_steps": int(info.get("episode_steps", 0)), "action_clip_count": int(info.get("action_clip_count", 0)), "policy_action_support_violation_count": support})
            env.close()
    finally:
        if scheduler is not None: scheduler.close()
    runtime = {"runtime_mode": "BC_DETERMINISTIC" if model_name == "BC" else "FAST_CUDA_GRAPH_EXACT_BATCH_ONE", "historical_batched_used": False, "checkpoint_sha256": sha256_file(checkpoint), "tasks": len(rows)}
    return rows, summarize(rows), runtime


def run_evaluations(manifests: dict[int, dict[str, Any]], device: torch.device) -> dict[str, Any]:
    current, current_scenes = current_val_tasks(); fresh, fresh_scenes = fresh_tasks(); current_rows: list[dict[str, Any]] = []; fresh_rows: list[dict[str, Any]] = []; run_status = []
    for fraction in FRACTIONS:
        for model in ("BC", "Diffusion"):
            for seed in TRAINING_SEEDS:
                train_path = OUT / "runs" / f"{model.lower()}_{fraction}_seed_{seed}" / "training_manifest.json"; status = json.loads(train_path.read_text(encoding="utf-8"));
                if status.get("status") != "TRAINED_FROZEN": raise RuntimeError("BLOCKER_MODEL_NOT_FROZEN")
                c_rows, c_summary, c_runtime = evaluate_policy(model, fraction, seed, "CURRENT_VAL", current, current_scenes, device); f_rows, f_summary, f_runtime = evaluate_policy(model, fraction, seed, "S7_FRESH_DEVELOPMENT", fresh, fresh_scenes, device); current_rows.extend(c_rows); fresh_rows.extend(f_rows); run_status.append({"model": model, "fraction": fraction, "seed": seed, "current": c_summary, "fresh": f_summary, "current_runtime": c_runtime, "fresh_runtime": f_runtime, "checkpoint_sha256": status["best_checkpoint_sha256"]}); print(f"EVAL {model} {fraction}% seed={seed} current={c_summary['success']}/54 fresh={f_summary['success']}/54", flush=True)
    write_csv(OUT / "current_val_metrics.csv", current_rows); write_csv(OUT / "fresh_metrics.csv", fresh_rows); return {"run_status": run_status, "current_rows": current_rows, "fresh_rows": fresh_rows}


def make_tables_and_plots(evidence: dict[str, Any]) -> dict[str, Any]:
    current, fresh = evidence["current_rows"], evidence["fresh_rows"]; paired = []
    for fraction in FRACTIONS:
        for seed in TRAINING_SEEDS:
            bc = next(summarize([r for r in current if r["model"] == "BC" and r["fraction"] == fraction and r["seed"] == seed])["success_rate"] for _ in [0]); diff = summarize([r for r in current if r["model"] == "Diffusion" and r["fraction"] == fraction and r["seed"] == seed])["success_rate"]; paired.append({"fraction": fraction, "seed": seed, "bc_rate": bc, "diffusion_rate": diff, "diffusion_minus_bc": diff - bc})
    write_csv(OUT / "paired_metrics.csv", paired)
    aggregate = {"current": [], "fresh": [], "paired": []}
    for split, values in (("current", current), ("fresh", fresh)):
        for fraction in FRACTIONS:
            for model in ("BC", "Diffusion"):
                rates = [summarize([r for r in values if r["model"] == model and r["fraction"] == fraction and r["seed"] == seed])["success_rate"] for seed in TRAINING_SEEDS]
                aggregate[split].append({"fraction": fraction, "model": model, "mean": float(np.mean(rates)), "sample_std": float(np.std(rates, ddof=1)), "rates": rates, "counts": [int(round(x * 54)) for x in rates]})
    for fraction in FRACTIONS:
        vals = [r for r in paired if r["fraction"] == fraction]; aggregate["paired"].append({"fraction": fraction, "rows": vals, "mean": float(np.mean([x["diffusion_minus_bc"] for x in vals])), "sample_std": float(np.std([x["diffusion_minus_bc"] for x in vals], ddof=1))})
    import matplotlib.pyplot as plt
    for key, filename in (("current", "current_success_vs_data_fraction.png"), ("fresh", "fresh_success_vs_data_fraction.png")):
        plt.figure(figsize=(6.0, 4.2)); x = np.arange(3); 
        for model, color in (("BC", "#1f77b4"), ("Diffusion", "#d62728")):
            vals = [next(v for v in aggregate[key] if v["fraction"] == f and v["model"] == model) for f in FRACTIONS]; plt.errorbar(x, [v["mean"] for v in vals], yerr=[v["sample_std"] for v in vals], marker="o", capsize=3, label=model, color=color)
        plt.xticks(x, ["25%", "50%", "100%"]); plt.ylim(0, 1); plt.ylabel("success rate"); plt.xlabel("expert-data fraction"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT / filename, dpi=180); plt.close()
    plt.figure(figsize=(6.0, 4.2)); x = np.arange(3)
    for model, color in (("BC", "#1f77b4"), ("Diffusion", "#d62728")):
        gaps = []
        for f in FRACTIONS:
            c = next(v for v in aggregate["current"] if v["fraction"] == f and v["model"] == model)["mean"]; q = next(v for v in aggregate["fresh"] if v["fraction"] == f and v["model"] == model)["mean"]; gaps.append(q - c)
        plt.plot(x, gaps, marker="o", label=model, color=color)
    plt.axhline(0, color="black", linewidth=.8); plt.xticks(x, ["25%", "50%", "100%"]); plt.ylabel("fresh - current success rate"); plt.xlabel("expert-data fraction"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "generalization_gap_vs_data_fraction.png", dpi=180); plt.close()
    return aggregate


def train_all(manifests: dict[int, dict[str, Any]], device: torch.device, smoke: bool = True) -> list[dict[str, Any]]:
    if not torch.cuda.is_available(): raise RuntimeError("BLOCKER_CUDA_UNAVAILABLE")
    statuses = []
    first = True
    for fraction in FRACTIONS:
        for seed in TRAINING_SEEDS:
            statuses.append(train_bc(fraction, seed, device, manifests, smoke=smoke and first)); first = False
    first = True
    for fraction in FRACTIONS:
        for seed in TRAINING_SEEDS:
            statuses.append(train_diffusion(fraction, seed, device, manifests, smoke=smoke and first)); first = False
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("manifests", "smoke", "train", "evaluate", "all"), default="all"); parser.add_argument("--no-smoke", action="store_true"); args = parser.parse_args(); manifests = make_manifests() if args.phase in ("manifests", "smoke", "train", "all") else load_manifests();
    if args.phase == "manifests": print(json.dumps(manifests, indent=2, ensure_ascii=False)); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.phase in ("smoke", "train", "all"): train_all(manifests, device, smoke=args.phase in ("smoke", "all") and not args.no_smoke)
    if args.phase in ("evaluate", "all"):
        evidence = run_evaluations(manifests, device); aggregate = make_tables_and_plots(evidence); write_json(OUT / "_evaluation_internal.json", {"aggregate": aggregate, "run_status": evidence["run_status"]})
    print(json.dumps({"task": TASK, "status": "COMPLETE" if args.phase != "smoke" else "SMOKE_COMPLETE", "formal_progress": "95%", "unique_next_task": "NONE — WAIT_FOR_CONTROLLER_REVIEW"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
