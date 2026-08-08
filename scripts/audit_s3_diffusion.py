"""Independent audit for the frozen S3-R0 diffusion sanity result."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "artifacts" / "s2"
S3 = ROOT / "artifacts" / "s3"
DATASET = S2 / "dataset"
CHECKPOINT = ROOT / "checkpoints" / "s3" / "best.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def validate_npz(path: Path, expected_episodes: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        offsets, lengths = data["episode_offsets"], data["episode_lengths"]
        valid = len(offsets) == expected_episodes and len(offsets) == len(lengths) and len(offsets) > 0
        valid = valid and bool(offsets[-1] + lengths[-1] == len(data["actions"]))
        valid = valid and bool(np.all(lengths > 0))
        valid = valid and bool(np.all(offsets[1:] == offsets[:-1] + lengths[:-1]))
        valid = valid and all(np.isfinite(data[name]).all() for name in data.files if data[name].dtype.kind == "f")
        valid = valid and data["observations"].shape[1:] == (34,) and data["actions"].shape[1:] == (3,)
        return {"path": str(path.relative_to(ROOT)), "episodes": int(len(offsets)),
                "transitions": int(len(data["actions"])), "valid": bool(valid)}


def main() -> None:
    summary_path = S3 / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_hashes = {name: sha256_file(DATASET / f"{name}.npz") for name in ("train", "val", "test")}
    identity_pass = dataset_hashes == summary["dataset_identity"]["sha256"]
    dataset_checks = [validate_npz(DATASET / "train.npz", 252), validate_npz(DATASET / "val.npz", 54), validate_npz(DATASET / "test.npz", 54)]
    identity_pass = identity_pass and all(item["valid"] for item in dataset_checks)
    curve = load_rows(S3 / "training_curve.csv")
    curve_pass = len(curve) == 10_000 and [int(row["update"]) for row in curve] == list(range(1, 10_001))
    curve_pass = curve_pass and all(finite_float(row["train_loss"]) and finite_float(row["gradient_norm"]) for row in curve)
    curve_pass = curve_pass and all(not row["val_loss"] or finite_float(row["val_loss"]) for row in curve)
    checkpoint_payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint_pass = bool(checkpoint_payload.get("model_frozen")) and sha256_file(CHECKPOINT) == summary["best_checkpoint_sha256"]
    val_rows, test_rows = load_rows(S3 / "val_rollout.csv"), load_rows(S3 / "test_rollout.csv")
    rollout_pass = len(val_rows) == 54 and len(test_rows) == 54
    rollout_pass = rollout_pass and all(sum(row["scene_id"] == scene for row in test_rows) == 6 for scene in (
        "OPEN_00", "OPEN_01", "OPEN_02", "BLOCK_00", "BLOCK_01", "BLOCK_02", "SBEND_00", "SBEND_01", "SBEND_02"))
    test_success = sum(row["success"].lower() == "true" for row in test_rows)
    test_collision = sum(row["collision"] == "1" for row in test_rows)
    test_ground = sum(row["ground_contact"] == "1" for row in test_rows)
    test_nonfinite = sum(row["nonfinite"] == "1" for row in test_rows)
    total_steps = sum(int(row["steps"]) for row in test_rows)
    total_clips = sum(int(row["raw_action_clip_count"]) for row in test_rows)
    clip_fraction = total_clips / max(1, total_steps)
    family_success = {family: sum(row["success"].lower() == "true" for row in test_rows if row["family"] == family)
                      for family in ("OPEN", "BLOCK", "SBEND")}
    smoke_pass = all(item["match"] and item["status"] == "PASS" for item in summary["determinism_smoke"].values())
    leakage_pass = bool(summary["privileged_information_leakage"]["passed"])
    gates = {"test_success_ge_27": test_success >= 27, "family_success_each": all(value > 0 for value in family_success.values()),
             "collision_ground_le_25_percent": (test_collision + test_ground) / 54 <= 0.25,
             "test_nonfinite_zero": test_nonfinite == 0, "executed_action_nonfinite_zero": test_nonfinite == 0,
             "raw_action_clip_le_1_percent": clip_fraction <= 0.01, "teacher_leakage_absent": leakage_pass,
             "test_not_accessed_before_freeze": summary["test_accessed_before_model_freeze"] is False,
             "determinism_smoke": smoke_pass, "dataset_identity": identity_pass, "training_curve": curve_pass,
             "checkpoint_frozen": checkpoint_pass, "rollout_task_counts": rollout_pass}
    if gates["raw_action_clip_le_1_percent"] is False:
        label = "BLOCKED_S3_DIFFUSION_ACTION_SUPPORT"
    elif gates["collision_ground_le_25_percent"] is False:
        label = "BLOCKED_S3_DIFFUSION_UNSAFE"
    elif gates["test_success_ge_27"] is False or gates["family_success_each"] is False:
        label = "BLOCKED_S3_DIFFUSION_CLOSED_LOOP_WEAK"
    elif all(gates.values()):
        label = "PASS_S3_DIFFUSION_ONLY_SANITY_V1"
    else:
        label = "BLOCKED_S3_DIFFUSION_PROTOCOL"
    env = dict(**__import__("os").environ)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    tests_pass = result.returncode == 0
    audit = {"final_label": label, "dataset_hashes_match": identity_pass, "dataset_checks": dataset_checks,
             "checkpoint_frozen": checkpoint_pass, "training_curve_valid": curve_pass, "rollout_counts_valid": rollout_pass,
             "test_success": test_success, "test_collision": test_collision, "test_ground_contact": test_ground,
             "test_nonfinite": test_nonfinite, "test_clip_count": total_clips, "test_clip_fraction": clip_fraction,
             "test_family_success": family_success, "gates": gates, "determinism_smoke_pass": smoke_pass,
             "pytest": {"command": "python -m pytest -q", "returncode": result.returncode,
                        "status": "PASS" if tests_pass else "FAIL", "output": result.stdout[-4000:]}}
    summary["final_label"] = label
    summary["audit"] = audit
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    raise SystemExit(0 if label == "PASS_S3_DIFFUSION_ONLY_SANITY_V1" and tests_pass else 1)


if __name__ == "__main__":
    main()
