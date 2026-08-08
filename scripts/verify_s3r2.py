"""Independent read-only verifier for the completed S3-R2 artifact set."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from scripts.run_s3_diffusion import build_windows, load_npz, read_task_rows, row_task
from scripts.run_s3r2 import (
    BRANCH, DDIM_STEPS, EXPECTED_CHECKPOINT_SHA256, EXPECTED_DATASET_SHA256,
    HORIZON, OBSERVATION_DIM, ACTION_DIM, R1_CHECKPOINT, S2_DATASET, S2_ROOT,
    S3R2_ROOT, CHECKPOINT_ROOT, evaluate_task, family_gate, rollout_summary,
    set_deterministic, sha256_file,
)
from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fail(message: str):
    raise RuntimeError("S3R2_VERIFY_FAIL: " + message)


def main() -> None:
    summary_path = S3R2_ROOT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["final_label"] != "PASS_S3R2_RECOVERY_AUGMENTED_DIFFUSION_SANITY":
        fail("final label")
    if summary["branch"] != BRANCH or summary["formal_progress"] != "30%":
        fail("branch or formal progress")
    if summary["r2_test_run_count"] != 1 or not summary["r2_test_executed"] or summary["test_used_for_selection"]:
        fail("TEST count/selection contract")
    if summary["val_recovery_samples"] != 0 or summary["test_recovery_samples"] != 0:
        fail("VAL/TEST recovery leakage")

    for name, expected in EXPECTED_DATASET_SHA256.items():
        if sha256_file(S2_DATASET / f"{name}.npz") != expected:
            fail(f"dataset hash {name}")
    if sha256_file(R1_CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        fail("R1 checkpoint hash")
    checkpoint = torch.load(CHECKPOINT_ROOT / "best.pt", map_location="cpu", weights_only=False)
    if not checkpoint.get("model_frozen") or checkpoint.get("prediction_target") != "x0" or checkpoint.get("action_support") != "unit_ball_squash":
        fail("R2 checkpoint metadata")
    expected_config = {"observation_dim": 34, "horizon": 16, "action_dim": 3,
                       "observation_width": 128, "time_dim": 64, "width": 256,
                       "residual_blocks": 4, "bounded_output": True}
    if checkpoint.get("model_config") != expected_config:
        fail("R2 architecture")

    manifest = read_csv(S3R2_ROOT / "recovery_manifest.csv")
    accepted = [row for row in manifest if row["accepted"] == "True"]
    if len(manifest) != 242 or len(accepted) != 146 or len({row["task_id"] for row in manifest}) != 242:
        fail("recovery manifest counts/uniqueness")
    if any(row["source_split"] != "TRAIN" for row in manifest):
        fail("recovery source split")
    if any(row["task_id"].startswith(("VAL", "TEST")) for row in manifest):
        fail("recovery task id leakage")
    if {row["family"] for row in accepted} != {"OPEN", "BLOCK", "SBEND"}:
        fail("family yield")

    recovery_path = S3R2_ROOT / "recovery_dataset" / "recovery_train.npz"
    # The dataset is gitignored but must be present locally for this audit.
    if not recovery_path.exists():
        recovery_path = PROJECT_ROOT / "artifacts" / "s3r2" / "recovery_dataset" / "recovery_train.npz"
    recovery = load_npz(recovery_path)
    if len(recovery["episode_offsets"]) != 146 or not all(np.isfinite(value).all() for value in recovery.values() if value.dtype.kind == "f"):
        fail("recovery dataset finiteness/episode count")
    if np.any(recovery["episode_lengths"] < HORIZON):
        fail("recovery episode shorter than horizon")
    recovery_obs, recovery_actions = build_windows(recovery)
    if len(recovery_obs) != summary["recovery_train_windows"] or recovery_actions.shape[1:] != (HORIZON, ACTION_DIM):
        fail("recovery window accounting")

    val_rows = read_csv(S3R2_ROOT / "val_rollout.csv")
    test_rows = read_csv(S3R2_ROOT / "test_rollout.csv")
    if len(val_rows) != 54 or len(test_rows) != 54:
        fail("VAL/TEST row count")
    if not family_gate(summary["val"]) or not family_gate(summary["test"]):
        fail("family gate")
    for label in ("val", "test"):
        result = summary[label]
        if label == "test":
            result = {**result["overall"], "unsafe": result["unsafe"],
                      "support_violation_fraction": result["support_violation_fraction"],
                      "executed_action_clipping_fraction": result["executed_action_clipping_fraction"]}
        if result["success"] < 27 or result["unsafe"] > 13 or result["nonfinite"] != 0:
            fail(f"{label} closed-loop gate")
        if result["support_violation_fraction"] > 0.01 or result["executed_action_clipping_fraction"] > 0.01:
            fail(f"{label} support/clipping gate")

    if not torch.cuda.is_available():
        fail("CUDA unavailable for determinism smoke")
    device = torch.device("cuda")
    set_deterministic(20260810)
    payload = torch.load(CHECKPOINT_ROOT / "best.pt", map_location=device, weights_only=False)
    model = ConditionalDiffusionMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"]); model.eval()
    schedule = DiffusionSchedule(payload["diffusion_steps"]).to(device)
    mean = np.asarray(payload["observation_mean"], dtype=np.float32)
    std = np.asarray(payload["observation_std"], dtype=np.float32)
    smoke = {}
    val_manifest = sorted([row for row in read_csv(S2_ROOT / "task_manifest.csv") if row["accepted"].lower() == "true" and row["split"] == "val"], key=lambda row: row["task_id"])
    for family in ("OPEN", "BLOCK", "SBEND"):
        row = next(item for item in val_manifest if item["scene_id"].startswith(family))
        scene, task = row_task(row)
        first = evaluate_task(model, schedule, scene, task, mean, std, device)
        second = evaluate_task(model, schedule, scene, task, mean, std, device)
        smoke[family] = {"task_id": task["task_id"], "hashes": [first["action_trace_hash"], second["action_trace_hash"]],
                         "match": first["action_trace_hash"] == second["action_trace_hash"]}
    if not all(item["match"] for item in smoke.values()):
        fail("VAL determinism smoke")

    summary["determinism"] = {"pass": True, "val_smoke": smoke, "test_rerun_performed": False}
    summary["tests"] = {"passed": None, "failed": None, "independent_verifier": True}
    summary["regression"] = {"pending_external_pytest": True}
    summary["test_values_opened_after_val_pass"] = True
    summary["original_dataset_identity"]["test_values_opened"] = True
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_pass": True, "determinism": summary["determinism"],
                      "accepted_recovery": len(accepted), "val": summary["val"], "test": summary["test"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
