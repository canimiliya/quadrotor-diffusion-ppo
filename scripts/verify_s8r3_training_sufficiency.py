"""Independent, read-only verifier for the S8-R3 sufficiency package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "s8r2_10k"
OUT = ROOT / "artifacts" / "s8r3"
TRAIN_SHA = "395a5c2ab1eb6cb3f3925fa3ecbe160d151833ef95cc5bc3b9187fbc497de3c5"
VAL_SHA = "0fc50886504d36890444b4c09c2990962cc5be6551cb0e54c008df0d26f5246e"
TEST_SHA = "b4eee85496acb4d1e7b02e4f4bb44715c18c355014e77a79f8499a5fffa3a74e"
FAMILIES = ("SINGLE_BLOCK", "OFFSET_BLOCK", "DOUBLE_BLOCK", "ALTERNATING_BLOCKS", "SBEND_RANDOM", "CHICANE_RANDOM", "NARROW_GATE", "DOUBLE_GATE", "LATERAL_DETOUR", "MIXED_CLUTTER")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))


def check(name: str, condition: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(condition), "detail": detail}


def verify() -> dict:
    checks: list[dict] = []
    summary_path = OUT / "summary.json"
    if not summary_path.exists(): raise RuntimeError("missing S8R3 summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hashes = {"train": sha(DATA / "train.npz"), "val": sha(DATA / "val.npz"), "test": sha(DATA / "test.npz")}
    checks += [check("train_hash", hashes["train"] == TRAIN_SHA, hashes["train"]), check("val_hash", hashes["val"] == VAL_SHA, hashes["val"]), check("test_hash", hashes["test"] == TEST_SHA, hashes["test"])]
    stats = json.loads((OUT / "window_statistics.json").read_text(encoding="utf-8"))
    checks += [check("train_windows", stats["train_h16_windows"] == 4776168, str(stats["train_h16_windows"])), check("val_windows", stats["val_h16_windows"] == 598153, str(stats["val_h16_windows"])), check("updates_per_pass", stats["updates_per_effective_pass"] == 9329, str(stats["updates_per_effective_pass"])), check("test_sealed", summary.get("test_accessed") is False and summary.get("test_values_opened") is False)]
    norm = np.load(OUT / "train_obs_normalization.npz", allow_pickle=False)
    checks += [check("normalization_shape", norm["mean"].shape == (34,) and norm["std"].shape == (34,)), check("normalization_finite", np.isfinite(norm["mean"]).all() and np.isfinite(norm["std"]).all() and (norm["std"] > 0).all()), check("normalization_train_only", str(norm["source_split"][0]) == "TRAIN")]
    checks += [check("ppo_zero", summary.get("ppo_run_count") == 0), check("test_run_zero", summary.get("test_run_count") == 0), check("formal_progress", summary.get("formal_progress") == "95%")]
    for kind, expected_params in (("bc", 592688), ("diffusion", 621360)):
        model = summary[kind]
        keys = sorted(model["checkpoints"].keys(), key=lambda x: int(x))
        checks += [check(f"{kind}_params", model["parameter_count"] == expected_params, str(model["parameter_count"])), check(f"{kind}_finite", model["all_finite"] is True), check(f"{kind}_checkpoint_completeness", keys == ["1", "5", "10", "20", "30"])]
        for p, item in model["checkpoints"].items():
            path = Path(item["path"])
            checks.append(check(f"{kind}_checkpoint_{p}_exists", path.exists()))
            if path.exists(): checks.append(check(f"{kind}_checkpoint_{p}_hash", sha(path) == item["sha256"]))
    curves = load_csv(OUT / "training_curve.csv")
    checks += [check("curve_metrics_present", len(curves) >= 100 and all(all(k in r for k in ("train_loss", "val_offline_loss", "gradient_norm", "learning_rate", "wall_time_s", "gpu_memory_bytes")) for r in curves)), check("curve_finite", all(np.isfinite(float(r["train_loss"])) and np.isfinite(float(r["val_offline_loss"])) and np.isfinite(float(r["gradient_norm"])) for r in curves))]
    closed = load_csv(OUT / "closed_loop_metrics.csv")
    family = load_csv(OUT / "family_metrics.csv")
    checks += [check("closed_loop_checkpoints", len(closed) == 10), check("closed_loop_val_count", all(int(r["count"]) == 1000 for r in closed)), check("family_rows", len(family) == 100), check("family_coverage", all(int(r["count"]) == 100 for r in family)), check("rollout_finite", all(int(r["nonfinite"]) == 0 for r in closed))]
    checks.append(check("next_task_wait", summary.get("unique_next_task") == "NONE — WAIT_FOR_CONTROLLER_REVIEW"))
    result = {"task": summary.get("task"), "checks": checks, "passed": sum(int(c["pass"]) for c in checks), "failed": sum(int(not c["pass"]) for c in checks)}
    result["final_label"] = "PASS_S8R3_INDEPENDENT_VERIFICATION" if result["failed"] == 0 else "BLOCKED_S8R3_INDEPENDENT_VERIFICATION"
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["failed"]: raise SystemExit(1)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); verify()
