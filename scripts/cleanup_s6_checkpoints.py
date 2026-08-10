"""Verify then remove only S6 intermediate checkpoints, retaining best/last."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = (ROOT / "checkpoints" / "s6").resolve()
ARTIFACTS = ROOT / "artifacts" / "s6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    records = []
    for run_dir in sorted((ARTIFACTS / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        train = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
        checkpoint_dir = (CHECKPOINT_ROOT / run_dir.name).resolve()
        if CHECKPOINT_ROOT not in checkpoint_dir.parents:
            raise RuntimeError(f"unsafe checkpoint directory: {checkpoint_dir}")
        best = checkpoint_dir / "best_model.zip"; last = checkpoint_dir / "last_model.zip"
        if sha256(best) != evaluation["best_checkpoint_sha256"]:
            raise RuntimeError(f"best checkpoint mismatch: {run_dir.name}")
        if sha256(last) != evaluation["last_checkpoint_sha256"]:
            raise RuntimeError(f"last checkpoint mismatch: {run_dir.name}")
        removed = []
        for item in train["checkpoint_schedule"]:
            path = Path(item["path"]).resolve()
            if checkpoint_dir not in path.parents or not path.name.startswith("checkpoint_") or path.suffix != ".zip":
                raise RuntimeError(f"unsafe intermediate target: {path}")
            if not path.exists() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"intermediate identity mismatch: {path}")
            removed.append({"name": path.name, "sha256": item["sha256"], "bytes": path.stat().st_size})
        for item in removed:
            (checkpoint_dir / item["name"]).unlink()
        records.append({
            "run": run_dir.name, "removed_intermediate_count": len(removed),
            "removed_intermediate_bytes": sum(item["bytes"] for item in removed),
            "removed_intermediates": removed,
            "retained_best": {"path": str(best), "sha256": sha256(best), "bytes": best.stat().st_size},
            "retained_last": {"path": str(last), "sha256": sha256(last), "bytes": last.stat().st_size},
        })
    if len(records) != 6 or any(item["removed_intermediate_count"] != 11 for item in records):
        raise RuntimeError(f"unexpected S6 cleanup scope: {records}")
    result = {
        "cleanup_gate": "AFTER_VAL_TEST_AND_INDEPENDENT_VERIFICATION_PASS",
        "runs": records, "total_removed_count": sum(item["removed_intermediate_count"] for item in records),
        "total_removed_bytes": sum(item["removed_intermediate_bytes"] for item in records),
        "retained_per_run": ["best_model.zip", "last_model.zip", "hashes", "curve", "summary", "raw VAL/TEST"],
        "recovery": "intermediate weights removed locally; hashes and evaluation evidence retained",
    }
    (ARTIFACTS / "storage_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"total_removed_count": result["total_removed_count"],
                      "total_removed_bytes": result["total_removed_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
