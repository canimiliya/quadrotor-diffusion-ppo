"""Independent read-only verification of the S3-D0 audit artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
S3D0 = ROOT / "artifacts" / "s3d0"
S3R1 = ROOT / "artifacts" / "s3r1"
CHECKPOINT = ROOT / "checkpoints" / "s3r1" / "best.pt"
EXPECTED_CHECKPOINT = "8afa677d7c8a34179c60f067209b6924170fc7854445048d7a7e9b55bbbc5fc8"
EXPECTED_DATASET = {
    "train": "701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21",
    "val": "c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc",
    "test": "88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads((S3D0 / "summary.json").read_text(encoding="utf-8"))
    with (S3D0 / "per_episode.csv").open(encoding="utf-8-sig", newline="") as handle:
        episodes = list(csv.DictReader(handle))
    with (S3D0 / "failure_taxonomy.csv").open(encoding="utf-8-sig", newline="") as handle:
        failures = list(csv.DictReader(handle))
    outcomes = {}
    for row in episodes:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    failure_types = {}
    for row in failures:
        failure_types[row["failure_type"]] = failure_types.get(row["failure_type"], 0) + 1
    checks = {
        "final_label": summary.get("final_label") == "PASS_S3D0_VAL_FAILURE_ROOT_CAUSE_AUDIT",
        "checkpoint_identity": CHECKPOINT.exists() and sha256(CHECKPOINT) == EXPECTED_CHECKPOINT,
        "dataset_identity": summary.get("dataset_identity", {}).get("sha256") == EXPECTED_DATASET,
        "test_not_accessed": summary.get("test_accessed") is False and not (S3R1 / "test_rollout.csv").exists(),
        "episodes_54": len(episodes) == 54 and outcomes == {"SUCCESS": 29, "COLLISION": 9, "GROUND_CONTACT": 6, "TIMEOUT": 10},
        "failure_taxonomy": failure_types == {"collision": 9, "ground_contact": 6, "timeout": 10},
        "teacher_independent": summary.get("policy_execution_teacher_independent", {}).get("passed") is True,
        "deterministic": summary.get("diagnostic_determinism", {}).get("pass") is True,
        "regression": summary.get("regression", {}).get("pytest_returncode") == 0 and summary.get("tests", {}).get("failed") == 0,
        "large_files_not_present": not any(path.suffix.lower() in {".mp4", ".gif", ".npz", ".pt", ".pth", ".ckpt"} for path in S3D0.rglob("*")),
    }
    pytest_result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    checks["pytest_live"] = pytest_result.returncode == 0
    result = {"audit_pass": all(checks.values()), "checks": checks, "outcomes": outcomes,
              "failure_types": failure_types, "pytest_returncode": pytest_result.returncode}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
