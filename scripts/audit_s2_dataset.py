"""Independent post-generation audit for the S2 dataset artifact."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from quadrotor_diffusion_ppo.envs.scene import SCENE_IDS, load_all_scenes
from quadrotor_diffusion_ppo.expert.s2_dataset import (
    SPLIT_SEED, TaskSpec, _task_scene, candidate_input_hash, candidate_input_rows,
    cleanup_work_root, run_gcopter_planner, rollout_task, validate_npz,
)
from quadrotor_diffusion_ppo.expert.trajectory import GcopterTrajectory


def _digest(data: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(data):
        value = np.asarray(data[name])
        digest.update(name.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    s2_root = root / "artifacts" / "s2"
    dataset_root = s2_root / "dataset"
    summary_path = s2_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = list(__import__("csv").DictReader((s2_root / "task_manifest.csv").open(encoding="utf-8", newline="")))
    split_manifest = list(__import__("csv").DictReader((s2_root / "split_manifest.csv").open(encoding="utf-8", newline="")))
    accepted = [row for row in manifest if row["accepted"].lower() in {"true", "1"}]
    assert len(accepted) == 360
    assert len({row["task_id"] for row in accepted}) == 360
    assert {row["split"] for row in accepted} == {"train", "val", "test"}
    assert len(split_manifest) == 360
    assert len({row["task_id"] for row in split_manifest}) == 360
    assert {split: sum(row["split"] == split for row in split_manifest) for split in ("train", "val", "test")} == {"train": 252, "val": 54, "test": 54}
    assert all(sum(row["scene_id"] == scene_id and row["split"] == split for row in split_manifest) == count
               for scene_id in SCENE_IDS for split, count in (("train", 28), ("val", 6), ("test", 6)))

    npz_stats = {split: validate_npz(dataset_root / f"{split}.npz") for split in ("train", "val", "test")}
    assert {split: stats["episodes"] for split, stats in npz_stats.items()} == {"train": 252, "val": 54, "test": 54}
    task_ids_by_split = {split: {row["task_id"] for row in split_manifest if row["split"] == split} for split in ("train", "val", "test")}
    assert not task_ids_by_split["train"] & task_ids_by_split["val"]
    assert not task_ids_by_split["train"] & task_ids_by_split["test"]
    assert not task_ids_by_split["val"] & task_ids_by_split["test"]

    scenes = load_all_scenes()
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    smoke_root = s2_root / "_smoke"
    smoke = {}
    try:
        for family in ("OPEN", "BLOCK", "SBEND"):
            manifest_row = next(row for row in accepted if scene_by_id[row["scene_id"]].family == family)
            scene = scene_by_id[manifest_row["scene_id"]]
            task = TaskSpec(scene_id=manifest_row["scene_id"], candidate_id=int(manifest_row["candidate_id"]),
                            task_id=manifest_row["task_id"], candidate_seed=int(manifest_row["candidate_seed"]),
                            start=np.asarray([float(manifest_row[f"start_{axis}"]) for axis in "xyz"]),
                            goal=np.asarray([float(manifest_row[f"goal_{axis}"]) for axis in "xyz"]))
            digests = []
            for repeat in range(2):
                out, _ = run_gcopter_planner(scene, task, smoke_root / family / str(repeat))
                trajectory = GcopterTrajectory.from_csv(out / "reference_coefficients.csv")
                contract = json.loads((root / "configs" / "m0_contract.json").read_text(encoding="utf-8"))
                row, data = rollout_task(_task_scene(scene, task), task, trajectory, contract)
                assert row["accepted"] and data is not None
                digests.append(_digest(data))
            smoke[family] = {"hashes": digests, "match": digests[0] == digests[1]}
            assert smoke[family]["match"]
    finally:
        cleanup_work_root(smoke_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    pytest = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, env=env,
                            capture_output=True, text=True, check=False)
    print(pytest.stdout)
    if pytest.stderr:
        print(pytest.stderr, file=sys.stderr)
    assert pytest.returncode == 0

    summary["determinism_smoke"] = {family: {"hashes": value["hashes"], "match": value["match"], "status": "PASS"}
                                     for family, value in smoke.items()}
    summary["tests"] = {"command": "python -m pytest -q", "status": "PASS", "returncode": pytest.returncode}
    summary["m0_regression"] = "PASS"
    summary["dataset_schema"] = "PASS"
    summary["episode_boundary_check"] = True
    summary["split_task_isolation"] = "PASS"
    summary["current_blocker"] = "NONE"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_schema": "PASS", "split_task_isolation": "PASS", "determinism_smoke": smoke,
                      "m0_regression": "PASS", "final_label": summary["final_label"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
