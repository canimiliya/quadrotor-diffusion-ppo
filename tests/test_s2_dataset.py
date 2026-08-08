from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_diffusion_ppo.envs.scene import SCENE_IDS, load_all_scenes
from quadrotor_diffusion_ppo.expert.s2_dataset import (
    MAX_CANDIDATE_ATTEMPTS, SPLIT_SEED, TARGET_ACCEPTED, candidate_input_hash,
    candidate_input_rows, split_accepted_records, validate_npz,
)


def test_task_sampler_is_deterministic_and_scene_balanced():
    scenes = load_all_scenes()
    rows_a = [row for index, scene in enumerate(scenes) for row in candidate_input_rows(scene, index)]
    rows_b = [row for index, scene in enumerate(scenes) for row in candidate_input_rows(scene, index)]
    assert len(rows_a) == 9 * MAX_CANDIDATE_ATTEMPTS
    assert rows_a == rows_b
    assert candidate_input_hash(rows_a) == candidate_input_hash(rows_b)
    assert {row["scene_id"] for row in rows_a} == set(SCENE_IDS)


def test_split_is_task_isolated_and_balanced():
    rows = []
    for scene_index, scene_id in enumerate(SCENE_IDS):
        for candidate_id in range(TARGET_ACCEPTED):
            rows.append({"scene_id": scene_id, "candidate_id": candidate_id, "task_id": f"{scene_id}_{candidate_id}", "accepted": True})
    split = split_accepted_records(rows, SPLIT_SEED)
    assert {name: len(value) for name, value in split.items()} == {"train": 252, "val": 54, "test": 54}
    ids = {name: {row["task_id"] for row in value} for name, value in split.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    for scene_id in SCENE_IDS:
        assert [sum(row["scene_id"] == scene_id for row in split[name]) for name in ("train", "val", "test")] == [28, 6, 6]


def test_dataset_schema_if_generated():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "s2" / "dataset"
    if not root.exists():
        return
    stats = {name: validate_npz(root / f"{name}.npz") for name in ("train", "val", "test")}
    assert {name: value["episodes"] for name, value in stats.items()} == {"train": 252, "val": 54, "test": 54}
    assert all(value["transitions"] > 0 for value in stats.values())
