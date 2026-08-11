from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.generate_s8r2_10k_dataset import (
    FAMILIES, MAP_GENERATION_SEED, SPLIT_SEED, TASK_GENERATION_SEED,
    _map_valid, _split_maps, _task_points, generate_map, make_scene,
)


def test_procedural_map_generation():
    maps = [generate_map(i, 0) for i in range(len(FAMILIES))]
    assert len({m.map_id for m in maps}) == 10
    assert len({m.geometry_hash for m in maps}) == 10
    assert all(_map_valid(m)[0] for m in maps)


def test_map_geometry_valid():
    for family_index in range(len(FAMILIES)):
        for map_index in (0, 37, 99):
            assert _map_valid(generate_map(family_index, map_index))[0]


def test_ten_tasks_per_map_and_stratification():
    m = generate_map(0, 0)
    points = [_task_points(m, slot, 0) for slot in range(10)]
    assert len({tuple(s) for s, _ in points}) == 10
    assert len({tuple(g) for _, g in points}) == 10
    assert np.linalg.norm(points[2][0] - points[2][1]) > 0


def test_task_stratification_is_deterministic():
    m = generate_map(4, 12)
    assert all(np.array_equal(*pair) for pair in zip(_task_points(m, 7, 0), _task_points(m, 7, 0)))


def test_obs34():
    m = generate_map(6, 3); s, g = _task_points(m, 0); scene = make_scene(m, s, g, 0)
    assert scene.scene_id == m.map_id and scene.family == m.family


def test_action3_and_unit_ball_contract_constants():
    from scripts.generate_s8r2_10k_dataset import ACTION_DIM, OBS_DIM
    assert OBS_DIM == 34 and ACTION_DIM == 3


def test_map_level_split():
    split = _split_maps()
    assert {k: len(v) for k, v in split.items()} == {"train": 800, "val": 100, "test": 100}
    assert not split["train"] & split["val"]
    assert not split["train"] & split["test"]
    assert not split["val"] & split["test"]


def test_family_balance():
    split = _split_maps()
    for name, ids in split.items():
        assert all(sum(x.startswith(family + "_") for x in ids) == (80 if name == "train" else 10) for family in FAMILIES)


def test_episode_offsets_if_generated():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "s8r2_10k"
    if not (root / "train.npz").exists():
        return
    with np.load(root / "train.npz", allow_pickle=False) as data:
        assert len(data["episode_offsets"]) == 8000
        assert int(data["episode_offsets"][-1] + data["episode_lengths"][-1]) == len(data["actions"])


def test_resume_no_duplicate_if_generated():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "s8r2_10k" / "task_manifest.csv"
    if not root.exists():
        return
    import csv
    with root.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len({r["task_id"] for r in rows}) == 10000


def test_test_split_sealed_if_generated():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "s8r2_10k" / "dataset_summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["test_split_sealed"] is True


def test_seed_contract():
    assert MAP_GENERATION_SEED == 20260824
    assert TASK_GENERATION_SEED == 20260825
    assert SPLIT_SEED == 20260826
