from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
META = REPO / "artifacts/s8r5_visualization/metadata"
EXPECTED = [
    "expert_family_overview",
    "ten_tasks_same_map",
    "single_expert_trajectory_3d",
    "expert_trajectory_timeseries",
    "expert_sample_format",
    "expert_dataset_statistics",
    "observation_ray_visualization",
    "expert_to_learning_pipeline",
]


def load(name: str) -> dict:
    return json.loads((META / name).read_text(encoding="utf-8"))


def test_s8r5_selection_is_deterministic_and_train_only() -> None:
    selection = load("selection_manifest.json")
    assert selection["random_seed"] is None
    assert selection["random_selection_used"] is False
    assert selection["test_payload_accessed"] is False
    assert len(selection["family_representatives"]) == 10
    assert all(item["split"] == "train" for item in selection["family_representatives"])
    assert selection["same_map"]["split"] == "train"
    assert len(selection["same_map"]["task_ids"]) == 10
    assert selection["hero_trajectory"]["split"] == "train"
    assert selection["observation_sample"]["split"] == "train"


def test_s8r5_manifest_records_frozen_scope() -> None:
    manifest = load("figure_manifest.json")
    assert manifest["test_payload_accessed"] is False
    assert manifest["training_or_model_evaluation_run"] is False
    assert manifest["scientific_contract_modified"] is False
    assert manifest["source_payloads_opened"] == ["artifacts/s8r2_10k/train.npz"]
    assert [item["name"] for item in manifest["figures"]] == EXPECTED


def test_s8r5_png_and_editable_svg_exports_exist() -> None:
    manifest = load("figure_manifest.json")
    for item in manifest["figures"]:
        png = REPO / item["png"]
        svg = REPO / item["svg"]
        assert png.stat().st_size > 0
        assert svg.stat().st_size > 0
        with Image.open(png) as image:
            assert image.size == tuple(item["pixel_size"])
            assert image.width >= 2400 and image.height >= 1350
            assert abs(image.width / image.height - 16 / 9) < 0.01
        assert "<text" in svg.read_text(encoding="utf-8")


def test_s8r5_statistics_and_h16_contract() -> None:
    statistics = load("statistics.json")
    assert statistics["dataset_scale"]["maps"] == 1000
    assert statistics["dataset_scale"]["trajectories"] == 10000
    assert statistics["dataset_scale"]["train_trajectories"] == 8000
    assert statistics["dataset_scale"]["train_h16_windows"] == 4_776_168
    assert statistics["quality"]["action_violation_count_recomputed_on_train"] == 0


def test_s8r5_generator_contains_no_test_archive_reference() -> None:
    source = (REPO / "scripts/generate_s8r5_expert_visualization.py").read_text(encoding="utf-8").lower()
    assert ("test" + ".npz") not in source
    assert "train_path" in source
