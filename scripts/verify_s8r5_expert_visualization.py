#!/usr/bin/env python3
"""Independently verify the frozen, TRAIN-only S8-R5 briefing bundle.

This verifier intentionally never imports or opens the S8-R2 TEST trajectory
archive.  It checks the generated artifacts, provenance, selection split, and
source hashes without running simulation, training, or model evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


TASK = "S8-R5-EXPERT-DATA-VISUALIZATION-FOR-ADVISOR-BRIEFING-V1"
PASS_LABEL = "PASS_S8R5_EXPERT_DATA_VISUALIZATION_READY"
EXPECTED_FIGURES = [
    "expert_family_overview",
    "ten_tasks_same_map",
    "single_expert_trajectory_3d",
    "expert_trajectory_timeseries",
    "expert_sample_format",
    "expert_dataset_statistics",
    "observation_ray_visualization",
    "expert_to_learning_pipeline",
]
EXPECTED_FAMILIES = {
    "SINGLE_BLOCK",
    "OFFSET_BLOCK",
    "DOUBLE_BLOCK",
    "ALTERNATING_BLOCKS",
    "SBEND_RANDOM",
    "CHICANE_RANDOM",
    "NARROW_GATE",
    "DOUBLE_GATE",
    "LATERAL_DETOUR",
    "MIXED_CLUTTER",
}
SOURCE_PATHS = {
    "dataset_summary.json": "artifacts/s8r2_10k/dataset_summary.json",
    "map_manifest.json": "artifacts/s8r2_10k/map_manifest.json",
    "task_manifest.csv": "artifacts/s8r2_10k/task_manifest.csv",
    "diversity_statistics.csv": "artifacts/s8r2_10k/diversity_statistics.csv",
    "train.npz": "artifacts/s8r2_10k/train.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify(repo: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    metadata_dir = repo / "artifacts/s8r5_visualization/metadata"
    selection = read_json(metadata_dir / "selection_manifest.json")
    figure_manifest = read_json(metadata_dir / "figure_manifest.json")
    statistics = read_json(metadata_dir / "statistics.json")

    check("task_identity", selection.get("task") == TASK == figure_manifest.get("task"), TASK)
    check("test_payload_unopened", selection.get("test_payload_accessed") is False and figure_manifest.get("test_payload_accessed") is False, "both manifests record false")
    check("training_not_run", figure_manifest.get("training_or_model_evaluation_run") is False, "no training or model evaluation")
    check("scientific_contract_unchanged", figure_manifest.get("scientific_contract_modified") is False, "producer records no scientific contract modification")
    check("only_train_payload_opened", figure_manifest.get("source_payloads_opened") == ["artifacts/s8r2_10k/train.npz"], str(figure_manifest.get("source_payloads_opened")))

    representatives = selection.get("family_representatives", [])
    rep_families = {item.get("family") for item in representatives}
    check("ten_family_representatives", len(representatives) == 10 and rep_families == EXPECTED_FAMILIES, f"count={len(representatives)}")
    check("family_representatives_train", all(item.get("split") == "train" for item in representatives), "all representative trajectories are TRAIN")

    same_map = selection.get("same_map", {})
    check("same_map_train", same_map.get("split") == "train", str(same_map.get("map_id")))
    check("same_map_ten_tasks", len(same_map.get("task_ids", [])) == 10 and len(set(same_map.get("task_ids", []))) == 10, "ten distinct task IDs")
    check("hero_train", selection.get("hero_trajectory", {}).get("split") == "train", str(selection.get("hero_trajectory", {}).get("task_id")))
    sample = selection.get("observation_sample", {})
    check("observation_sample_train", sample.get("split") == "train", str(sample.get("task_id")))
    check("sample_has_h16_room", int(sample.get("step", -1)) >= 0 and int(sample.get("step", -1)) + 16 <= int(sample.get("length", -1)), f"step={sample.get('step')}, length={sample.get('length')}")

    manifest_figures = figure_manifest.get("figures", [])
    check("eight_exact_figure_names", [item.get("name") for item in manifest_figures] == EXPECTED_FIGURES, str([item.get("name") for item in manifest_figures]))
    artifact_results: list[dict[str, Any]] = []
    for item in manifest_figures:
        png = repo / item["png"]
        svg = repo / item["svg"]
        png_exists = png.is_file() and png.stat().st_size > 0
        svg_exists = svg.is_file() and svg.stat().st_size > 0
        width = height = 0
        if png_exists:
            with Image.open(png) as image:
                width, height = image.size
                image.verify()
        png_hash = sha256(png) if png_exists else ""
        svg_hash = sha256(svg) if svg_exists else ""
        svg_text = svg.read_text(encoding="utf-8") if svg_exists else ""
        passed = (
            png_exists
            and svg_exists
            and width >= 2400
            and height >= 1350
            and abs(width / height - 16 / 9) < 0.01
            and png_hash == item.get("png_sha256")
            and svg_hash == item.get("svg_sha256")
            and "<text" in svg_text
        )
        artifact_results.append({
            "name": item.get("name"),
            "passed": passed,
            "pixel_size": [width, height],
            "png_sha256": png_hash,
            "svg_sha256": svg_hash,
            "editable_svg_text": "<text" in svg_text,
        })
    check("artifact_integrity", all(item["passed"] for item in artifact_results), "PNG/SVG exist, hashes match, 16:9 high-resolution PNG, editable SVG text")

    source_results: dict[str, Any] = {}
    for name, relative in SOURCE_PATHS.items():
        path = repo / relative
        actual = sha256(path) if path.is_file() else ""
        expected = figure_manifest.get("source_hashes", {}).get(name, "")
        source_results[name] = {"path": relative, "actual_sha256": actual, "expected_sha256": expected, "passed": bool(actual) and actual == expected}
    check("frozen_source_hashes", all(item["passed"] for item in source_results.values()), "metadata plus TRAIN archive match producer provenance")

    generator_source = (repo / "scripts/generate_s8r5_expert_visualization.py").read_text(encoding="utf-8")
    forbidden_literal = "test" + ".npz"
    check("generator_has_no_test_archive_literal", forbidden_literal.lower() not in generator_source.lower(), "static source scan")
    check("statistics_contract", statistics.get("dataset_scale", {}).get("train_h16_windows") == 4_776_168 and statistics.get("quality", {}).get("action_violation_count_recomputed_on_train") == 0, "H=16 TRAIN windows and action-support check")

    passed = all(item["passed"] for item in checks)
    return {
        "task": TASK,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_label": PASS_LABEL if passed else "BLOCKED_S8R5_EXPERT_DATA_VISUALIZATION_VERIFICATION",
        "passed": passed,
        "checks": checks,
        "artifacts": artifact_results,
        "sources": source_results,
        "test_payload_accessed": False,
        "training_or_model_evaluation_run": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = verify(repo)
    output = repo / "artifacts/s8r5_visualization/metadata/verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"final_label": result["final_label"], "checks": len(result["checks"]), "output": str(output)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
