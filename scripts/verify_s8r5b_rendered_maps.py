"""Verify the S8-R5B rendered expert-map bundle without simulation or training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "artifacts" / "s8r2_10k"
BUNDLE_ROOT = ROOT / "artifacts" / "s8r5_rendered_maps"
FIG_ROOT = BUNDLE_ROOT / "figures"
META_ROOT = BUNDLE_ROOT / "metadata"
MANIFEST_PATH = META_ROOT / "render_manifest.json"

EXPECTED = {
    "render_double_gate_hero": ("DOUBLE_GATE_075", "DOUBLE_GATE_075_TASK_08"),
    "render_sbend_hero": ("SBEND_RANDOM_023", "SBEND_RANDOM_023_TASK_05"),
    "render_chicane_or_detour_hero": ("CHICANE_RANDOM_090", "CHICANE_RANDOM_090_TASK_02"),
    "render_clutter_or_gate_hero": ("MIXED_CLUTTER_078", "MIXED_CLUTTER_078_TASK_04"),
    "render_same_map_multi_trajectories": ("DOUBLE_GATE_075", "DOUBLE_GATE_075_TASK_00"),
    "render_single_scene_with_drone_marker": ("MIXED_CLUTTER_078", "MIXED_CLUTTER_078_TASK_04"),
    "rendered_expert_map_overview": ("MULTI_SCENE_OVERVIEW", "DOUBLE_GATE_075_TASK_08"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    metadata = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    figures = metadata.get("figures", [])
    names = [str(item.get("figure_name")) for item in figures]
    check("seven_figures_listed", names == list(EXPECTED), str(names))
    check("test_payload_unopened", metadata.get("test_payload_accessed") is False, "manifest records false")
    check("training_not_run", metadata.get("model_training_run_count") == 0, "model_training_run_count=0")
    check("scientific_contract_unchanged", metadata.get("scientific_contract_changed") is False, "manifest records false")
    check("train_payload_only", metadata.get("source_payloads_opened") == ["artifacts/s8r2_10k/train.npz"], str(metadata.get("source_payloads_opened")))

    with (DATA_ROOT / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        task_rows = {row["task_id"]: row for row in csv.DictReader(handle)}
    map_rows = {row["map_id"]: row for row in json.loads((DATA_ROOT / "map_manifest.json").read_text(encoding="utf-8"))}
    with np.load(DATA_ROOT / "train.npz", allow_pickle=False) as data:
        train_task_ids = set(data["task_ids"].astype(str).tolist())
        train_count = len(train_task_ids)
    check("train_episode_index", train_count == 8000, f"unique TRAIN episodes={train_count}")

    artifact_passed = True
    provenance_passed = True
    for item in figures:
        name = str(item["figure_name"])
        path = FIG_ROOT / f"{name}.png"
        expected_size = (3840, 2160) if name == "rendered_expert_map_overview" else (2560, 1440)
        exists = path.is_file() and path.stat().st_size > 0
        size = Image.open(path).size if exists else (0, 0)
        variance = float(np.asarray(Image.open(path).convert("L")).var()) if exists else 0.0
        item_hash = sha256(path) if exists else ""
        artifact_ok = exists and size == expected_size and variance > 20.0 and item_hash == item.get("file_sha256")
        artifact_passed = artifact_passed and artifact_ok
        check(f"artifact_{name}", artifact_ok, f"exists={exists}, size={size}, variance={variance:.2f}")

        expected_map, expected_task = EXPECTED[name]
        map_ok = expected_map in str(item.get("map_id"))
        task_ok = expected_task in str(item.get("task_id"))
        if name != "rendered_expert_map_overview":
            task_ok = task_ok and all(task_id in train_task_ids for task_id in str(item.get("task_id")).split("; "))
        if expected_map != "MULTI_SCENE_OVERVIEW":
            map_ok = map_ok and expected_map in map_rows
        if name != "rendered_expert_map_overview":
            task_ok = task_ok and expected_task in task_rows and task_rows[expected_task]["split"] == "train"
        provenance_ok = bool(item.get("based_on_real_trajectory")) and item.get("split") == "train" and map_ok and task_ok
        provenance_passed = provenance_passed and provenance_ok
        check(f"provenance_{name}", provenance_ok, f"map_ok={map_ok}, task_ok={task_ok}")

    check("all_artifacts_pass", artifact_passed, "all PNGs are nonblank, hashed, and resolution-checked")
    check("all_provenance_pass", provenance_passed, "all records identify TRAIN-backed map/task sources")
    passed = all(bool(item["passed"]) for item in checks)
    result = {
        "task": "S8-R5B-RENDERED-EXPERT-TRAJECTORY-MAPS-FOR-ADVISOR-V1",
        "final_label": "PASS_S8R5B_RENDERED_EXPERT_MAPS_READY" if passed else "BLOCKED_S8R5B_RENDERED_EXPERT_MAPS_VERIFICATION",
        "passed": passed,
        "checks": checks,
        "test_payload_accessed": False,
        "model_training_run_count": 0,
        "scientific_contract_changed": False,
    }
    output = META_ROOT / "verification.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"final_label": result["final_label"], "passed": passed, "checks": len(checks), "output": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
