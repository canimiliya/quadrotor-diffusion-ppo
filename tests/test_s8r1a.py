"""Lightweight regression checks for the frozen S8-R1A evidence package."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_s8r1a import BC_CONFIG, DIFFUSION_CONFIG, OUT, object_hash


def test_s8r1a_manifests_are_nested_and_balanced():
    manifests = {fraction: json.loads((OUT / f"data_budget_{fraction}.json").read_text(encoding="utf-8"))
                 for fraction in (25, 50, 100)}
    assert [manifests[f]["total_task_count"] for f in (25, 50, 100)] == [63, 126, 252]
    assert set(manifests[25]["selected_task_ids"]) < set(manifests[50]["selected_task_ids"]) < set(manifests[100]["selected_task_ids"])
    for fraction, manifest in manifests.items():
        assert all(len(ids) == fraction * 28 // 100 for ids in manifest["scene_wise_selected_task_ids"].values())
        assert manifest["sha256"] == object_hash({k: v for k, v in manifest.items() if k != "sha256"})


def test_s8r1a_independent_verifier_is_complete():
    verification = json.loads((OUT / "independent_verification.json").read_text(encoding="utf-8"))
    assert verification["passed"] == verification["total"] == 24
    assert verification["classification"] == "PASS_S8R1A_PRIOR_DATA_ABLATION_BC_COMPETITIVE_OR_BETTER"
    assert verification["s2_test_accessed"] is False
    assert verification["s6_development_test_accessed"] is False


def test_s8r1a_config_hashes_are_frozen():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["bc_config_hash"] == object_hash(BC_CONFIG)
    assert summary["diffusion_config_hash"] == object_hash(DIFFUSION_CONFIG)
    assert summary["run_count"] == 18
