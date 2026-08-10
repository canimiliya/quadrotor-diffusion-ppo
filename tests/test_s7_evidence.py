import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_s7_final_evidence_contract():
    summary = json.loads((ROOT / "artifacts/s7/summary.json").read_text())
    assert summary["current_distribution"]["prior_advantage_supported"] is True
    assert summary["current_distribution"]["diffusion_specific_supported"] is False
    assert summary["fresh"]["diffusion_success"] == 10
    assert summary["fresh"]["bc_success"] == 4
    assert summary["fresh"]["pure_success"] == [0, 0, 0]
    assert summary["classification"]["safety"] == "NOT_SUPPORTED"


def test_s7_fresh_holdout_is_frozen_and_balanced():
    frozen = json.loads((ROOT / "artifacts/s7/fresh_holdout/frozen_manifest.json").read_text())
    with (ROOT / "artifacts/s7/fresh_holdout/task_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert frozen["topology_count"] == 6 and frozen["task_count"] == 54
    assert frozen["used_for_training_or_selection"] is False
    assert {family: sum(r["family"] == family for r in rows) for family in frozen["families"]} == {family: 9 for family in frozen["families"]}


def test_s7_controller_handoff_tables_are_complete():
    with (ROOT / "artifacts/s7/online_seed_curves.csv").open(encoding="utf-8-sig", newline="") as handle:
        curves = list(csv.DictReader(handle))
    with (ROOT / "artifacts/s7/current_prior_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        priors = list(csv.DictReader(handle))
    with (ROOT / "artifacts/s7/fresh_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        fresh = list(csv.DictReader(handle))
    assert len(curves) == 3 * 3 * 11
    assert len(priors) == 2 * 4
    assert len(fresh) == 12 * 7
    assert {"collision", "ground_contact", "unsafe"}.issubset(fresh[0])
