from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_s8r3_audit_artifacts_and_contract():
    out = ROOT / "artifacts" / "s8r3"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["test_accessed"] is False
    assert summary["ppo_run_count"] == 0
    assert summary["window_statistics"]["train_h16_windows"] == 4776168
    assert summary["window_statistics"]["val_h16_windows"] == 598153
    assert summary["window_statistics"]["updates_per_effective_pass"] == 9329
    assert set(summary["bc"]["checkpoints"]) == {"1", "5", "10", "20", "30"}
    assert set(summary["diffusion"]["checkpoints"]) == {"1", "5", "10", "20", "30"}


def test_s8r3_closed_loop_has_all_models_and_passes():
    out = ROOT / "artifacts" / "s8r3"
    rows = (out / "closed_loop_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 11
    assert "BC" in rows[1] and "DIFFUSION" in rows[-1]
    verification = json.loads((out / "independent_verification.json").read_text(encoding="utf-8"))
    assert verification["failed"] == 0
