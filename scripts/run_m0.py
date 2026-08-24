#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quadrotor_diffusion_ppo.paths import configure_external_imports
configure_external_imports()

from quadrotor_diffusion_ppo.evaluation.m0_rollout import evaluate_m0


if __name__ == "__main__":
    summary = evaluate_m0(ROOT)
    print(f"{summary['final_label']} scenes={summary['scene_count']} reference={summary['reference_completed_count']}/9 goal={summary['goal_reached_count']}/9")
    raise SystemExit(0 if summary["final_label"] == "PASS_GCOPTER_PIVOT_M0" else 1)
