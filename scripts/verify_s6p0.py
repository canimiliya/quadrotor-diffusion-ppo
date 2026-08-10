"""Independent evidence verifier for S6-P0 runtime optimization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "s6p0"
REQUIRED = (
    "profile_baseline.json", "profile_optimized.json", "equivalence_audit.json",
    "runtime_benchmark.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (ARTIFACTS / name).is_file()]
    if missing:
        raise RuntimeError(f"missing S6-P0 evidence: {missing}")
    baseline = json.loads((ARTIFACTS / "profile_baseline.json").read_text(encoding="utf-8"))
    optimized = json.loads((ARTIFACTS / "profile_optimized.json").read_text(encoding="utf-8"))
    audit = json.loads((ARTIFACTS / "equivalence_audit.json").read_text(encoding="utf-8"))
    fast = audit["fast_cuda_graph_54_task"]
    reference = audit["reference_54_task"]
    batched = audit["batched_ddim_54_task"]
    train = optimized["train_smoke"]
    checks = {
        "hardware_cpu_24": baseline["hardware"]["logical_processors"] == 24,
        "hardware_rtx_5060_ti": "RTX 5060 Ti" in baseline["hardware"]["gpu"],
        "frozen_n_envs_8": baseline["hardware"]["n_envs_frozen"] == 8,
        "optimized_eager_bit_exact": audit["optimized_reference_bit_exact"],
        "cuda_graph_bit_exact": audit["cuda_graph_reference_bit_exact"],
        "cuda_graph_action_tolerance": audit["cuda_graph_reference_max_abs_error"] <= audit["action_tolerance"],
        "reference_54_identity": reference["success"] == 37 and reference["unsafe"] == 9
                                 and reference["outcome_mismatch_count"] == 0,
        "fast_54_identity": fast["success"] == 37 and fast["unsafe"] == 9
                            and fast["outcome_mismatch_count"] == 0,
        "batched_rejected": audit["fast_batched_diffusion_inference"] == "REJECTED"
                            and batched["outcome_mismatch_count"] > 0,
        "test_not_accessed": audit["test_partition_accessed"] is False,
        "worker_sweep_complete": {int(row["workers"]) for row in optimized["evaluation_worker_benchmarks"]}
                                 == {8, 12, 16, 20},
        "all_train_smokes_exact_8192": all(row["env_steps"] == 8192 for row in train.values()),
        "real_checkpoint_serialization_rng_unchanged": all(
            row["checkpoint_serialization_rng_unchanged"] for row in train.values()
        ),
        "fast_train_faster_than_reference": train["diffusion_integrated_fast"]["env_steps_per_s"]
                                            > train["diffusion_integrated_reference"]["env_steps_per_s"],
    }
    training_estimate_s = 500_000 / train["diffusion_integrated_fast"]["env_steps_per_s"]
    eleven_val_estimate_s = 11 * fast["wall_time_s"]
    projected_s = training_estimate_s + eleven_val_estimate_s
    verification = {
        "task": "S6-P0-HIGH-THROUGHPUT-EXPERIMENT-RUNTIME-OPTIMIZATION-V1",
        "independent_verification_pass": all(checks.values()),
        "checks": checks,
        "projected_500k_train_only_s_from_8192_smoke": training_estimate_s,
        "projected_11_val_s_from_measured_54_task": eleven_val_estimate_s,
        "projected_500k_plus_11_posthoc_val_s": projected_s,
        "projected_end_to_end_speedup_vs_5792s": 5792.2036968 / projected_s,
        "projection_is_not_a_full_500k_measurement": True,
        "evidence_sha256": {name: sha256(ARTIFACTS / name) for name in REQUIRED},
    }
    (ARTIFACTS / "independent_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))
    if not verification["independent_verification_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
