from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from quadrotor_diffusion_ppo.evaluation.runtime import (
    DiffusionInferenceScheduler, RuntimeMode, checkpoint_selection,
)
from quadrotor_diffusion_ppo.ppo.contract import N_ENVS, PPO_CONFIG, TOTAL_ENV_STEPS
from quadrotor_diffusion_ppo.ppo.runtime import FrozenCheckpointCallback, capture_rng_state


ROOT = Path(__file__).resolve().parents[1]


class FakePrior:
    def predict_reference(self, observations, seeds):
        return np.asarray(observations, dtype=np.float32)[:, :3] + np.asarray(seeds)[:, None] * 0.0

    def predict_batched(self, observations, seeds):
        return self.predict_reference(observations, seeds)

    def predict_fast(self, observations, seeds):
        return self.predict_reference(observations, seeds)


class FakeModel:
    def save(self, path):
        Path(path).with_suffix(".zip").write_bytes(b"checkpoint")


def rng_equal(left, right):
    return (
        left["python"] == right["python"]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))
    )


def test_frozen_scientific_contract_is_not_changed():
    assert N_ENVS == 8
    assert TOTAL_ENV_STEPS == 500_000
    assert PPO_CONFIG["n_steps"] == 1024
    assert PPO_CONFIG["batch_size"] == 512
    assert PPO_CONFIG["n_epochs"] == 10


def test_scheduler_reports_mode_and_preserves_request_results():
    observation = np.arange(34, dtype=np.float32)
    for mode in RuntimeMode:
        with DiffusionInferenceScheduler(FakePrior(), mode, max_batch_size=4) as scheduler:
            result = scheduler.predict(observation, 17)
            diagnostics = scheduler.diagnostics()
        assert np.array_equal(result, observation[:3])
        assert diagnostics["runtime_mode"] == mode.value
        assert diagnostics["inference_requests"] == 1


def test_checkpoint_serialization_does_not_consume_rng(tmp_path):
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    callback = FrozenCheckpointCallback(tmp_path, (0, 50_000, 500_000))
    callback.model = FakeModel()
    before = capture_rng_state(); callback._save(0); after = capture_rng_state()
    assert rng_equal(before, after)
    assert callback.saved == [{"env_steps": 0, "path": str(tmp_path / "checkpoint_000000.zip"),
                               "rng_unchanged": True}]


def test_posthoc_selection_is_order_independent_and_retains_tie_break():
    records = [
        {"env_steps": 50_000, "success": 20, "unsafe": 10},
        {"env_steps": 100_000, "success": 22, "unsafe": 9},
        {"env_steps": 150_000, "success": 22, "unsafe": 8},
    ]
    key = lambda row: (row["success"], -row["unsafe"], -row["env_steps"])
    expected = checkpoint_selection(records, key)
    assert expected["env_steps"] == 150_000
    assert checkpoint_selection(list(reversed(records)), key) == expected


def test_runner_never_accesses_test_partition_and_profiles_all_worker_counts():
    source = (ROOT / "scripts" / "run_s6p0_runtime.py").read_text(encoding="utf-8")
    assert "WORKER_COUNTS = (8, 12, 16, 20)" in source
    assert '"--runtime-mode"' in source
    assert 'args.phase == "runtime-smoke"' in source
    assert "selected_runtime_mode=args.runtime_mode" in source
    assert 'tasks["test"]' not in source
    assert '"test_partition_accessed": False' in source
    assert "predict_original_reference" in source
    assert "predict_reference" in source
    assert "predict_batched" in source


def test_historical_runner_default_is_unchanged_and_verified_modes_are_explicit():
    source = (ROOT / "scripts" / "run_s5_residual_ppo.py").read_text(encoding="utf-8")
    assert 'runtime_mode: str = "historical_batched"' in source
    assert 'self.runtime_mode == "reference"' in source
    assert 'self.runtime_mode == "fast"' in source
    assert "predict_fast" in source


def test_runtime_module_has_no_algorithm_or_split_mutation():
    source = (ROOT / "src" / "quadrotor_diffusion_ppo" / "evaluation" / "runtime.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "reward" not in lowered
    assert "learning_rate" not in lowered
    assert "train" not in lowered
    assert "test" not in lowered
