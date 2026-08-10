"""Training/evaluation orchestration that is independent of PPO mathematics."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Callable, Sequence

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


class FrozenCheckpointCallback(BaseCallback):
    """Save preregistered checkpoints without evaluating or consuming RNG."""

    def __init__(self, checkpoint_dir: Path, targets: Sequence[int]):
        super().__init__(verbose=0)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.targets = tuple(int(value) for value in targets)
        if tuple(sorted(set(self.targets))) != self.targets:
            raise ValueError("checkpoint targets must be unique and increasing")
        self.next_index = 0
        self.saved: list[dict[str, object]] = []

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        while self.next_index < len(self.targets) and self.targets[self.next_index] == 0:
            self._save(0)

    def _save(self, target: int) -> None:
        before = capture_rng_state()
        path = self.checkpoint_dir / f"checkpoint_{target:06d}"
        self.model.save(str(path))
        after = capture_rng_state()
        rng_unchanged = (
            before["python"] == after["python"]
            and np.array_equal(before["numpy"][1], after["numpy"][1])
            and before["numpy"][:1] == after["numpy"][:1]
            and before["numpy"][2:] == after["numpy"][2:]
            and torch.equal(before["torch_cpu"], after["torch_cpu"])
            and all(torch.equal(left, right) for left, right in zip(
                before["torch_cuda"], after["torch_cuda"]
            ))
        )
        if not rng_unchanged:
            raise RuntimeError("checkpoint serialization changed training RNG state")
        self.saved.append({"env_steps": target, "path": str(path.with_suffix(".zip")),
                           "rng_unchanged": True})
        self.next_index += 1

    def _on_step(self) -> bool:
        env_steps = int(self.num_timesteps)
        while self.next_index < len(self.targets) and env_steps >= self.targets[self.next_index]:
            self._save(self.targets[self.next_index])
        return True


def posthoc_evaluate(
    checkpoints: Sequence[dict[str, object]],
    load_model: Callable[[Path], object],
    evaluate: Callable[[object, int], dict],
) -> list[dict]:
    """Evaluate every frozen checkpoint after training, in declared order."""
    records = []
    for checkpoint in checkpoints:
        env_steps = int(checkpoint["env_steps"])
        path = Path(str(checkpoint["path"]))
        records.append({"env_steps": env_steps, "path": str(path),
                        "summary": evaluate(load_model(path), env_steps)})
    return records
