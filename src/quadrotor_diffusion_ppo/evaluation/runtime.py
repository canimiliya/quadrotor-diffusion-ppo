"""High-throughput evaluation primitives with auditable runtime semantics."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
import threading
import time
from typing import Callable, Sequence

import numpy as np

from quadrotor_diffusion_ppo.ppo.residual import FrozenDiffusionPrior


class RuntimeMode(str, Enum):
    REFERENCE = "reference"
    FAST = "fast"


@dataclass
class InferenceRequest:
    observation: np.ndarray
    seed: int
    completed: threading.Event
    result: np.ndarray | None = None
    error: BaseException | None = None


class DiffusionInferenceScheduler:
    """Single-owner CUDA scheduler shared by independent CPU environments.

    Reference mode retains one-by-one denoising.  Fast mode combines requests
    only when its separate numerical and closed-loop equivalence gates pass.
    """

    def __init__(self, prior: FrozenDiffusionPrior, mode: RuntimeMode | str,
                 *, max_batch_size: int = 20, gather_timeout_s: float = 0.0005,
                 audit_unverified_batched: bool = False):
        self.prior = prior
        self.mode = RuntimeMode(mode)
        self.max_batch_size = int(max_batch_size)
        self.gather_timeout_s = float(gather_timeout_s)
        self.audit_unverified_batched = bool(audit_unverified_batched)
        if self.audit_unverified_batched and self.mode is not RuntimeMode.FAST:
            raise ValueError("unverified batched inference is audit-only")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self._requests: Queue[InferenceRequest | None] = Queue()
        self._thread = threading.Thread(target=self._serve, name="diffusion-inference", daemon=True)
        self._closed = False
        self.batch_sizes: list[int] = []
        self.inference_wall_time_s = 0.0
        self._thread.start()

    def predict(self, observation: np.ndarray, seed: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("inference scheduler is closed")
        request = InferenceRequest(
            np.asarray(observation, dtype=np.float32).reshape(34), int(seed), threading.Event()
        )
        self._requests.put(request)
        request.completed.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("inference request completed without a result")
        return request.result

    def _serve(self) -> None:
        while True:
            first = self._requests.get()
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self.gather_timeout_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._requests.get(timeout=remaining)
                except Empty:
                    break
                if item is None:
                    self._requests.put(None)
                    break
                batch.append(item)
            observations = np.stack([item.observation for item in batch])
            seeds = [item.seed for item in batch]
            started = time.perf_counter()
            try:
                if self.audit_unverified_batched:
                    actions = self.prior.predict_batched(observations, seeds)
                elif self.mode is RuntimeMode.REFERENCE:
                    actions = self.prior.predict_reference(observations, seeds)
                else:
                    actions = self.prior.predict_fast(observations, seeds)
                self.inference_wall_time_s += time.perf_counter() - started
                self.batch_sizes.append(len(batch))
                for item, action in zip(batch, actions):
                    item.result = np.asarray(action, dtype=np.float32)
            except BaseException as error:  # propagate CUDA failures to every waiter
                for item in batch:
                    item.error = error
            finally:
                for item in batch:
                    item.completed.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join()

    def diagnostics(self) -> dict[str, float | int | str]:
        count = len(self.batch_sizes)
        return {
            "runtime_mode": self.mode.value,
            "inference_path": (
                "unverified_batched_audit" if self.audit_unverified_batched
                else "cuda_graph_exact_batch_one" if self.mode is RuntimeMode.FAST
                else "optimized_eager_exact_batch_one"
            ),
            "inference_batches": count,
            "inference_requests": int(sum(self.batch_sizes)),
            "mean_inference_batch_size": float(np.mean(self.batch_sizes)) if count else 0.0,
            "max_inference_batch_size": max(self.batch_sizes, default=0),
            "inference_wall_time_s": self.inference_wall_time_s,
        }

    def __enter__(self) -> "DiffusionInferenceScheduler":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def checkpoint_selection(records: Sequence[dict],
                         key: Callable[[dict], tuple]) -> dict:
    """Select post-hoc with the same deterministic rule used online."""
    if not records:
        raise ValueError("at least one evaluation record is required")
    return max(records, key=key)


def rng_state_equal(before: dict, after: dict) -> bool:
    """Compare serialized Python/NumPy/Torch RNG snapshots."""
    return (
        before["python"] == after["python"]
        and np.array_equal(before["numpy"][1], after["numpy"][1])
        and before["numpy"][:1] == after["numpy"][:1]
        and before["numpy"][2:] == after["numpy"][2:]
        and bool((before["torch_cpu"] == after["torch_cpu"]).all())
        and all(bool((left == right).all()) for left, right in zip(
            before.get("torch_cuda", []), after.get("torch_cuda", [])
        ))
    )
