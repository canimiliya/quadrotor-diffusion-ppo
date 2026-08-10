"""Frozen S3-R2 diffusion prior and auditable residual-action composition."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from quadrotor_diffusion_ppo.diffusion.model import ConditionalDiffusionMLP, unit_ball_squash
from quadrotor_diffusion_ppo.diffusion.schedule import DiffusionSchedule


EXPECTED_S3R2_SHA256 = "98de9a5d765ec1aeb48648497ac44ec98a6b6a071a607a01b4e49f3e13ab76cd"
EVAL_BASE_SEED = 20260811
DDIM_STEPS = 10
RESIDUAL_SCALE = 0.25
RESIDUAL_LOG_STD_INIT = -2.0
SUPPORT_TOLERANCE = 1.0e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_prior_seed(task_id: str) -> int:
    """Exact S3 deployment seed for the first action of a task."""
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int((EVAL_BASE_SEED + int.from_bytes(digest[:8], "little")) % (2**63 - 1))


def compose_residual_action(
    prior: np.ndarray, residual: np.ndarray, alpha: float = RESIDUAL_SCALE
) -> tuple[np.ndarray, np.ndarray]:
    """Compose actions and project only vectors outside the closed unit ball.

    Returns the composed action and a per-row boolean projection mask.  A zero
    residual is bit-identical to an in-support prior.
    """
    prior_array = np.asarray(prior, dtype=np.float32)
    residual_array = np.asarray(residual, dtype=np.float32)
    if prior_array.shape != residual_array.shape or prior_array.shape[-1] != 3:
        raise ValueError("prior and residual must have matching (..., 3) shapes")
    if not (0.0 < float(alpha) <= 0.5):
        raise ValueError("residual scale must satisfy 0 < alpha <= 0.5")
    z = prior_array + np.float32(alpha) * residual_array
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    projected = norms[..., 0] > 1.0
    result = np.where(projected[..., None], z / np.maximum(norms, 1.0e-12), z)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite composed residual action")
    return result.astype(np.float32, copy=False), projected


class FrozenDiffusionPrior:
    """Load and execute the immutable S3-R2 bounded-x0 DDIM policy."""

    def __init__(self, checkpoint: Path, device: torch.device | str = "cuda"):
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        if self.checkpoint_sha256 != EXPECTED_S3R2_SHA256:
            raise RuntimeError("BLOCKED_S5_DIFFUSION_CHECKPOINT_IDENTITY")
        self.device = torch.device(device)
        payload = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        required = {
            "model_frozen": True,
            "prediction_target": "x0",
            "action_support": "unit_ball_squash",
            "diffusion_steps": 100,
            "ddim_steps": 10,
            "ddim_eta": 0.0,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise RuntimeError("BLOCKED_S5_DIFFUSION_CHECKPOINT_METADATA")
        self.model = ConditionalDiffusionMLP(**payload["model_config"]).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.schedule = DiffusionSchedule(int(payload["diffusion_steps"])).to(self.device)
        self.mean = torch.as_tensor(payload["observation_mean"], dtype=torch.float32, device=self.device)
        raw_std = torch.as_tensor(payload["observation_std"], dtype=torch.float32, device=self.device)
        self.scale = torch.clamp(raw_std, min=1.0e-6)
        if self.mean.shape != (34,) or self.scale.shape != (34,):
            raise RuntimeError("BLOCKED_S5_DIFFUSION_NORMALIZATION_SHAPE")
        self.payload = payload
        # DDIM inference is always ten steps under the frozen S3 contract.  The
        # original sampler rebuilt these tensors and synchronized diagnostics
        # back to the CPU for every action.  Cache execution-only tensors once;
        # this changes no floating-point operation in the denoising recurrence.
        schedule = torch.linspace(
            self.schedule.steps - 1, 0, DDIM_STEPS, device=self.device
        ).round().long()
        self._ddim_timesteps = tuple(int(value) for value in schedule.cpu().tolist())
        self._ddim_t_batch_one = tuple(
            torch.full((1,), value, device=self.device, dtype=torch.long)
            for value in self._ddim_timesteps
        )
        self._ddim_alpha = tuple(self.schedule.alpha_bars[value] for value in self._ddim_timesteps)
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._graph_condition: torch.Tensor | None = None
        self._graph_initial: torch.Tensor | None = None
        self._graph_output: torch.Tensor | None = None

    def _conditions(self, observations: np.ndarray) -> torch.Tensor:
        values = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != 34:
            raise ValueError("prior observations must have shape (B, 34)")
        normalized = (values - self.mean) / self.scale
        if not torch.isfinite(normalized).all():
            raise FloatingPointError("nonfinite normalized diffusion observation")
        return normalized

    @torch.inference_mode()
    def predict_original_reference(self, observations: np.ndarray, seeds: Sequence[int]) -> np.ndarray:
        """Unmodified S3-R2 sampler retained as the audit oracle."""
        conditions = self._conditions(observations)
        if len(conditions) != len(seeds):
            raise ValueError("one seed is required per observation")
        actions = []
        for condition, seed in zip(conditions, seeds):
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            sampled = self.schedule.ddim_sample_x0(
                self.model, condition.unsqueeze(0), steps=DDIM_STEPS, generator=generator
            )
            actions.append(sampled[0, 0])
        return torch.stack(actions).cpu().numpy().astype(np.float32)

    def _initial_noise(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn(
            (1, self.model.horizon, self.model.action_dim),
            device=self.device,
            generator=generator,
        )

    def _ddim_from_initial(self, condition: torch.Tensor, sample: torch.Tensor,
                           *, trace: bool = False) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Run the exact bounded-x0 recurrence without unused host syncs."""
        records: list[dict[str, torch.Tensor]] = []
        for index, (t, t_batch, alpha_bar) in enumerate(zip(
            self._ddim_timesteps, self._ddim_t_batch_one, self._ddim_alpha
        )):
            input_sample = sample
            raw = self.model.predict_raw(input_sample, condition, t_batch)
            x0 = unit_ball_squash(raw)
            epsilon = (input_sample - alpha_bar.sqrt() * x0) / (1.0 - alpha_bar).sqrt()
            if index == len(self._ddim_timesteps) - 1:
                sample = x0
            else:
                alpha_next = self._ddim_alpha[index + 1]
                sample = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * epsilon
            if trace:
                records.append({
                    "timestep": torch.tensor(t),
                    "x_t": input_sample.detach().clone(),
                    "raw_x0": raw.detach().clone(),
                    "bounded_x0": x0.detach().clone(),
                    "epsilon": epsilon.detach().clone(),
                    "next_sample": sample.detach().clone(),
                })
        return sample, records

    @torch.inference_mode()
    def predict_reference(self, observations: np.ndarray, seeds: Sequence[int]) -> np.ndarray:
        """Optimized exact batch-one path, numerically audited against the oracle."""
        conditions = self._conditions(observations)
        if len(conditions) != len(seeds):
            raise ValueError("one seed is required per observation")
        actions = []
        for condition, seed in zip(conditions, seeds):
            sampled, _ = self._ddim_from_initial(condition.unsqueeze(0), self._initial_noise(int(seed)))
            actions.append(sampled[0, 0])
        return torch.stack(actions).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def _capture_reference_graph(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph reference inference requires CUDA")
        self._graph_condition = torch.empty((1, 34), dtype=torch.float32, device=self.device)
        self._graph_initial = torch.empty(
            (1, self.model.horizon, self.model.action_dim), dtype=torch.float32, device=self.device
        )
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._ddim_from_initial(self._graph_condition, self._graph_initial)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._graph_output, _ = self._ddim_from_initial(
                self._graph_condition, self._graph_initial
            )

    @torch.inference_mode()
    def predict_fast(self, observations: np.ndarray, seeds: Sequence[int]) -> np.ndarray:
        """Verified fixed-shape CUDA Graph of the exact batch-one recurrence."""
        if self.device.type != "cuda":
            return self.predict_reference(observations, seeds)
        conditions = self._conditions(observations)
        if len(conditions) != len(seeds):
            raise ValueError("one seed is required per observation")
        if self._cuda_graph is None:
            self._capture_reference_graph()
        if self._graph_condition is None or self._graph_initial is None or self._graph_output is None:
            raise RuntimeError("CUDA Graph capture did not initialize static buffers")
        actions = []
        for condition, seed in zip(conditions, seeds):
            self._graph_condition.copy_(condition.unsqueeze(0))
            self._graph_initial.copy_(self._initial_noise(int(seed)))
            self._cuda_graph.replay()
            actions.append(self._graph_output[0, 0].clone())
        return torch.stack(actions).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def trace_reference(self, observation: np.ndarray, seed: int) -> list[dict[str, np.ndarray | int]]:
        """Expose every frozen DDIM intermediate for equivalence audits."""
        condition = self._conditions(observation)[0].unsqueeze(0)
        _, records = self._ddim_from_initial(condition, self._initial_noise(seed), trace=True)
        return [
            {key: (int(value.item()) if key == "timestep" else value.cpu().numpy())
             for key, value in record.items()}
            for record in records
        ]

    @torch.inference_mode()
    def predict_batched(self, observations: np.ndarray, seeds: Sequence[int]) -> np.ndarray:
        """Batched DDIM with per-item noise identical to S3-R2 reference seeds."""
        conditions = self._conditions(observations)
        if len(conditions) != len(seeds):
            raise ValueError("one seed is required per observation")
        initial = []
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            initial.append(torch.randn((1, self.model.horizon, self.model.action_dim),
                                       device=self.device, generator=generator))
        sample = torch.cat(initial, dim=0)
        for index, t in enumerate(self._ddim_timesteps):
            t_batch = torch.full((len(sample),), t, device=self.device, dtype=torch.long)
            raw = self.model.predict_raw(sample, conditions, t_batch)
            x0 = unit_ball_squash(raw)
            epsilon = (
                sample - self.schedule.alpha_bars[t].sqrt() * x0
            ) / (1.0 - self.schedule.alpha_bars[t]).sqrt()
            if index == len(self._ddim_timesteps) - 1:
                sample = x0
            else:
                alpha_next = self._ddim_alpha[index + 1]
                sample = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * epsilon
        return sample[:, 0].cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def trace_batched(self, observations: np.ndarray, seeds: Sequence[int]) -> list[dict[str, Any]]:
        """Trace batched DDIM at each layer using reference-identical initial noise."""
        conditions = self._conditions(observations)
        if len(conditions) != len(seeds):
            raise ValueError("one seed is required per observation")
        sample = torch.cat([self._initial_noise(int(seed)) for seed in seeds], dim=0)
        records: list[dict[str, Any]] = []
        for index, (t, alpha_bar) in enumerate(zip(self._ddim_timesteps, self._ddim_alpha)):
            input_sample = sample
            t_batch = torch.full((len(sample),), t, device=self.device, dtype=torch.long)
            raw = self.model.predict_raw(input_sample, conditions, t_batch)
            x0 = unit_ball_squash(raw)
            epsilon = (input_sample - alpha_bar.sqrt() * x0) / (1.0 - alpha_bar).sqrt()
            if index == len(self._ddim_timesteps) - 1:
                sample = x0
            else:
                alpha_next = self._ddim_alpha[index + 1]
                sample = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * epsilon
            records.append({
                "timestep": t,
                "x_t": input_sample.cpu().numpy(),
                "raw_x0": raw.cpu().numpy(),
                "bounded_x0": x0.cpu().numpy(),
                "epsilon": epsilon.cpu().numpy(),
                "next_sample": sample.cpu().numpy(),
            })
        return records

    @property
    def frozen(self) -> bool:
        return not self.model.training and all(not parameter.requires_grad for parameter in self.model.parameters())

    @property
    def gradient_present(self) -> bool:
        return any(parameter.grad is not None for parameter in self.model.parameters())
