"""Analytic GCOPTER polynomial trajectory reader and velocity sampler."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Piece:
    duration: float
    coefficients: np.ndarray


class GcopterTrajectory:
    def __init__(self, pieces: list[Piece]):
        if not pieces:
            raise ValueError("empty GCOPTER trajectory")
        self.pieces = pieces
        self.durations = np.asarray([p.duration for p in pieces], dtype=float)
        self.cumulative = np.cumsum(self.durations)

    @property
    def total_duration(self) -> float:
        return float(self.cumulative[-1])

    @classmethod
    def from_csv(cls, path: str | Path) -> "GcopterTrajectory":
        rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
        pieces = []
        for row in rows:
            coeff = np.zeros((3, 6), dtype=float)
            for degree in range(6):
                for axis, key in enumerate(("x", "y", "z")):
                    coeff[axis, degree] = float(row[f"c{degree}{key}"])
            pieces.append(Piece(float(row["duration"]), coeff))
        return cls(pieces)

    def evaluate(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        t = float(np.clip(time_s, 0.0, self.total_duration))
        index = min(int(np.searchsorted(self.cumulative, t, side="left")), len(self.pieces) - 1)
        start = 0.0 if index == 0 else float(self.cumulative[index - 1])
        local = min(max(t - start, 0.0), self.pieces[index].duration)
        c = self.pieces[index].coefficients[:, ::-1]
        p = c @ np.asarray([local ** k for k in range(6)])
        v = c[:, 1:] @ np.asarray([k * local ** (k - 1) for k in range(1, 6)])
        a = c[:, 2:] @ np.asarray([k * (k - 1) * local ** (k - 2) for k in range(2, 6)])
        j = c[:, 3:] @ np.asarray([k * (k - 1) * (k - 2) * local ** (k - 3) for k in range(3, 6)])
        return p, v, a, j, index


def reference_path(project_root: Path, scene_id: str) -> Path:
    return project_root / ".deps" / "gcopter_reference" / scene_id / "reference_coefficients.csv"

