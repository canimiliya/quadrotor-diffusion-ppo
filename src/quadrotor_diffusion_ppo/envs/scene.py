"""Read-only import of the nine frozen clean-reproduction scenes."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

from quadrotor_diffusion_ppo.paths import clean_reproduction_root as configured_clean_root

SCENE_IDS = ("OPEN_00", "OPEN_01", "OPEN_02", "BLOCK_00", "BLOCK_01", "BLOCK_02", "SBEND_00", "SBEND_01", "SBEND_02")
DEFAULT_CLEAN_ROOT = configured_clean_root()


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    family: str
    world_bounds: dict[str, np.ndarray]
    start: np.ndarray
    goal: np.ndarray
    obstacles: tuple[dict[str, np.ndarray | str], ...]
    corridors: tuple[dict[str, np.ndarray | str], ...]
    safety_radius: float
    control_hz: int
    physics_hz: int
    reference_hz: int
    max_reference_speed: float
    raw: dict


def clean_reproduction_root() -> Path:
    return configured_clean_root()


def _value(text: str):
    text = text.strip()
    if text.startswith("["):
        return list(ast.literal_eval(text))
    try:
        return float(text)
    except ValueError:
        return text


def _parse_yaml_subset(path: Path) -> dict:
    scene = {"world_bounds": {}, "start": {}, "goal": {}, "obstacles": [], "corridors": []}
    section = None
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- id:"):
            if section in {"obstacles", "corridors"}:
                current = {"id": line.split(":", 1)[1].strip()}
                scene[section].append(current)
            continue
        if line.endswith(":") and not line.startswith("-"):
            key = line[:-1]
            section = key if key in {"start", "goal", "obstacles", "corridors", "world_bounds"} else section
            current = None
            continue
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        parsed = _value(value)
        if key in {"scene_id", "family", "vehicle_safety_radius", "physics_hz", "control_hz", "reference_hz", "max_reference_speed"}:
            scene[key] = parsed
        elif section == "world_bounds" and key in {"x", "y", "z"}:
            scene["world_bounds"][key] = parsed
        elif section in {"start", "goal"} and key == "position":
            scene[section][key] = parsed
        elif section in {"obstacles", "corridors"} and current is not None and key in {"center", "size", "min", "max"}:
            current[key] = parsed
    return scene


def _to_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite with shape (3,), got {arr}")
    return arr


def _bounds(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (2,) or not np.isfinite(arr).all() or arr[0] > arr[1]:
        raise ValueError(f"{name} must be finite ordered bounds, got {arr}")
    return arr


def load_scene(scene_id: str) -> SceneSpec:
    if scene_id not in SCENE_IDS:
        raise KeyError(scene_id)
    path = clean_reproduction_root() / "scenes" / "pybullet" / f"{scene_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"frozen scene missing: {path}")
    raw = _parse_yaml_subset(path)
    obstacles = tuple({k: (_to_array(v, f"{scene_id}.{k}") if k in {"center", "size"} else v) for k, v in item.items()} for item in raw["obstacles"])
    corridors = tuple({k: (_to_array(v, f"{scene_id}.{k}") if k in {"min", "max"} else v) for k, v in item.items()} for item in raw["corridors"])
    return SceneSpec(scene_id=raw["scene_id"], family=raw["family"],
                     world_bounds={k: _bounds(v, f"{scene_id}.world_bounds.{k}") for k, v in raw["world_bounds"].items()},
                     start=_to_array(raw["start"]["position"], f"{scene_id}.start"), goal=_to_array(raw["goal"]["position"], f"{scene_id}.goal"),
                     obstacles=obstacles, corridors=corridors, safety_radius=float(raw["vehicle_safety_radius"]),
                     control_hz=int(raw["control_hz"]), physics_hz=int(raw["physics_hz"]), reference_hz=int(raw["reference_hz"]),
                     max_reference_speed=float(raw["max_reference_speed"]), raw=raw)


def load_all_scenes() -> list[SceneSpec]:
    scenes = [load_scene(scene_id) for scene_id in SCENE_IDS]
    if len(scenes) != 9 or len({scene.scene_id for scene in scenes}) != 9:
        raise AssertionError("M0 scene import must contain exactly nine unique scenes")
    return scenes
