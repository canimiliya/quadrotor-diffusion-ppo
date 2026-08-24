"""Portable paths for optional third-party inputs.

The research code consumes GCOPTER scenes and gym-pybullet-drones read-only.
Both locations are configured with environment variables so a checkout never
depends on the original author's computer layout.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = REPOSITORY_ROOT / ".deps"


def _configured_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def clean_reproduction_root() -> Path:
    """Return the read-only GCOPTER-Clean-Reproduction checkout."""
    return _configured_path(
        "GCOPTER_CLEAN_REPRODUCTION",
        DEFAULT_EXTERNAL_ROOT / "GCOPTER-Clean-Reproduction",
    )


def gym_pybullet_drones_root() -> Path:
    """Return the read-only gym-pybullet-drones source checkout."""
    return _configured_path(
        "GYM_PYBULLET_DRONES_ROOT",
        DEFAULT_EXTERNAL_ROOT / "gym-pybullet-drones",
    )


def gcopter_planner_path() -> Path:
    """Return the optional prebuilt GCOPTER YAML planner executable."""
    return _configured_path(
        "GCOPTER_YAML_SCENE_PLANNER",
        DEFAULT_EXTERNAL_ROOT / "gcopter_reference" / "gcopter_yaml_scene_planner",
    )


def configure_external_imports() -> None:
    """Add the external gym source to ``sys.path`` when it is available."""
    root = gym_pybullet_drones_root()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
