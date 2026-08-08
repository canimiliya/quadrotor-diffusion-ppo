from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"D:\Desktop\research_progress_management\single_quad_ppo_diffusion\third_party\gym-pybullet-drones")
sys.path.insert(0, str(ROOT / "src"))
if EXTERNAL.exists():
    sys.path.insert(0, str(EXTERNAL))

