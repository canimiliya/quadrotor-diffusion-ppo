from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quadrotor_diffusion_ppo.paths import configure_external_imports
configure_external_imports()
