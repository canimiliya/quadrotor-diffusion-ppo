# Local storage and environment audit

Audit date: 2026-08-24. This document records the read-only inventory made
before cleanup and the resulting repository policy. No virtual environment,
dataset, checkpoint, third-party checkout, or formal result was deleted or
moved.

## 1. Formal project directory and Git baseline

| field | observed value |
|---|---|
| formal project directory | `D:\Desktop\my_project\quadrotor_diffusion_ppo` |
| remote | `https://github.com/canimiliya/quadrotor-diffusion-ppo.git` |
| branch | `agent/s8r4-unet-diffusion-audit-v1` |
| HEAD at audit start | `d82dd8a6ce59fc7054cd093386a84de8176de5d9` |
| initial Git status | branch matched its origin branch; pre-existing untracked `docs/local_orchestration/` and three GCOPTER RViz helper scripts were present |
| tracked working-tree bytes | 23,445,680 |
| untracked non-ignored direct-file bytes at audit start | 36,367 |
| ignored research payloads | approximately 3.9 GB across `artifacts/`, `checkpoints/`, `.deps/`, and logs; these are not counted as ordinary Git untracked files |

The HEAD value above is the exact commit read before this cleanup. The final
status after the requested files are added is reported separately; existing
untracked S8-R4/RViz work was preserved rather than overwritten.

## 2. Large directories and local storage

| path | size | purpose | reproducible | safe_to_delete |
|---|---:|---|---|---|
| `artifacts/` | 2,703,540,835 B | all stage evidence, datasets, logs, maps and derived outputs | mixed; contracts and frozen inputs required | NO; review children individually |
| `artifacts/s8r2_10k/` | 2,111,208,813 B | frozen 10K expert data, manifests and generation cache | NPZ rebuildable but expensive | NO until independent rebuild |
| `artifacts/s8r2_10k/_cache/` | about 1.08 GB | planner/rollout generation cache | yes from frozen generation inputs | LIKELY_SAFE after review |
| `artifacts/s3r2/` | 324,131,665 B | S3-R2 recovery data, planner outputs and evidence | partially; recovery is historical evidence | NO |
| `artifacts/s7/` | 163,667,217 B | fresh-holdout and feasibility diagnostics | partially; failed cases are historical evidence | LIKELY_SAFE after review |
| `artifacts/s8r4/` | 3,189,644 B | S8-R4 audit summaries, VAL rollouts, logs and plots | derived from retained checkpoints/data | NO while audit is under review |
| `checkpoints/` | 1,197,929,814 B | all historical trained models and interrupted runs | expensive or not exactly reproducible | NO |
| `checkpoints/s8r4/` | 942,441,935 B | five 188 MB U-Net checkpoints | only by rerunning the audit | NO |
| `checkpoints/s8r3/` | 24,389,710 B | BC and Diffusion pass 01/05/10/20/30 checkpoints | expensive but scripted | NO |
| `checkpoints/s8r3_interrupted_20260811/` | 19,396,888 B | interrupted historical S8-R3 evidence | not a formal replacement | NO; preserve history |
| `.deps/gcopter_reference/` | 13,067,803 B | external planner and frozen reference trajectories | planner can be rebuilt; coefficients are evidence inputs | NO until bootstrap verified |
| `.deps/s2_probe/` | 1,217,186 B | probe outputs | yes | LIKELY_SAFE after review |
| `logs/` | 6,684,193 B | runtime logs | usually yes, but historical | LIKELY_SAFE after review |
| `.pytest_cache/`, `__pycache__/` | generated caches | Python/test cache | yes | SAFE_TO_DELETE |

Top-level `dataset/`, `data/`, `cache/`, `.cache/`, `outputs/`, `temporary/`,
and `temp/` directories were not present as independent project roots at the
audit start. Their roles are represented by the stage-specific directories
above. Ignored nested `_work`, cache, and failed-preflight directories remain
untouched.

## 3. Environments found

Conda was not on the ordinary shell PATH, so the inventory used the installed
Conda root under `D:\anaconda`. No project-local `.venv` or `venv` was found.

| environment | evidence | classification |
|---|---|---|
| `D:\anaconda\envs\smd-blackwell` | S8-R4 logs invoke `conda.exe run -n smd-blackwell`; CUDA formal environment captured in `artifacts/reproducibility/current_environment.json` | KEEP / ACTIVE FORMAL |
| `D:\anaconda\envs\dp_quad_py310` | prior project visualization reports use this interpreter; CPU PyTorch 2.13.0 environment | KEEP until owner confirms no longer needed |
| `D:\anaconda\envs\quad_audit` | name indicates audit use, but no current S8-R4 proof | UNKNOWN |
| `D:\anaconda\envs\smd` | older SMD environment; not the S8-R4 formal runtime | PROBABLY_LEGACY |
| `D:\anaconda\envs\rl` | separate robotics/RL environment; no proof it belongs to this project | UNKNOWN |
| `D:\anaconda\envs\dp_quad` | separate quadrotor environment; no current S8-R4 proof | UNKNOWN |
| `D:\anaconda\envs\pmm_uav_build` | PMM/UAV build environment, outside this project | PROBABLY_LEGACY / OTHER PROJECT |
| `D:\anaconda\envs\pmm_uav_build_modern` | PMM/UAV build environment, outside this project | PROBABLY_LEGACY / OTHER PROJECT |
| `D:\anaconda\envs\python_robotics` | separate learning project environment | KEEP / OTHER PROJECT |
| `D:\anaconda\envs\uav_sway` | separate UAV sway project environment | KEEP / OTHER PROJECT |
| `D:\anaconda\envs\v9_neural_predictor` | separate project environment | UNKNOWN / OTHER PROJECT |

No environment is authorized for automatic deletion. Directory names alone are
not sufficient evidence for deletion.

## 4. Real import scan and minimum dependency map

The scan covered `src/`, `scripts/`, and `tests/`, rather than trusting the old
README. The import roots were:

| import root | package / role | management |
|---|---|---|
| `numpy` | arrays, contracts, datasets | `requirements-base.txt` |
| `scipy` | KD-tree audit and numerical utilities | `requirements-base.txt` |
| `torch` | Diffusion, U-Net, BC and PPO policy modules | explicit PyTorch installer; also project dependency |
| `gymnasium` | spaces and environment interface | `requirements-base.txt` |
| `pybullet` | physics and ray observations | `requirements-base.txt` |
| `gym_pybullet_drones` | external VelocityAviary implementation | external pinned checkout, not copied into Git |
| `stable_baselines3` | PPO implementation and policy base classes | `requirements-base.txt` |
| `matplotlib` | plots and visualization scripts | `requirements-base.txt` |
| `PyYAML` (`yaml`) | scene and visualization YAML | `requirements-base.txt` |
| `Pillow` (`PIL`) | visualization image composition/tests | `requirements-base.txt` |
| `psutil` | runtime audit telemetry | `requirements-base.txt` |
| `transforms3d` | dependency of the external drone stack | `requirements-base.txt` |
| `pytest` | tests only | `requirements-dev.txt` |

`pandas` and `tqdm` were explicitly checked and are not imported by this
project's `src/`, `scripts/`, or `tests/`; they are therefore not added to the
minimal dependency files even though they occur in the historical formal
environment's full freeze.

## 5. Formal S8-R4 environment

The actual recent S8-R4 route is `smd-blackwell`, not the obsolete path in the
old README:

- executable: `D:\anaconda\envs\smd-blackwell\python.exe`
- Python 3.9.25
- PyTorch 2.7.1+cu128; CUDA runtime 12.8; CUDA available
- GPU: NVIDIA GeForce RTX 5060 Ti, driver 581.29, 16,311 MiB
- NumPy 1.24.1; SciPy 1.10.1; PyBullet 3.2.7
- Gymnasium 1.1.1; Stable-Baselines3 2.6.0

The complete, non-minimal `pip freeze` is saved at
`artifacts/reproducibility/pip_freeze_s8r4_smd-blackwell.txt`. It is evidence
of the historical environment, not the project's minimum dependency file.

## 6. Personal absolute path audit

Before cleanup, the runnable code and tests contained historical paths to an
old `single_quad_ppo_diffusion` checkout and a personal GCOPTER checkout. They
were replaced by `GCOPTER_CLEAN_REPRODUCTION`, `GYM_PYBULLET_DRONES_ROOT`,
`GCOPTER_YAML_SCENE_PLANNER`, repository-relative `.deps/`, and runtime temp
directories. A post-edit scan of `src/`, `scripts/`, `tests/`, README,
packaging, and third-party manifests found no personal drive-letter paths.

The requested environment record intentionally retains the formal interpreter
path, and this audit intentionally records the formal project directory; those
are provenance fields, not runtime requirements.

## 7. Cleanup boundary

Only packaging, documentation, path configuration, bootstrap, and smoke-test
files were changed. No model training, formal TEST, data deletion, checkpoint
deletion, virtual-environment deletion, third-party source modification, or
scientific algorithm change was performed.

## 8. Fresh-environment verification

The new temporary environment `D:\anaconda\envs\quadrotor-repro-test` was
created from Python 3.10.20 and installed through the new environment files
and bootstrap flow. The final verification used the cloned external inputs at
the pinned refs:

- `python scripts/verify_installation.py`: PASS; CUDA visible, NumPy/Torch/
  PyBullet/Gymnasium/project imports passed, observation/action dimensions were
  34/3, and Diffusion MLP, temporal U-Net, BC, and PPO imports/forward checks
  passed.
- `python -m pytest -q`: PASS, `121 passed`, 13 dependency deprecation warnings.
- No training command and no formal TEST evaluation was run as part of this
  cleanup.
