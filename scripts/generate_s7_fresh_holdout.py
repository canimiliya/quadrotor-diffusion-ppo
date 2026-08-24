"""Generate and freeze a policy-blind 54-task fresh-topology holdout."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from quadrotor_diffusion_ppo.paths import configure_external_imports
configure_external_imports()

import numpy as np

from quadrotor_diffusion_ppo.envs.scene import SceneSpec, clean_reproduction_root
from quadrotor_diffusion_ppo.expert.s2_dataset import TaskSpec, rollout_task
from quadrotor_diffusion_ppo.expert.trajectory import GcopterTrajectory


GENERATION_SEED = 20260816
TASKS_PER_TOPOLOGY = 9
MAX_CANDIDATES = 30
OUT = ROOT / "artifacts" / "s7" / "fresh_holdout"
WORK = OUT / "feasibility_work"
PLANNER = ROOT / ".deps" / "gcopter_reference" / "gcopter_yaml_scene_planner"
CONTRACT = {"speed_limit_m_per_s": 0.801, "ray_max_range_m": 3.0,
            "goal_tolerance_m": 0.30, "post_reference_settle_s": 2.0}


def box(identifier, center, size):
    return {"id": identifier, "center": list(center), "size": list(size)}


def corridor(identifier, low, high):
    return {"id": identifier, "min": list(low), "max": list(high)}


TOPOLOGIES = (
    {"scene_id": "FRESH_CROSSOVER_00", "family": "DIAGONAL_CROSSOVER", "bounds": [[-0.5, 7.5], [-2.8, 2.8], [0.2, 2.8]],
     "start": [0, -1.1, 1], "goal": [7, 1.1, 1], "obstacles": [box("O0", [3.5, 0.15, 1.1], [1.2, 1.7, 2.2])],
     "corridors": [corridor("C0", [-0.2,-1.6,0.7],[2.4,-0.6,1.4]), corridor("C1", [2.0,-1.8,0.7],[4.8,-0.9,1.5]), corridor("C2", [4.4,-1.3,0.7],[5.6,1.5,1.5]), corridor("C3", [5.2,0.6,0.7],[7.2,1.6,1.4])]},
    {"scene_id": "FRESH_DOUBLE_00", "family": "DOUBLE_ALTERNATING", "bounds": [[-0.5, 8.5], [-2.5, 2.5], [0.2, 2.8]],
     "start": [0,0,1], "goal": [8,0,1], "obstacles": [box("O0",[2.8,0.5,1.1],[1.0,1.6,2.2]), box("O1",[6.2,-0.5,1.1],[1.1,1.7,2.2])],
     "corridors": [corridor("C0",[-0.2,-0.6,0.7],[1.5,0.6,1.4]), corridor("C1",[1.2,-1.5,0.7],[2.7,-0.4,1.5]), corridor("C2",[2.4,-1.5,0.7],[4.3,-0.4,1.5]), corridor("C3",[4.0,-1.4,0.7],[5.0,1.4,1.5]), corridor("C4",[4.7,0.5,0.7],[6.1,1.5,1.5]), corridor("C5",[5.8,0.5,0.7],[7.1,1.5,1.5]), corridor("C6",[6.8,-0.6,0.7],[8.2,0.6,1.4])]},
    {"scene_id": "FRESH_CHICANE_00", "family": "TRIPLE_CHICANE", "bounds": [[-0.5, 11.5], [-2.5, 2.5], [0.2, 2.8]],
     "start": [0,0,1], "goal": [11,0,1], "obstacles": [box("O0",[2.5,0.65,1.1],[1.0,1.8,2.2]), box("O1",[5.5,-0.65,1.1],[1.0,1.8,2.2]), box("O2",[8.5,0.65,1.1],[1.0,1.8,2.2])],
     "corridors": [corridor("C0",[-0.2,-0.6,0.7],[1.4,0.6,1.4]), corridor("C1",[1.1,-1.7,0.7],[2.5,-0.4,1.5]), corridor("C2",[2.2,-1.7,0.7],[3.7,-0.4,1.5]), corridor("C3",[3.4,-1.6,0.7],[4.7,1.6,1.5]), corridor("C4",[4.4,0.4,0.7],[5.5,1.7,1.5]), corridor("C5",[5.2,0.4,0.7],[6.7,1.7,1.5]), corridor("C6",[6.4,-1.6,0.7],[7.7,1.6,1.5]), corridor("C7",[7.4,-1.7,0.7],[8.5,-0.4,1.5]), corridor("C8",[8.2,-1.7,0.7],[9.7,-0.4,1.5]), corridor("C9",[9.4,-0.6,0.7],[11.2,0.6,1.4])]},
    {"scene_id": "FRESH_GATE_00", "family": "NARROW_GATE", "bounds": [[-0.5, 8.5], [-2.5, 2.5], [0.2, 2.8]],
     "start": [0,0,1], "goal": [8,0,1], "obstacles": [box("O0",[4,-1.35,1.1],[1.2,2.0,2.2]), box("O1",[4,1.35,1.1],[1.2,2.0,2.2])],
     "corridors": [corridor("C0",[-0.2,-0.5,0.7],[2.8,0.5,1.4]), corridor("C1",[2.5,-0.25,0.7],[5.5,0.25,1.5]), corridor("C2",[5.2,-0.5,0.7],[8.2,0.5,1.4])]},
    {"scene_id": "FRESH_DETOUR_00", "family": "WIDE_LATERAL_DETOUR", "bounds": [[-0.5, 8.5], [-3.2, 3.2], [0.2, 2.8]],
     "start": [0,0,1], "goal": [8,0,1], "obstacles": [box("O0",[4,0,1.15],[1.6,3.4,2.3])],
     "corridors": [corridor("C0",[-0.2,-0.5,0.7],[2.5,0.5,1.4]), corridor("C1",[2.1,0.2,0.7],[3.0,2.4,1.5]), corridor("C2",[2.1,1.85,0.7],[5.9,2.65,1.5]), corridor("C3",[5.0,0.2,0.7],[6.0,2.4,1.5]), corridor("C4",[5.5,-0.5,0.7],[8.2,0.5,1.4])]},
    {"scene_id": "FRESH_TWIN_GATE_00", "family": "OFFSET_TWIN_GATE", "bounds": [[-0.5, 10.5], [-2.5, 2.5], [0.2, 2.8]],
     "start": [0,-0.3,1], "goal": [10,0.3,1.1], "obstacles": [box("O0",[3.0,-1.5,1.1],[1.0,1.4,2.2]), box("O1",[3.0,0.9,1.1],[1.0,1.4,2.2]), box("O2",[7.0,-0.9,1.1],[1.0,1.4,2.2]), box("O3",[7.0,1.5,1.1],[1.0,1.4,2.2])],
     "corridors": [corridor("C0",[-0.2,-0.7,0.7],[2.2,0.3,1.4]), corridor("C1",[1.9,-0.65,0.7],[4.1,0.05,1.5]), corridor("C2",[3.8,-0.65,0.7],[6.2,0.65,1.5]), corridor("C3",[5.9,-0.05,0.7],[8.1,0.65,1.5]), corridor("C4",[7.8,-0.3,0.8],[10.2,0.7,1.6])]},
)


def scene_spec(definition) -> SceneSpec:
    bounds = {axis: np.asarray(definition["bounds"][i], dtype=float) for i, axis in enumerate("xyz")}
    obstacles = tuple({"id": x["id"], "center": np.asarray(x["center"], dtype=float), "size": np.asarray(x["size"], dtype=float)} for x in definition["obstacles"])
    corridors = tuple({"id": x["id"], "min": np.asarray(x["min"], dtype=float), "max": np.asarray(x["max"], dtype=float)} for x in definition["corridors"])
    return SceneSpec(definition["scene_id"], definition["family"], bounds, np.asarray(definition["start"], dtype=float),
                     np.asarray(definition["goal"], dtype=float), obstacles, corridors, 0.08, 48, 240, 240, 0.801, definition)


def yaml_text(scene: SceneSpec, start: np.ndarray, goal: np.ndarray) -> str:
    lines = [f"contract_id: S7_{scene.scene_id}_V1", f"scene_id: {scene.scene_id}", f"family: {scene.family}", "world_bounds:"]
    for axis in "xyz": lines.append(f"  {axis}: [{scene.world_bounds[axis][0]}, {scene.world_bounds[axis][1]}]")
    lines += [f"start:\n  position: {start.tolist()}\n  velocity: [0, 0, 0]\n  acceleration: [0, 0, 0]",
              f"goal:\n  position: {goal.tolist()}\n  velocity: [0, 0, 0]\n  acceleration: [0, 0, 0]", "obstacles:"]
    for obstacle in scene.obstacles:
        lines += [f"  - id: {obstacle['id']}", f"    center: {obstacle['center'].tolist()}", f"    size: {obstacle['size'].tolist()}"]
    lines += ["vehicle_safety_radius: 0.08", "physics_hz: 240", "control_hz: 48", "reference_hz: 240",
              "max_reference_speed: 0.801", "max_reference_acceleration: 1.501", "max_reference_body_rate: 2.501", "yaw_reference: 0.0", "corridors:"]
    for corridor_item in scene.corridors:
        lines += [f"  - id: {corridor_item['id']}", f"    min: {corridor_item['min'].tolist()}", f"    max: {corridor_item['max'].tolist()}"]
    return "\n".join(lines) + "\n"


def wsl_path(path: Path) -> str:
    path = path.resolve(); return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def run_planner(scene: SceneSpec, task: TaskSpec, task_root: Path):
    task_root.mkdir(parents=True, exist_ok=True); yaml_path = task_root / "task.yaml"
    yaml_path.write_text(yaml_text(scene, task.start, task.goal), encoding="utf-8")
    output = task_root / "planner_output"; output.mkdir(exist_ok=True)
    command = " ".join(shlex.quote(x) for x in (wsl_path(PLANNER), wsl_path(yaml_path), wsl_path(output), wsl_path(clean_reproduction_root())))
    completed = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], capture_output=True, timeout=120)
    if completed.returncode: raise RuntimeError(completed.stderr.decode(errors="replace")[-500:])
    summary = json.loads((output / "planner_summary.json").read_text(encoding="utf-8"))
    if not all(summary.get(k) for k in ("setup_success", "optimize_success", "reference_pass")):
        raise RuntimeError(f"planner gate failed: {summary}")
    return output, summary


def sample_task(scene: SceneSpec, scene_index: int, candidate: int) -> TaskSpec:
    seed = GENERATION_SEED + scene_index * 100_000 + candidate; rng = np.random.default_rng(seed)
    def sample(c):
        low, high = np.asarray(c["min"]), np.asarray(c["max"]); return rng.uniform(low + .25*(high-low), high - .25*(high-low))
    return TaskSpec(scene.scene_id, candidate, f"{scene.scene_id}_TASK_{candidate:03d}", seed,
                    sample(scene.corridors[0]), sample(scene.corridors[-1]))


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    if OUT.exists() and (OUT / "frozen_manifest.json").exists(): raise RuntimeError("fresh holdout already frozen")
    if not PLANNER.exists(): raise FileNotFoundError(PLANNER)
    OUT.mkdir(parents=True, exist_ok=True); rows = []; evidence = []
    for scene_index, definition in enumerate(TOPOLOGIES):
        scene = scene_spec(definition); (OUT / "scenes").mkdir(exist_ok=True)
        (OUT / "scenes" / f"{scene.scene_id}.yaml").write_text(yaml_text(scene, scene.start, scene.goal), encoding="utf-8")
        accepted = 0
        for candidate in range(MAX_CANDIDATES):
            if accepted == TASKS_PER_TOPOLOGY: break
            task = sample_task(scene, scene_index, candidate); record = {"scene_id": scene.scene_id, "family": scene.family,
                "task_id": task.task_id, "candidate_id": candidate, "candidate_seed": task.candidate_seed,
                **{f"start_{a}": task.start[i] for i,a in enumerate("xyz")}, **{f"goal_{a}": task.goal[i] for i,a in enumerate("xyz")}}
            try:
                output, planner_summary = run_planner(scene, task, WORK / scene.scene_id / task.task_id)
                record.update({"feasible": True, "planner_reference_pass": True,
                               "trajectory_duration_s": planner_summary.get("final_duration", ""), "failure": ""})
                try:
                    trajectory = GcopterTrajectory.from_csv(output / "reference_coefficients.csv")
                    rollout, _ = rollout_task(scene, task, trajectory, CONTRACT)
                    playback = bool(rollout["goal_reached"] and rollout["collision_count"] == 0 and
                                    rollout["ground_contact_count"] == 0 and rollout["nonfinite_count"] == 0 and
                                    rollout["clip_count"] == 0)
                    record.update({"reference_playback_pass": playback,
                                   "reference_playback_goal_reached": bool(rollout["goal_reached"]),
                                   "reference_playback_collision_count": int(rollout["collision_count"]),
                                   "reference_playback_ground_count": int(rollout["ground_contact_count"])})
                except Exception as playback_error:
                    record.update({"reference_playback_pass": False, "reference_playback_goal_reached": False,
                                   "reference_playback_collision_count": "", "reference_playback_ground_count": "",
                                   "reference_playback_error": f"{type(playback_error).__name__}:{playback_error}"[:500]})
            except Exception as exc:
                record.update({"feasible": False, "planner_reference_pass": False, "reference_playback_pass": False,
                               "trajectory_duration_s": "", "failure": f"{type(exc).__name__}:{exc}"[:500]})
            evidence.append(record.copy())
            if record["feasible"]: rows.append(record); accepted += 1
            print(f"FRESH {scene.scene_id} candidate={candidate} accepted={accepted}/{TASKS_PER_TOPOLOGY} feasible={record['feasible']}", flush=True)
        if accepted != TASKS_PER_TOPOLOGY: raise RuntimeError(f"fresh feasibility yield failed for {scene.scene_id}: {accepted}")
    fields = list(dict.fromkeys(k for r in evidence for k in r))
    for name, values in (("task_manifest.csv", rows), ("feasibility_evidence.csv", evidence)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    topology_payload = list(TOPOLOGIES); topology_hash = canonical_hash(topology_payload)
    task_payload = [{k: r[k] for k in ("scene_id","family","task_id","candidate_seed","start_x","start_y","start_z","goal_x","goal_y","goal_z")} for r in rows]
    frozen = {"task": "S7_FRESH_TOPOLOGY_HOLDOUT", "generation_seed": GENERATION_SEED,
              "topology_count": len(TOPOLOGIES), "task_count": len(rows), "tasks_per_topology": TASKS_PER_TOPOLOGY,
              "topology_hash": topology_hash, "task_manifest_hash": canonical_hash(task_payload),
              "families": [x["family"] for x in TOPOLOGIES], "all_gcopter_feasible": all(r["feasible"] for r in rows),
              "created_before_policy_evaluation": True, "used_for_training_or_selection": False,
              "policy_evaluation_started": False, "topologies": topology_payload}
    (OUT / "frozen_manifest.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2), flush=True)


if __name__ == "__main__": main()
