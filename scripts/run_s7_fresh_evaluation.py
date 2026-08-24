"""Evaluate all frozen S7 methods once on the immutable fresh holdout."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from quadrotor_diffusion_ppo.paths import configure_external_imports
configure_external_imports()

import numpy as np
import torch
from stable_baselines3 import PPO

from quadrotor_diffusion_ppo.evaluation.runtime import DiffusionInferenceScheduler, RuntimeMode
from quadrotor_diffusion_ppo.ppo.bc import FrozenBCPrior
from quadrotor_diffusion_ppo.ppo.env import PurePPONavigationEnv, TaskEndpoint
from quadrotor_diffusion_ppo.ppo.residual import FrozenDiffusionPrior, compose_residual_action, stable_prior_seed
from scripts.generate_s7_fresh_holdout import canonical_hash, scene_spec
from scripts.run_s6_core import S3R2_CHECKPOINT, preflight_fast, write_csv, write_json


OUT = ROOT / "artifacts" / "s7" / "fresh_holdout"
BC_CHECKPOINT = ROOT / "checkpoints" / "s7" / "bc" / "best.pt"
SEEDS = (20260812, 20260813, 20260814)


def load_frozen():
    frozen = json.loads((OUT / "frozen_manifest.json").read_text(encoding="utf-8"))
    if frozen["policy_evaluation_started"] or frozen["used_for_training_or_selection"]:
        raise RuntimeError("fresh manifest state is not pristine")
    if canonical_hash(frozen["topologies"]) != frozen["topology_hash"]: raise RuntimeError("topology hash mismatch")
    scenes = {d["scene_id"]: scene_spec(d) for d in frozen["topologies"]}
    tasks = []
    with (OUT / "task_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            tasks.append(TaskEndpoint(row["scene_id"], row["task_id"], int(row["candidate_seed"]),
                np.asarray([float(row[f"start_{a}"]) for a in "xyz"]), np.asarray([float(row[f"goal_{a}"]) for a in "xyz"]), "FRESH_HOLDOUT"))
    payload = [{k: row for k, row in zip(("scene_id","family","task_id","candidate_seed","start_x","start_y","start_z","goal_x","goal_y","goal_z"),
        (t.scene_id, scenes[t.scene_id].family, t.task_id, t.candidate_seed, *t.start.tolist(), *t.goal.tolist()))} for t in tasks]
    if canonical_hash(payload) != frozen["task_manifest_hash"] or len(tasks) != 54: raise RuntimeError("fresh task hash mismatch")
    return frozen, scenes, tasks


def summarize(rows):
    def one(values):
        n = max(1, len(values)); return {"tasks": len(values), "success": sum(r["success"] for r in values),
            "success_rate": sum(r["success"] for r in values)/n, "collision": sum(r["collision"] for r in values),
            "unsafe": sum(r["unsafe"] for r in values), "timeout": sum(r["timeout"] for r in values),
            "nonfinite": sum(r["nonfinite"] for r in values), "mean_return": float(np.mean([r["mean_return"] for r in values])) if values else 0.0}
    result = one(rows); result["by_family"] = {f: one([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})}; return result


def rollout_method(method, tasks, scenes, *, model=None, bc=None, diffusion=None, workers=16):
    policy_lock = threading.Lock(); bc_lock = threading.Lock(); scheduler = None
    if diffusion is not None: scheduler = DiffusionInferenceScheduler(diffusion, RuntimeMode.FAST, max_batch_size=workers)
    def run(task):
        env = PurePPONavigationEnv(scenes[task.scene_id], task, train_mode=False, gui=False)
        base_seed = stable_prior_seed(task.task_id); total = 0.0; info = {}; support = 0; projection = 0
        try:
            observation, _ = env.reset(seed=base_seed); terminated = truncated = False
            while not (terminated or truncated):
                if method == "greedy_goal":
                    direction = env.scene.goal - np.asarray(env._getDroneStateVector(0)[:3]); norm = np.linalg.norm(direction)
                    action = (direction / max(norm, 1e-8)).astype(np.float32)
                elif method.startswith("pure_ppo"):
                    with policy_lock: action, _ = model.predict(observation, deterministic=True)
                elif method == "bc_only":
                    with bc_lock: action = bc.predict(np.asarray(observation).reshape(1,34))[0]
                elif method.startswith("bc_ppo"):
                    with policy_lock: residual, _ = model.predict(observation, deterministic=True)
                    with bc_lock: prior = bc.predict(np.asarray(observation).reshape(1,34))
                    action, projected = compose_residual_action(prior, np.asarray(residual).reshape(1,3)); action=action[0]; projection += int(projected[0])
                elif method == "diffusion_only":
                    action = scheduler.predict(observation, base_seed + env.episode_steps)
                else: raise ValueError(method)
                action = np.asarray(action, dtype=np.float32).reshape(3); support += int(np.linalg.norm(action)>1+1e-6)
                observation, reward, terminated, truncated, info = env.step(action); total += reward
        finally: env.close()
        collision=bool(info.get("collision")); ground=bool(info.get("ground_contact")); nonfinite=bool(info.get("nonfinite"))
        return {"method": method, "task_id": task.task_id, "scene_id": task.scene_id, "family": scenes[task.scene_id].family,
            "success": int(bool(info.get("success"))), "collision": int(collision), "ground_contact": int(ground),
            "unsafe": int(collision or ground or nonfinite), "timeout": int(bool(info.get("timeout"))), "nonfinite": int(nonfinite),
            "mean_return": float(total), "episode_steps": int(info.get("episode_steps",0)), "action_clip_count": int(info.get("action_clip_count",0)),
            "policy_action_support_violation_count": support, "projection_count": projection}
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"s7-{method}") as executor: rows=list(executor.map(run,tasks))
    finally:
        if scheduler is not None: scheduler.close()
    return rows, summarize(rows)


def zero_residual_audit(path: Path):
    model = PPO.load(str(path), device="cuda"); probes=np.zeros((7,34),dtype=np.float32); actions,_=model.predict(probes,deterministic=True)
    maximum=float(np.abs(actions).max()); return model, {"checkpoint": str(path), "max_abs_deterministic_residual": maximum, "exact_zero": maximum==0.0}


def main():
    frozen, scenes, tasks = load_frozen(); bc=FrozenBCPrior(BC_CHECKPOINT,"cuda"); diffusion=FrozenDiffusionPrior(S3R2_CHECKPOINT,"cuda")
    diffusion_preflight=preflight_fast(diffusion); all_rows=[]; summaries={}; reuse={}
    def execute(name, **kwargs):
        rows, summary=rollout_method(name,tasks,scenes,**kwargs); all_rows.extend(rows); summaries[name]=summary
        print(f"FRESH method={name} success={summary['success']}/54 unsafe={summary['unsafe']} return={summary['mean_return']:.4f}",flush=True)
    execute("greedy_goal")
    for seed in SEEDS:
        model=PPO.load(str(ROOT/"checkpoints"/"s6"/f"pure_ppo_seed_{seed}"/"best_model.zip"),device="cuda")
        execute(f"pure_ppo_seed_{seed}",model=model)
    execute("bc_only",bc=bc)
    bc_only_rows=[r for r in all_rows if r["method"]=="bc_only"]
    for seed in SEEDS:
        path=ROOT/"checkpoints"/"s7"/f"bc_ppo_seed_{seed}"/"best_model.zip"; model,audit=zero_residual_audit(path)
        best=json.loads((ROOT/"artifacts"/"s7"/"runs"/f"bc_ppo_seed_{seed}"/"evaluation_manifest.json").read_text())["best_step"]
        if best==0 and audit["exact_zero"]:
            name=f"bc_ppo_seed_{seed}"; rows=[dict(r,method=name) for r in bc_only_rows]; all_rows.extend(rows); summaries[name]=summarize(rows)
            reuse[name]={"source":"bc_only","reason":"selected_step_0_exact_zero_residual","audit":audit}
        else: execute(f"bc_ppo_seed_{seed}",model=model,bc=bc)
    execute("diffusion_only",diffusion=diffusion)
    diffusion_rows=[r for r in all_rows if r["method"]=="diffusion_only"]
    for seed in SEEDS:
        path=ROOT/"checkpoints"/"s6"/f"diffusion_ppo_seed_{seed}"/"best_model.zip"; _,audit=zero_residual_audit(path)
        if not audit["exact_zero"]: raise RuntimeError("S6 selected diffusion residual is not exact zero")
        name=f"diffusion_ppo_seed_{seed}"; rows=[dict(r,method=name) for r in diffusion_rows]; all_rows.extend(rows); summaries[name]=summarize(rows)
        reuse[name]={"source":"diffusion_only","reason":"selected_step_0_exact_zero_residual","audit":audit}
    write_csv(OUT/"policy_rollouts.csv",all_rows)
    result={"task":"S7_FRESH_TOPOLOGY_HOLDOUT_EVALUATION","frozen_topology_hash":frozen["topology_hash"],
        "frozen_task_manifest_hash":frozen["task_manifest_hash"],"policy_evaluation_after_freeze_commit":True,
        "used_for_training_or_selection":False,"diffusion_runtime_preflight":diffusion_preflight,"summaries":summaries,
        "equivalence_reuse":reuse,"raw_rollouts_sha256":hashlib.sha256((OUT/"policy_rollouts.csv").read_bytes()).hexdigest()}
    write_json(OUT/"evaluation_summary.json",result); print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
