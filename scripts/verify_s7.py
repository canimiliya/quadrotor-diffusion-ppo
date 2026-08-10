"""Independent read-only verifier for the completed S7 evidence package."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scripts.generate_s7_fresh_holdout import canonical_hash

OUT=ROOT/"artifacts"/"s7"; SEEDS=(20260812,20260813,20260814)


def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))


def auc(path):
    r=rows(path); x=np.array([float(v["env_steps"]) for v in r]); y=np.array([float(v["success_rate"]) for v in r]); return float(np.trapz(y,x)/(x[-1]-x[0]))


def main():
    summary=json.loads((OUT/"summary.json").read_text()); frozen=json.loads((OUT/"fresh_holdout"/"frozen_manifest.json").read_text()); fresh=json.loads((OUT/"fresh_holdout"/"evaluation_summary.json").read_text())
    task_rows=rows(OUT/"fresh_holdout"/"task_manifest.csv")
    task_payload=[{k:r[k] if k in ("scene_id","family","task_id") else int(r[k]) if k=="candidate_seed" else float(r[k]) for k in ("scene_id","family","task_id","candidate_seed","start_x","start_y","start_z","goal_x","goal_y","goal_z")} for r in task_rows]
    raw=rows(OUT/"fresh_holdout"/"policy_rollouts.csv"); methods=sorted({r["method"] for r in raw})
    recomputed={m:{"success":sum(int(r["success"]) for r in raw if r["method"]==m),"unsafe":sum(int(r["unsafe"]) for r in raw if r["method"]==m),"tasks":sum(r["method"]==m for r in raw)} for m in methods}
    pure=[]; bc=[]; diff=[]
    for seed in SEEDS:
        pure.append(auc(OUT.parent/"s6"/"runs"/f"pure_ppo_seed_{seed}"/"curve.csv")); bc.append(auc(OUT/"runs"/f"bc_ppo_seed_{seed}"/"curve.csv")); diff.append(auc(OUT.parent/"s6"/"runs"/f"diffusion_ppo_seed_{seed}"/"curve.csv"))
    checks={
        "formal_runs_complete":all(json.loads((OUT/"runs"/f"bc_ppo_seed_{s}"/"training_manifest.json").read_text())["actual_env_steps"]==500000 for s in SEEDS),
        "eleven_val_points":all(len(rows(OUT/"runs"/f"bc_ppo_seed_{s}"/"curve.csv"))==11 for s in SEEDS),
        "bc_checkpoint_identity":hashlib.sha256((ROOT/"checkpoints"/"s7"/"bc"/"best.pt").read_bytes()).hexdigest()==json.loads((OUT/"bc_training_summary.json").read_text())["best_checkpoint_sha256"],
        "topology_hash":canonical_hash(frozen["topologies"])==frozen["topology_hash"],
        "task_hash":canonical_hash(task_payload)==frozen["task_manifest_hash"],
        "fresh_counts":len(task_rows)==54 and len({r["task_id"] for r in task_rows})==54 and len({r["family"] for r in task_rows})==6,
        "fresh_raw_complete":len(raw)==648 and len(methods)==12 and all(v["tasks"]==54 for v in recomputed.values()),
        "fresh_raw_hash":hashlib.sha256((OUT/"fresh_holdout"/"policy_rollouts.csv").read_bytes()).hexdigest()==fresh["raw_rollouts_sha256"],
        "fresh_summary_recomputed":all(recomputed[m]["success"]==fresh["summaries"][m]["success"] and recomputed[m]["unsafe"]==fresh["summaries"][m]["unsafe"] for m in methods),
        "paired_auc_recomputed":np.allclose(np.array(diff)-np.array(bc),summary["current_distribution"]["paired_diffusion_minus_bc"],rtol=0,atol=1e-12) and np.allclose(np.array(bc)-np.array(pure),summary["current_distribution"]["paired_bc_minus_pure"],rtol=0,atol=1e-12),
        "classification_recomputed":all((np.array(diff)-np.array(bc))>0) and float(np.mean(np.array(diff)-np.array(bc)))<.05 and all((np.array(bc)-np.array(pure))>0),
        "fresh_claim_recomputed":recomputed["diffusion_only"]["success"]==10 and recomputed["bc_only"]["success"]==4 and all(recomputed[f"pure_ppo_seed_{s}"]["success"]==0 for s in SEEDS),
        "zero_residual_reuse_audited":len(fresh["equivalence_reuse"])==4 and all(x["audit"]["exact_zero"] for x in fresh["equivalence_reuse"].values()),
        "leakage_contract":not summary["test_leakage"]["s2_test_used_in_s7_training"] and not summary["test_leakage"]["s2_test_used_in_s7_selection"] and not summary["test_leakage"]["fresh_used_in_training_or_selection"],
        "safety_conservative":summary["classification"]["safety"]=="NOT_SUPPORTED",
    }
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"recomputed":{"pure_auc":pure,"bc_auc":bc,"diffusion_auc":diff,"fresh":recomputed}}
    (OUT/"independent_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result["status"]=="PASS" else 1)


if __name__=="__main__":main()
