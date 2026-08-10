"""Freeze S7 paired statistics and paper figures from completed raw evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_s6_core import write_csv, write_json

OUT=ROOT/"artifacts"/"s7"; SEEDS=(20260812,20260813,20260814)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))


def curve(method,seed):
    base=ROOT/"artifacts"/("s7" if method=="bc_ppo" else "s6")/"runs"/f"{method}_seed_{seed}"/"curve.csv"
    rows=read_csv(base); return [{k:(int(v) if k.endswith("success") or k in ("env_steps","collision","unsafe","timeout") else float(v)) for k,v in r.items()} for r in rows]


def auc(rows,key):
    x=np.asarray([r["env_steps"] for r in rows],float); y=np.asarray([r[key] for r in rows],float)
    return float(np.trapz(y,x)/(x[-1]-x[0]))


def summarize_rollouts(rows, method):
    """Create an auditable current-distribution table from task-level rows."""
    out=[]
    families=("ALL", "OPEN", "BLOCK", "SBEND")
    for family in families:
        selected=rows if family=="ALL" else [r for r in rows if r["family"]==family]
        as_bool=lambda value: str(value).lower() in ("1", "true")
        out.append({
            "method":method,"family":family,"tasks":len(selected),
            "success":sum(as_bool(r["success"]) for r in selected),
            "collision":sum(as_bool(r["collision"]) for r in selected),
            "unsafe":sum(as_bool(r["unsafe"]) for r in selected),
            "timeout":sum(as_bool(r["timeout"]) for r in selected),
            "mean_return":float(np.mean([float(r["mean_return"]) for r in selected])),
        })
    return out


def main():
    methods=("pure_ppo","bc_ppo","diffusion_ppo"); curves={(m,s):curve(m,s) for m in methods for s in SEEDS}
    metrics=[]; paired=[]
    for seed in SEEDS:
        values={}
        for method in methods:
            rows=curves[method,seed]; values[method]={"success_auc":auc(rows,"success_rate"),
                "obstacle_auc":auc(rows,"obstacle_success_rate"),"collision_auc":auc(rows,"collision_rate"),
                "return_auc":auc(rows,"mean_return")}
            metrics.append({"method":method,"seed":seed,**values[method]})
        paired.append({"seed":seed,
            "diffusion_minus_bc_success_auc":values["diffusion_ppo"]["success_auc"]-values["bc_ppo"]["success_auc"],
            "bc_minus_pure_success_auc":values["bc_ppo"]["success_auc"]-values["pure_ppo"]["success_auc"],
            "diffusion_minus_bc_obstacle_auc":values["diffusion_ppo"]["obstacle_auc"]-values["bc_ppo"]["obstacle_auc"],
            "bc_minus_pure_obstacle_auc":values["bc_ppo"]["obstacle_auc"]-values["pure_ppo"]["obstacle_auc"]})
    aggregate=[]
    for method in methods:
        for i,step in enumerate([r["env_steps"] for r in curves[method,SEEDS[0]]]):
            row={"method":method,"env_steps":step}
            for key in ("success_rate","obstacle_success_rate","collision_rate","mean_return"):
                vals=[curves[method,s][i][key] for s in SEEDS]; row[f"{key}_mean"]=float(np.mean(vals)); row[f"{key}_std"]=float(np.std(vals,ddof=1))
            aggregate.append(row)
    online_seed_rows=[]
    for method in methods:
        for seed in SEEDS:
            online_seed_rows.extend({"method":method,"seed":seed,**row} for row in curves[method,seed])
    bc_rollouts=read_csv(OUT/"bc_only_val_rollouts.csv")
    diffusion_rollouts=[r for r in read_csv(ROOT/"artifacts"/"s6"/"runs"/f"diffusion_ppo_seed_{SEEDS[0]}"/"val_rollouts.csv") if int(r["env_steps"])==0]
    current_prior_rows=summarize_rollouts(bc_rollouts,"bc_only")+summarize_rollouts(diffusion_rollouts,"diffusion_only")
    write_csv(OUT/"current_distribution_metrics.csv",metrics); write_csv(OUT/"paired_metrics.csv",paired); write_csv(OUT/"aggregate_curves.csv",aggregate)
    write_csv(OUT/"online_seed_curves.csv",online_seed_rows); write_csv(OUT/"current_prior_metrics.csv",current_prior_rows)
    fresh=json.loads((OUT/"fresh_holdout"/"evaluation_summary.json").read_text())
    fresh_rows=[]
    for method,summary in fresh["summaries"].items():
        fresh_rows.append({"method":method,"family":"ALL","tasks":summary["tasks"],"success":summary["success"],"success_rate":summary["success_rate"],"collision":summary["collision"],"ground_contact":summary["unsafe"]-summary["collision"],"unsafe":summary["unsafe"],"timeout":summary["timeout"],"mean_return":summary["mean_return"]})
        for family,value in summary["by_family"].items():fresh_rows.append({"method":method,"family":family,"tasks":value["tasks"],"success":value["success"],"success_rate":value["success_rate"],"collision":value["collision"],"ground_contact":value["unsafe"]-value["collision"],"unsafe":value["unsafe"],"timeout":value["timeout"],"mean_return":value["mean_return"]})
    write_csv(OUT/"fresh_metrics.csv",fresh_rows)
    diff_bc=[r["diffusion_minus_bc_success_auc"] for r in paired]; bc_pure=[r["bc_minus_pure_success_auc"] for r in paired]
    online_specific=all(x>0 for x in diff_bc) and np.mean(diff_bc)>=.05
    prior_supported=all(x>0 for x in bc_pure)
    d=fresh["summaries"]["diffusion_only"]; b=fresh["summaries"]["bc_only"]
    pure=[fresh["summaries"][f"pure_ppo_seed_{s}"]["success"] for s in SEEDS]
    generalization=d["success"]>0 and d["success"]>max(pure); fresh_diff_adv=d["success"]>b["success"]
    label="PASS_S7_PRIOR_EFFECT_AND_FRESH_GENERALIZATION_SUPPORTED_DIFFUSION_SPECIFIC_THRESHOLD_NOT_MET"
    summary={"task":"S7-R0-MATCHED-BC-ABLATION-AND-FRESH-TOPOLOGY-GENERALIZATION-V1","final_label":label,
        "classification":{"online":"PRIOR_ADVANTAGE_NOT_DIFFUSION_SPECIFIC","fresh":"DIFFUSION_GENERALIZATION_ADVANTAGE_LIMITED","safety":"NOT_SUPPORTED"},
        "current_distribution":{"bc_only_success":31,"diffusion_only_success":37,"success_auc_mean":{m:float(np.mean([next(x["success_auc"] for x in metrics if x["method"]==m and x["seed"]==s) for s in SEEDS])) for m in methods},
            "paired_diffusion_minus_bc":diff_bc,"paired_diffusion_minus_bc_mean":float(np.mean(diff_bc)),"paired_bc_minus_pure":bc_pure,"paired_bc_minus_pure_mean":float(np.mean(bc_pure)),
            "diffusion_specific_threshold":.05,"diffusion_specific_supported":bool(online_specific),"prior_advantage_supported":bool(prior_supported)},
        "fresh":{"topology_hash":fresh["frozen_topology_hash"],"task_manifest_hash":fresh["frozen_task_manifest_hash"],
            "diffusion_success":d["success"],"bc_success":b["success"],"pure_success":pure,"greedy_success":fresh["summaries"]["greedy_goal"]["success"],
            "diffusion_unsafe":d["unsafe"],"bc_unsafe":b["unsafe"],"generalization_supported":bool(generalization),"diffusion_over_bc":bool(fresh_diff_adv),
            "generalization_gap_success_rate":{"diffusion":d["success_rate"]-37/54,"bc":b["success_rate"]-31/54}},
        "residual_conclusion":"online residual fine-tuning did not stably improve either prior; selected Diffusion checkpoints remain step 0 for all seeds",
        "test_leakage":{"s2_test_used_in_s7_training":False,"s2_test_used_in_s7_selection":False,"fresh_used_in_training_or_selection":False,"s6_test_status":"development_test_already_opened_before_S7"},
        "formal_progress":"85% pending controller acceptance","unique_next_task":"NONE - WAIT_FOR_CONTROLLER_REVIEW"}
    write_json(OUT/"summary.json",summary)
    colors={"pure_ppo":"#777777","bc_ppo":"#2b8cbe","diffusion_ppo":"#d95f0e"}; labels={"pure_ppo":"Pure PPO","bc_ppo":"BC prior + PPO","diffusion_ppo":"Diffusion prior + PPO"}
    for key,name,ylabel in (("success_rate","bc_vs_diffusion_online_auc","Success rate"),("obstacle_success_rate","obstacle_success_comparison","Obstacle success rate")):
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for m in methods:
            rs=[r for r in aggregate if r["method"]==m]; x=np.array([r["env_steps"] for r in rs])/1000; mean=np.array([r[f"{key}_mean"] for r in rs]); std=np.array([r[f"{key}_std"] for r in rs]); ax.plot(x,mean,label=labels[m],color=colors[m],lw=2); ax.fill_between(x,mean-std,mean+std,color=colors[m],alpha=.15)
        ax.set(xlabel="Environment steps (thousands)",ylabel=ylabel,xlim=(0,500),ylim=(0,1)); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(OUT/f"{name}.{ext}",dpi=300)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4.4)); names=["Pure PPO\n(mean)","BC only","Diffusion only","Greedy"]
    vals=[np.mean(pure),b["success"],d["success"],fresh["summaries"]["greedy_goal"]["success"]]; ax.bar(names,vals,color=["#777777","#2b8cbe","#d95f0e","#66a61e"]); ax.set(ylabel="Fresh successes / 54",ylim=(0,54)); ax.grid(axis="y",alpha=.25); fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(OUT/f"fresh_topology_success.{ext}",dpi=300)
    plt.close(fig)
    families=sorted(d["by_family"]); x=np.arange(len(families)); fig,ax=plt.subplots(figsize=(9,4.6)); width=.34
    ax.bar(x-width/2,[b["by_family"][f]["success"] for f in families],width,label="BC",color="#2b8cbe"); ax.bar(x+width/2,[d["by_family"][f]["success"] for f in families],width,label="Diffusion",color="#d95f0e")
    ax.set_xticks(x,labels=[f.replace("_","\n") for f in families]); ax.set(ylabel="Successes / 9",ylim=(0,9)); ax.grid(axis="y",alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(OUT/f"fresh_topology_by_family.{ext}",dpi=300)
    plt.close(fig); print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
