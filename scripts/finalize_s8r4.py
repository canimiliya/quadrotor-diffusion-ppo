"""Freeze S8-R4 metrics after all eager-exact VAL rollouts are complete."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
from scripts.run_s8r4_unet_audit import OUT, CHECKPOINT_PASSES, FAMILIES, TASK, START_HEAD, TRAIN_SHA, VAL_SHA, NORMALIZATION_SHA, TRAINING_SEED, MAX_PASSES, sha256_file, canonical_sha
from scripts.run_s8r3_training_sufficiency import summarize_rows

def read_rows(path):
    rows=[]
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("success","collision","ground","unsafe","timeout","nonfinite"): r[k]=r[k].lower()=="true"
            for k in ("return","wall_time_s"): r[k]=float(r[k])
            for k in ("steps","action_clip_count"): r[k]=int(float(r[k]))
            rows.append(r)
    return rows

def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    curve=[r for r in csv.DictReader((OUT/"training_curve.csv").open(encoding="utf-8",newline="")) if r.get("model") != "DIFFUSION"]
    mlp_curve=list(csv.DictReader((ROOT/"artifacts/s8r3/training_curve.csv").open(encoding="utf-8",newline="")))
    curve += [r for r in mlp_curve if r["model"]=="DIFFUSION"]
    write_csv(OUT/"training_curve.csv", curve)
    for r in curve: r["effective_pass"]=float(r["effective_pass"]); r["val_offline_loss"]=float(r["val_offline_loss"])
    def nearest(model):
        rs=[r for r in curve if r["model"]==model]; return {p:min(rs,key=lambda r:abs(r["effective_pass"]-p)) for p in CHECKPOINT_PASSES}
    ur=nearest("UNET")
    selection=json.loads((OUT/"checkpoint_selection.json").read_text(encoding="utf-8"))
    if not selection.get("frozen_before_closed_loop") or selection.get("selected_by_closed_loop_success"): raise RuntimeError("BLOCKED_S8R4_SELECTION_FREEZE")
    metrics=[]; family=[]; val_wall_time_s=0.0; unet_by_pass={"0":{"status":"INITIAL_UNTRAINED","count":0,"success_rate":None}}
    for p in CHECKPOINT_PASSES:
        rows=read_rows(OUT/f"unet_pass_{p:02d}_val_rollouts.csv")
        if len(rows)!=1000: raise RuntimeError(f"BLOCKED_S8R4_VAL_COUNT:{p}:{len(rows)}")
        val_wall_time_s += sum(float(r.get("wall_time_s", 0.0)) for r in rows)
        result=summarize_rows(rows); result.update(model="UNET",pass_=p)
        result["pass"]=p; result.pop("pass_",None); unet_by_pass[str(p)]=result; metrics.append(result)
        for fam,v in result["by_family"].items(): family.append({"model":"UNET","pass":p,"family":fam,**v})
    # MLP/BC are references only; their S8-R3 VAL evidence is copied, not rerun.
    s3=json.loads((ROOT/"artifacts/s8r3/summary.json").read_text(encoding="utf-8")); ref=s3["closed_loop"]
    for kind in ("DIFFUSION","BC"):
        for p in CHECKPOINT_PASSES:
            r=dict(ref[kind][str(p)]); r["model"]=kind; r["pass"]=p; metrics.append(r)
            for fam,v in r.get("by_family",{}).items(): family.append({"model":kind,"pass":p,"family":fam,**v})
    write_csv(OUT/"closed_loop_metrics.csv",metrics); write_csv(OUT/"family_metrics.csv",family)
    uvals={p:float(unet_by_pass[str(p)]["success_rate"]) for p in CHECKPOINT_PASSES}; mvals={p:float(ref["DIFFUSION"][str(p)]["success_rate"]) for p in CHECKPOINT_PASSES}
    us=int(selection["unet"]["selected_pass"]); ms=int(selection["mlp"]["selected_pass"])
    late_u=float(np.mean([uvals[p] for p in (10,20,30)])); late_m=float(np.mean([mvals[p] for p in (10,20,30)]))
    peak_u=max(uvals.values()); peak_m=max(mvals.values()); final_u=uvals[30]; final_m=mvals[30]
    delta_selected=uvals[us]-mvals[ms]; late_delta=late_u-late_m; drop_u=peak_u-final_u; drop_m=peak_m-final_m
    promising=(delta_selected>=.03) or (late_delta>=.03 and drop_u<drop_m)
    label="PASS_S8R4_UNET_PROMISING" if promising else "PASS_S8R4_UNET_NO_CLEAR_ADVANTAGE"
    fam_focus=[r for r in family if r["pass"] in (1,us,30) and r["model"] in ("UNET","DIFFUSION")]
    bcvals={p:float(ref["BC"][str(p)]["success_rate"]) for p in CHECKPOINT_PASSES}
    prior_summary=json.loads((OUT/"summary.json").read_text())
    train_wall_time_s=float(prior_summary.get("train_wall_time_s", prior_summary.get("unet", {}).get("training_wall_time_s", 0.0)))
    summary={"task":TASK,"final_label":label,"start_head":START_HEAD,"dataset":{"train_sha256":TRAIN_SHA,"val_sha256":VAL_SHA,"test_accessed":False},"normalization_sha256":NORMALIZATION_SHA,"training_seed":TRAINING_SEED,"effective_passes":MAX_PASSES,"updates_per_pass":9329,"mlp_parameter_count":621360,"bc_parameter_count":592688,"unet_parameter_count":json.loads((OUT/"architecture_summary.json").read_text())["parameter_count"],"unet_config_sha256":sha256_file(OUT/"unet_config.json"),"unet":uvals,"mlp_reference":mvals,"bc_reference":bcvals,"mlp_offline_selected_pass":ms,"unet_offline_selected_pass":us,"offline_selected_delta":delta_selected,"late_mean_unet":late_u,"late_mean_mlp":late_m,"late_mean_delta":late_delta,"mlp_peak_to_final_drop":drop_m,"unet_peak_to_final_drop":drop_u,"family_results":fam_focus,"test_accessed":False,"test_run_count":0,"ppo_run_count":0,"train_wall_time_s":train_wall_time_s,"val_wall_time_s":val_wall_time_s,"formal_progress":"95%","unique_next_task":"NONE — WAIT_FOR_CONTROLLER_REVIEW","architecture_capacity_audit":True,"what_was_proven":"Only whether the frozen single-seed temporal U-Net backbone changes the current implementation under the frozen 10K contract.","what_was_not_proven":"Three-seed superiority, parameter-matched superiority, or Diffusion superiority over BC."}
    write_json(OUT/"summary.json",summary)
    print(json.dumps({"final_label":label,"unet":uvals,"selected_delta":delta_selected,"late_delta":late_delta,"drop_u":drop_u,"drop_m":drop_m},ensure_ascii=False,indent=2))

def write_json(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
if __name__=="__main__": main()
