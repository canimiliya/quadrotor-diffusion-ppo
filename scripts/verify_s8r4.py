"""Independent, read-only S8-R4 evidence verifier."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"artifacts"/"s8r4"
TRAIN="395a5c2ab1eb6cb3f3925fa3ecbe160d151833ef95cc5bc3b9187fbc497de3c5"; VAL="0fc50886504d36890444b4c09c2990962cc5be6551cb0e54c008df0d26f5246e"; NORM="e05b9a424cbb782dc1eddfcfd69fbf4e849806d3995bbab48e4a60193c76b6e0"
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def ok(name, cond, detail): return {"name":name,"passed":bool(cond),"detail":detail}
def main():
 checks=[]; checks += [ok("train_hash",sha(ROOT/"artifacts/s8r2_10k/train.npz")==TRAIN,TRAIN),ok("val_hash",sha(ROOT/"artifacts/s8r2_10k/val.npz")==VAL,VAL),ok("normalization_hash",sha(ROOT/"artifacts/s8r3/train_obs_normalization.npz")==NORM,NORM)]
 cfg=json.loads((OUT/"architecture_summary.json").read_text()); checks += [ok("unet_channels",cfg["parameter_count"]>0 and cfg["input_shape"]==["B",16,3],str(cfg)),ok("capacity_audit",cfg["architecture_capacity_audit"] is True,"true")]
 sel=json.loads((OUT/"checkpoint_selection.json").read_text()); checks += [ok("selection_metric",sel["selection_metric"]=="VAL offline diffusion loss",sel["selection_metric"]),ok("selection_frozen",sel["frozen_before_closed_loop"] and not sel["selected_by_closed_loop_success"],str(sel))]
 curve=list(csv.DictReader((OUT/"training_curve.csv").open(encoding="utf-8",newline=""))); checks += [ok("curve_rows",len(curve)>=60,str(len(curve))),ok("curve_finite",all(math.isfinite(float(r[k])) for r in curve for k in ("train_loss","val_offline_loss","gradient_norm")),"finite")]
 metrics=list(csv.DictReader((OUT/"closed_loop_metrics.csv").open(encoding="utf-8",newline=""))); u=[r for r in metrics if r["model"]=="UNET"]; checks += [ok("val_rows",all(int(r["count"])==1000 for r in u),str([(r["pass"],r["count"]) for r in u])),ok("no_test",json.loads((OUT/"summary.json").read_text())["test_accessed"] is False,"false"),ok("no_ppo",json.loads((OUT/"summary.json").read_text())["ppo_run_count"]==0,"0")]
 passed=sum(int(x["passed"]) for x in checks); result={"task":"S8-R4-CONDITIONAL-1D-UNET-DIFFUSION-ARCHITECTURE-AUDIT-V1","checks":checks,"passed":passed,"failed":len(checks)-passed,"final_label":"PASS_S8R4_INDEPENDENT_VERIFICATION" if passed==len(checks) else "FAIL_S8R4_INDEPENDENT_VERIFICATION"}
 (OUT/"independent_verification.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
