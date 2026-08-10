"""Remove only verified formal S7 intermediate checkpoints after evidence freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; CHECKPOINTS=ROOT/"checkpoints"/"s7"; OUT=ROOT/"artifacts"/"s7"


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification=json.loads((OUT/"independent_verification.json").read_text())
    if verification["status"]!="PASS":raise RuntimeError("S7 verification has not passed")
    removed=[]; retained=[]
    for run in sorted(CHECKPOINTS.glob("bc_ppo_seed_*")):
        if not (run/"best_model.zip").exists() or not (run/"last_model.zip").exists():raise RuntimeError(f"missing retained model in {run}")
        manifest=json.loads((OUT/"runs"/run.name/"training_manifest.json").read_text())
        expected={Path(x["path"]).name:x["sha256"] for x in manifest["checkpoint_schedule"]}
        for path in sorted(run.glob("checkpoint_*.zip")):
            if path.name not in expected or digest(path)!=expected[path.name]:raise RuntimeError(f"identity mismatch {path}")
            removed.append({"path":str(path),"bytes":path.stat().st_size,"sha256":expected[path.name]}); path.unlink()
        for name in ("best_model.zip","last_model.zip"):
            path=run/name; retained.append({"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path)})
    for path in sorted((CHECKPOINTS/"bc").glob("*.pt")):
        retained.append({"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path)})
    failed=list((CHECKPOINTS/"bc_ppo_seed_20260812"/"failed_attempt_01_timeout_100k").glob("*.zip"))
    for path in failed:retained.append({"path":str(path),"bytes":path.stat().st_size,"sha256":digest(path),"reason":"preserved engineering failure"})
    result={"status":"PASS","removed_count":len(removed),"removed_bytes":sum(x["bytes"] for x in removed),
        "removed":removed,"retained":retained,"recovery":"formal intermediate weights removed locally; hashes, best/last models, curves, and raw evidence retained; failed attempt preserved"}
    (OUT/"storage_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({k:result[k] for k in ("status","removed_count","removed_bytes")},indent=2))


if __name__=="__main__":main()
