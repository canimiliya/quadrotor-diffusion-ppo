# Safe to delete after review

This file is a recommendation only. No item below was deleted by this task.

## SAFE_TO_DELETE

- `__pycache__/` directories under the repository, after closing Python
  processes. They are interpreter-generated and recreated automatically.
- `.pytest_cache/`, after confirming no test run is currently being diagnosed.

## LIKELY_SAFE

- `artifacts/s8r2_10k/_cache/` (approximately 1.08 GB): derived planner cache;
  retain until an independent S8-R2 rebuild has been accepted.
- `artifacts/s7/fresh_holdout/*/` directories named as failed or preflight
  work: historical diagnostic evidence; inspect before removal.
- `logs/` and ignored S8-R4 console logs: potentially reproducible runtime
  output, but preserve while the S8-R4 audit remains under review.
- `.deps/s2_probe/`: probe output only if its provenance is no longer needed.

## DO_NOT_DELETE

- `artifacts/s2/dataset/`, `artifacts/s3r2/recovery_dataset/`, and
  `artifacts/s8r2_10k/{train,val,test}.npz`;
- all `checkpoints/` required by the S3-R2, S8-R3 and S8-R4 evidence chains;
- `.deps/gcopter_reference/` and the external GCOPTER scene checkout until a
  fresh bootstrap has been independently verified;
- the active `smd-blackwell` environment and any environment not yet classified
  by its owner;
- interrupted/failed run artifacts, sealed TEST data, and historical logs.

Deletion requires controller confirmation, a hash/archive decision, and a
post-delete Git/status audit.
