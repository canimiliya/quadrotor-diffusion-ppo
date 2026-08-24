# Local cleanup disposition

Updated on 2026-08-24 after explicit owner approval. No virtual environment,
dataset, sealed TEST asset, canonical repository, or selected checkpoint was
deleted or moved.

## CLEANED_AFTER_APPROVAL

- duplicate reproducibility fresh clones and the disposable GCOPTER probe clone;
- four S8-R4 Git worktrees, removed with `git worktree remove` while preserving
  their branch refs and commits in the canonical repository;
- historical checkpoints except `checkpoints/s8r4/unet/pass_10.pt`;
- `artifacts/s8r2_10k/_cache/`, `.pytest_cache/`, `.deps/s2_probe/`, temporary
  logs, duplicate validation rollout CSVs, RViz helper outputs/scripts, and
  local orchestration notes after their relevant conclusions were consolidated.

These filesystem items were sent to the Windows Recycle Bin where applicable.
They are not part of the active project tree, but physical disk space is not
fully reclaimed until the owner empties the Recycle Bin. Do not empty the
whole Recycle Bin automatically because it may contain unrelated files.

## RETAIN

- all datasets listed in `docs/DATA_AND_CHECKPOINT_MANIFEST.md`, including S2,
  S3-R2 recovery, S7 fresh holdout, and S8-R2 10K splits;
- `artifacts/s8r3/train_obs_normalization.npz`;
- `checkpoints/s8r4/unet/pass_10.pt` with SHA256
  `8ab962ea18e818d9fe6ef481f4f0f3162b39a30ba75fa87c3aacdeb27f66545d`;
- `.deps/gcopter_reference/` and required third-party inputs;
- the active Python/Conda environments;
- Git-tracked stage reports, manifests, summaries, and small scientific
  evidence files;
- S7 failed/preflight holdout directories, because they are inside the retained
  dataset boundary and their unique data purpose has not been disproved.

## FUTURE_REVIEW_ONLY

- generated `__pycache__/` directories may be cleaned after Python processes
  close;
- the retained S7 failed/preflight data may be reconsidered only after a
  separate file-level manifest proves which content is reproducible and not
  needed for the frozen holdout evidence;
- no environment, dataset, or selected checkpoint may be deleted without a new
  explicit owner decision.
