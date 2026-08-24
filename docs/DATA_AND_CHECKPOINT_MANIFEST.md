# Data and checkpoint manifest

Captured on 2026-08-24 and updated after owner-approved local cleanup. No
formal TEST or training was performed. Dataset paths were not moved. Historical
weights marked `DELETED_LOCAL` below were removed only after their hashes and
scientific conclusions had been preserved in Git.

## Frozen datasets and recovery data

| item | path | size (bytes) | SHA256 | rebuildable | long-term action |
|---|---|---:|---|---|---|
| S2 TRAIN | `artifacts/s2/dataset/train.npz` | 22,630,109 | `701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21` | yes, from frozen S2 generation inputs and GCOPTER | KEEP |
| S2 VAL | `artifacts/s2/dataset/val.npz` | 4,883,986 | `c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc` | yes | KEEP |
| S2 TEST | `artifacts/s2/dataset/test.npz` | 4,807,917 | `88b39f83216e5e8985505ed77b9fbd89c01100237bdd8c09972fa29b90d70e66` | yes, but sealed evidence must be preserved | KEEP / SEALED |
| S3-R2 recovery TRAIN | `artifacts/s3r2/recovery_dataset/recovery_train.npz` | 11,830,269 | `ac87057d9363ee0557f8f02f2dcd2f1d4a27ae3bd956307800953eae2298f119` | yes, with recovery planner outputs | KEEP |
| S7 fresh-topology holdout | `artifacts/s7/fresh_holdout/` | 162,998,510 | frozen topology hash `504a08fcbac4ac5bb20a411212586b63f80d26a89724170f74bcb8d66153870d` | expensive historical holdout including feasibility evidence | KEEP |
| S8-R2 10K TRAIN | `artifacts/s8r2_10k/train.npz` | 818,927,261 | `395a5c2ab1eb6cb3f3925fa3ecbe160d151833ef95cc5bc3b9187fbc497de3c5` | expensive but rebuildable from frozen seeds, maps, GCOPTER, and acceptance code | KEEP |
| S8-R2 10K VAL | `artifacts/s8r2_10k/val.npz` | 102,578,989 | `0fc50886504d36890444b4c09c2990962cc5be6551cb0e54c008df0d26f5246e` | expensive but rebuildable | KEEP |
| S8-R2 10K TEST | `artifacts/s8r2_10k/test.npz` | 102,299,963 | `b4eee85496acb4d1e7b02e4f4bb44715c18c355014e77a79f8499a5fffa3a74e` | expensive but rebuildable; preserve sealed boundary | KEEP / SEALED |
| S8-R3 TRAIN normalization | `artifacts/s8r3/train_obs_normalization.npz` | 1,080 | `e05b9a424cbb782dc1eddfcfd69fbf4e849806d3995bbab48e4a60193c76b6e0` | yes, but required by the retained S8-R4 model contract | KEEP |

The former S8-R2 generated `_cache/` was approximately 1.08 GB and ignored by
Git. It was a rebuildable derivation aid, not a substitute for the three NPZ
files, and was removed after owner approval. All three NPZ splits remain.

## Checkpoints

The original hashes are retained as provenance even when the local file has
been deleted.

| route | path | size (bytes) | SHA256 | local status |
|---|---|---:|---|---|
| S3-R2 diffusion | `checkpoints/s3r2/best.pt` | 2,495,233 | `98de9a5d765ec1aeb48648497ac44ec98a6b6a071a607a01b4e49f3e13ab76cd` | DELETED_LOCAL |
| S8-R3 BC pass 01 | `checkpoints/s8r3/bc/pass_01.pt` | 2,381,531 | `c829d8d5af280a6aea6cb8d3dd950579173a82bbf4f78b102982c08435697e1f` | DELETED_LOCAL |
| S8-R3 BC pass 05 | `checkpoints/s8r3/bc/pass_05.pt` | 2,381,531 | `9fbf34b9d40d4b731a379d40ed7bd676c6a4817426dc4e26817959744904ec45` | DELETED_LOCAL |
| S8-R3 BC pass 10 | `checkpoints/s8r3/bc/pass_10.pt` | 2,381,531 | `da8d59a83199a16a9adb17926502669483d50bcab5d26903830ab7a534776bba` | DELETED_LOCAL |
| S8-R3 BC pass 20 | `checkpoints/s8r3/bc/pass_20.pt` | 2,381,531 | `521edeeb2ee7ed06ab87261af49b1e15622e18f9c6d3dc8cb88a2761d53ef0fe` | DELETED_LOCAL |
| S8-R3 BC pass 30 | `checkpoints/s8r3/bc/pass_30.pt` | 2,381,531 | `2b0b0db50d9afbba2f4a302551c825fd3088bb4500b023244a7d9d8b6f1e0e4d` | DELETED_LOCAL |
| S8-R3 Diffusion pass 01 | `checkpoints/s8r3/diffusion/pass_01.pt` | 2,496,411 | `f7093b82d1a18e0ccc6b86f8a8afba153a165988d433523d4f098e1dfb7b73e0` | DELETED_LOCAL |
| S8-R3 Diffusion pass 05 | `checkpoints/s8r3/diffusion/pass_05.pt` | 2,496,411 | `e2af0b3f5e7b3c7b7adbf20569aa9f786a7ef876240a66ed643fe26fe8f24758` | DELETED_LOCAL |
| S8-R3 Diffusion pass 10 | `checkpoints/s8r3/diffusion/pass_10.pt` | 2,496,411 | `acbfec8cf750cef2a9d4f315905ff057c3aa6d271e3cc384e9e37f4473e4e5d6` | DELETED_LOCAL |
| S8-R3 Diffusion pass 20 | `checkpoints/s8r3/diffusion/pass_20.pt` | 2,496,411 | `d7e4669f45cf3d66677915ec0d7f6276d431f5e9dbc0b9c326fd3f4ac4f25b11` | DELETED_LOCAL |
| S8-R3 Diffusion pass 30 | `checkpoints/s8r3/diffusion/pass_30.pt` | 2,496,411 | `2e91bcc260409ad6b5e5fddb876f6be6292327a2cbd50ab06105c865ccb012dc` | DELETED_LOCAL |
| S8-R4 U-Net pass 01 | `checkpoints/s8r4/unet/pass_01.pt` | 188,488,387 | `ade843faeb9acc3fea87309a6453402c7f5978af7af5986e5e28f7d0f26becb2` | DELETED_LOCAL |
| S8-R4 U-Net pass 05 | `checkpoints/s8r4/unet/pass_05.pt` | 188,488,387 | `bb00120907ec5ee2a94311e5a6bb17bf4356e4e6f8367b2f8541d83049d140d3` | DELETED_LOCAL |
| S8-R4 U-Net pass 10 | `checkpoints/s8r4/unet/pass_10.pt` | 188,488,387 | `8ab962ea18e818d9fe6ef481f4f0f3162b39a30ba75fa87c3aacdeb27f66545d` | **RETAINED** |
| S8-R4 U-Net pass 20 | `checkpoints/s8r4/unet/pass_20.pt` | 188,488,387 | `daa5faacfef96aad00414f9a980bafaaf5b3e9dd98ac3c5e3ea182c4bddf0b15` | DELETED_LOCAL |
| S8-R4 U-Net pass 30 | `checkpoints/s8r4/unet/pass_30.pt` | 188,488,387 | `bf4c7cbc01c7e8262722c0815d38f950cb9b32e44456f0ef0b1a453d7987fccf` | DELETED_LOCAL |

The owner explicitly approved retaining only the formally selected S8-R4
Temporal U-Net pass 10. Results, limitations, and future directions are
consolidated in `docs/RESEARCH_HISTORY_AND_NEXT_DIRECTIONS.md`.

## Recommended migration

Datasets currently remain classified by stage in their canonical relative
paths. This avoids breaking code or introducing a machine-specific data root.
If a future migration is required, copy first, verify every listed hash, update
an explicit data-root contract, and remove the original only after a second
independent hash audit.
