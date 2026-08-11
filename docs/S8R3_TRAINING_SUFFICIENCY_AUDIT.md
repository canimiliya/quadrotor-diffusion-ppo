# S8-R3 Training Sufficiency Audit

## 先给结论

这轮审计已经完成，数据、训练数值、闭环 VAL 和独立 verifier 都通过。它回答了“旧的固定 10k updates 比较是否可能把新 10K 数据训练得不够”这一问题，但结果不是支持“Diffusion 明显欠训”的证据：Diffusion 在 pass 1 已达到 82.4%，pass 30 为 82.5%，中间 pass 5 达到 95.5% 后又回落。相反，Matched BC 的 pass 1 到 pass 30 从 62.7% 提高到 84.2%，BC 自身存在明显训练不足/训练曲线不单调现象。

正式审计标签：`PASS_S8R3_TRAINING_SUFFICIENCY_AUDIT_COMPLETE`。

Diffusion 分类：`INCONCLUSIVE_WITHOUT_UNDERTRAINING_GATE`。它既不满足 under-training 的 +10 percentage-point gate，也不满足 pass10→20→30 最大差异小于 0.05 的饱和 gate。因此不能把本轮结果写成“Diffusion 欠训已证实”，也不能把它写成“已饱和”。这不是数据完整性 blocker，而是控制器需要知晓的结果边界。

## 任务卡 1–7 的直接回答

1. 新 10K TRAIN 有 **4,776,168 个 H=16 windows**；VAL 有 **598,153 个 windows**。窗口严格按每个 episode 的 `max(0, length-16+1)` 计算。
2. batch=512，因此 `ceil(4,776,168/512)=9,329 updates/effective pass`。旧 10k updates 只相当于约 **1.072 个新数据 pass**。
3. BC 的 VAL success：pass1 **62.7%**、pass5 **98.2%**、pass10 **73.2%**、pass20 **82.6%**、pass30 **84.2%**。pass1→30 提升 **21.5 pp**，所以 BC 自身明显不是“10k 已足够”的稳定基线。
4. ConditionalDiffusionMLP 的 VAL success：pass1 **82.4%**、pass5 **95.5%**、pass10 **92.0%**、pass20 **80.8%**、pass30 **82.5%**。pass1→30 只有 **0.1 pp**，且曲线非单调。
5. 对 Diffusion，`Success(pass30)-Success(pass1)=+0.1 pp`，不满足 +10 pp under-training gate；因此本轮**没有支持“此前 Diffusion 比较被欠训混淆”的证据**。BC 则满足该 gate，说明 BC 的 10k-equivalent exposure 明显不足。
6. pass30 时 Diffusion **82.5%**，BC **84.2%**，Diffusion 比 BC 低 **1.7 pp**。完整的 pass1/5/10/20/30 曲线都保留；不能只挑 pass5 的 95.5%。
7. 下一步**没有自动授权替换 Temporal U-Net**。本轮既没有证明 Diffusion 欠训，也没有证明 MLP 已饱和；是否做 U-Net 由 controller review 决定，本任务到此停止。

## 冻结合同与数据

- S8-R2 hashes：TRAIN `395a5c2ab1eb6cb3f3925fa3ecbe160d151833ef95cc5bc3b9187fbc497de3c5`；VAL `0fc50886504d36890444b4c09c2990962cc5be6551cb0e54c008df0d26f5246e`；TEST `b4eee85496acb4d1e7b02e4f4bb44715c18c355014e77a79f8499a5fffa3a74e`。
- TRAIN-only normalization SHA256：`e05b9a424cbb782dc1eddfcfd69fbf4e849806d3995bbab48e4a60193c76b6e0`。
- BC：34→128→128→256×4 residual→8，592,688 parameters；SmoothL1 + AdamW，lr 3e-4，weight decay 1e-4，batch 512，clip 1.0。
- Diffusion：现有 ConditionalDiffusionMLP，34/128/128 observation encoder，time64，48→256×4 residual→48，621,360 parameters；bounded x0、T=100、DDIM 10、eta=0、cosine schedule、MSE_x0。
- BC config SHA256：`09cf8c9a287d5a773ed765752b13ca5f7a5ae57fb558492f5c8492d60dd3f082`。
- Diffusion frozen config SHA256：`03cf747fce8c6af6be98804f8a0f2bf2a280269284718cb27a3c1070520ffba8`。
- TRAINING_SEED=`20260820`；Diffusion VAL 使用 `FAST_CUDA_GRAPH_EXACT_BATCH_ONE`；两者都为 H=16、单 34D observation、3D unit-ball action、receding-horizon first-action。

## 闭环 VAL 摘要

每个 checkpoint 均完整评估 VAL 1000 tasks，10 个 procedural families 各 100 tasks；family 级结果见 `family_metrics.csv`。

| model | pass1 | pass5 | pass10 | pass20 | pass30 |
|---|---:|---:|---:|---:|---:|
| Matched BC | 62.7% | 98.2% | 73.2% | 82.6% | 84.2% |
| MLP Diffusion | 82.4% | 95.5% | 92.0% | 80.8% | 82.5% |

所有 rollout 都 finite；TEST 未访问；PPO_RUN_COUNT=0；TEST_RUN_COUNT=0。VAL 总墙钟约 7,700 s（task-sharded 并行执行，统计语义不变）。训练墙钟约 BC 2,469.5 s、Diffusion 2,571.1 s。

## 证据与验证

- [training_curve.csv](../artifacts/s8r3/training_curve.csv)：60 个 0.5-pass 记录，包含 train loss、VAL offline loss、gradient norm、lr、wall time、GPU memory。
- [closed_loop_metrics.csv](../artifacts/s8r3/closed_loop_metrics.csv)：10 个 checkpoint 的整体 VAL 指标。
- [family_metrics.csv](../artifacts/s8r3/family_metrics.csv)：10 families×10 checkpoints 的分解。
- [window_statistics.json](../artifacts/s8r3/window_statistics.json)、[train_obs_normalization.npz](../artifacts/s8r3/train_obs_normalization.npz)、[diffusion_frozen_config.json](../artifacts/s8r3/diffusion_frozen_config.json)。
- [independent_verification.json](../artifacts/s8r3/independent_verification.json)：47/47 checks passed。
- 图：`figures/offline_loss_vs_effective_pass.png`、`figures/closed_loop_success_vs_effective_pass.png`、`figures/family_success_at_passes.png`。

本次曾有一条因终端等待器一小时上限中断的工程运行，已原样保留在 `checkpoints/s8r3_interrupted_20260811` 与 `artifacts/s8r3_interrupted_run.json`，不混入正式证据；正式结果来自后台重新完成的单条连续 30-pass 训练轨迹。

## Git 与阶段边界

- START_HEAD=`a8805cf62258a86948a3496267aaebbfad486425`。
- Branch=`agent/s8r3-training-sufficiency-audit-v1`。
- END_HEAD/REMOTE_HEAD=`ed2beb64ce72ea181cf20c37f05b2f7c51e05f3e`。
- TEST 保持 sealed；不提交 10K NPZ、cache 或模型 checkpoint。
- `FORMAL_PROGRESS = 95%`。
- `UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW`。

本任务完成后停止；不自行实施 Temporal U-Net、不训练 PPO、不打开 TEST、不改变 observation/H/action chunk 或 Diffusion runtime。
