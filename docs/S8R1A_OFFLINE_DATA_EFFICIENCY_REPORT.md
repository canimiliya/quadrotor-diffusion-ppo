# S8-R1A 离线专家数据量消融报告

## 先说结论

18 个离线训练和两套固定闭环评测全部完成。结论不是“Diffusion 在低数据下明显更强”：25% 时 Diffusion 只领先 6.17 个百分点，低于预注册的 10 个百分点门槛；50% 和 100% 时 BC 反而更强。

正式标签：`PASS_S8R1A_PRIOR_DATA_ABLATION_BC_COMPETITIVE_OR_BETTER`

这是一条有效科研结果。本轮没有回训、改阈值、换 seed、增加 PPO 或进入后续拓扑。

## 直接回答六个问题

1. 25%：BC 为 26/54、27/54、29/54；Diffusion 为 27/54、40/54、25/54。均值分别为 50.62% 和 56.79%。
2. 50%：BC 为 28/54、37/54、34/54；Diffusion 为 34/54、31/54、26/54。均值分别为 61.11% 和 56.17%。
3. 100%：BC 为 32/54、40/54、37/54；Diffusion 为 27/54、28/54、23/54。均值分别为 67.28% 和 48.15%。
4. Diffusion 优势没有随数据减少而稳定扩大：25% 为 +6.17 个百分点，50% 为 −4.94，100% 为 −19.14。预注册条件 A+B 均不成立。
5. Fresh topology 没有同样的稳定趋势：25% BC 12.35% vs Diffusion 11.11%；50% BC 5.56% vs Diffusion 10.49%；100% BC 5.56% vs Diffusion 3.70%。
6. Diffusion-specific 低数据论文故事没有变强；本轮更支持“BC competitive or better”。

## 预注册判定

- 25% paired Diffusion−BC：`+0.061728`，sample std `0.161794`。
- 50% paired Diffusion−BC：`−0.049383`，sample std `0.140220`。
- 100% paired Diffusion−BC：`−0.191358`，sample std `0.087515`。
- 条件 A（25%、50% 两个 reduced budgets 均高于 BC）：失败。
- 条件 B（25% 或 50% 至少 +0.10）：失败。

## 冻结配置与数据

- Diffusion 完全使用 S3-R2 bounded-x0 contract：34D、H=16、action_dim=3、T=100、DDIM=10、eta=0、unit-ball、cosine schedule；只改变 TRAIN subset 和 training seed。
- BC 完全使用 S7 matched architecture：34→128→128→256→4 residual blocks→48，592,688 参数；SmoothL1、AdamW、lr=3e-4、batch=512、10,000 updates。
- 数据只来自 S2 TRAIN 与 S3-R2 TRAIN-only recovery；manifest seed `20260817`。25/50/100% 为每 scene 7/14/28 task，recovery episode 按 task_id 整条保留。
- training seeds：`20260820`、`20260821`、`20260822`；正式 run 数 18；PPO run 数 0。
- Current VAL 为原 54 tasks；Fresh 为已有 S7 development topology，不称 untouched final test。首次 Current VAL 前已冻结模型。
- Diffusion runtime：`FAST_CUDA_GRAPH_EXACT_BATCH_ONE`；未使用 historical_batched、batched DDIM 或 FP16 approximate inference。

## Git 与证据

- TASK：`S8-R1A-OFFLINE-EXPERT-DATA-BUDGET-BC-VS-DIFFUSION-ABLATION-V1`
- START_HEAD：`1b160f5e5ba2d785196ce5ee74e1a10378e6804e`
- END_HEAD：`d2d1885`（evidence commit）
- REMOTE_HEAD：`d2d1885`（S8R1A branch）
- CANONICAL_MAIN：`1b160f5e5ba2d785196ce5ee74e1a10378e6804e`
- BRANCH：`agent/s8r1a-offline-data-ablation-v1`
- DATA_BUDGET_SEED：`20260817`
- manifest hashes：25% `e328ff96256f3cd753fcc1e85467c27a1d79679216bf25296a0b3a6c2fd5a8b2`；50% `841915f12f4523f30090a4286c07eff6a17aad8a33c3ecf79d925f98f305974b`；100% `daefc04fdceff2c82343af74c93e5af00ffc3751cc54f69030325e9e662fb0a8`
- BC_CONFIG_HASH：`741547b64f0405ce4a2423e5d87b9d3068d82f98c6edec54118d0c8d137baa27`
- DIFFUSION_CONFIG_HASH：`b4630c5bce2b2fa9aa3de07d47df2019450c807199cf4375431ab3031ffb7bf8`
- S2_TEST_ACCESSED：`false`；S6_DEVELOPMENT_TEST_ACCESSED：`false`。
- Independent verifier：24/24 checks PASS。
- pytest：`90 passed, 1 skipped, 4 warnings`。

## 产物

- [summary.json](../artifacts/s8r1a/summary.json)
- [independent_verification.json](../artifacts/s8r1a/independent_verification.json)
- [data_budget_25.json](../artifacts/s8r1a/data_budget_25.json)、[data_budget_50.json](../artifacts/s8r1a/data_budget_50.json)、[data_budget_100.json](../artifacts/s8r1a/data_budget_100.json)
- [current_val_metrics.csv](../artifacts/s8r1a/current_val_metrics.csv)、[fresh_metrics.csv](../artifacts/s8r1a/fresh_metrics.csv)、[paired_metrics.csv](../artifacts/s8r1a/paired_metrics.csv)
- [current_success_vs_data_fraction.png](../artifacts/s8r1a/current_success_vs_data_fraction.png)、[fresh_success_vs_data_fraction.png](../artifacts/s8r1a/fresh_success_vs_data_fraction.png)、[generalization_gap_vs_data_fraction.png](../artifacts/s8r1a/generalization_gap_vs_data_fraction.png)

## 阶段边界

- `FORMAL_PROGRESS = 95%`。
- 不生成 final untouched topology，不开始论文最终 freeze，不宣布项目 100%。
- `UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW`。
