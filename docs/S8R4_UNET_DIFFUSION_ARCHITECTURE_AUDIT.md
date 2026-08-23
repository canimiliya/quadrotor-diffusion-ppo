# S8-R4 Conditional 1-D U-Net Diffusion Architecture Audit

**FINAL_LABEL:** `PASS_S8R4_UNET_PROMISING`

## 先回答 9 个问题

1. U-Net 参数量：**47,110,915**（MLP 621,360；BC 592,688）。这是 architecture-capacity audit，不是 parameter-matched comparison。
2. 训练稳定性：PASS：training curve 全部 finite，独立 verifier 的 curve_finite 与 val_rows 检查均通过。
3. U-Net success：pass1=97.6%，pass5=98.7%，pass10=99.1%，pass20=99.6%，pass30=99.8%。
4. Offline loss checkpoint：MLP pass 20；U-Net pass 10。选择只看 VAL offline diffusion loss，且在闭环前冻结。
5. Offline-selected delta：+18.30 pp。
6. Late mean delta：+14.40 pp。
7. Peak→final drop：U-Net 0.00 pp；MLP 13.00 pp。
8. Family 改善：见 `family_metrics.csv` 与 family figure；不按单一 family 峰值判定。
9. 是否继续 3 seeds：**是，交 controller 再授权**。

## 固定 checkpoint 对照

| pass | U-Net | MLP Diffusion | BC参考 |
|---:|---:|---:|---:|
| 1 | 97.6% | 82.4% | 0.627 |
| 5 | 98.7% | 95.5% | 0.982 |
| 10 | 99.1% | 92.0% | 0.732 |
| 20 | 99.6% | 80.8% | 0.826 |
| 30 | 99.8% | 82.5% | 0.842 |

## 冻结数据与执行边界

TRAIN SHA256 `395a5c2ab1eb6cb3f3925fa3ecbe160d151833ef95cc5bc3b9187fbc497de3c5`；VAL SHA256 `0fc50886504d36890444b4c09c2990962cc5be6551cb0e54c008df0d26f5246e`；normalization `e05b9a424cbb782dc1eddfcfd69fbf4e849806d3995bbab48e4a60193c76b6e0`。
seed=20260820；effective_passes=30；updates/pass=9329；TEST_ACCESSED=False；PPO_RUN_COUNT=0。
H=16、单帧34D、action3、bounded x0、MSE_x0、T=100、DDIM10/eta0、EAGER_EXACT、first-action receding horizon 均冻结。

## 科研解释边界

如果 U-Net 更强，只能说 temporal U-Net backbone improves the current Diffusion implementation under the frozen 10K contract；single seed 不能证明 Diffusion 优于 BC。若无优势，不自行加入 action chunk、observation history、EMA 或 3 seeds。

## 证据文件

- `artifacts/s8r4/training_curve.csv`
- `artifacts/s8r4/closed_loop_metrics.csv`
- `artifacts/s8r4/family_metrics.csv`
- `artifacts/s8r4/checkpoint_selection.json`
- `artifacts/s8r4/independent_verification.json`
- `artifacts/s8r4/figures/`

FORMAL_PROGRESS = 95%

UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW
