# S6 三随机种子核心验证报告

## 先说人话

本轮完成了 6 个全新的 500k 正式训练，不复用旧 Seed 1。三组对比中，Diffusion+PPO 的完整 0-500k success AUC 都高于 Pure PPO，平均绝对归一化提升为 `0.29753`，超过预注册门槛 `0.10`。因此可以确认：冻结的规划引导 Diffusion prior 显著提高了 PPO 的在线 environment-interaction sample efficiency。

但是，PPO residual 没有继续改善 frozen Diffusion。三个 seed 的训练后最佳 success 都低于 prior 的 `37/54`，最终分别降到 `23/54`、`5/54`、`17/54`。因此论文不能声称“PPO 会把 Diffusion 优化得更好”；准确定位应是“规划引导的 Diffusion prior 提高在线 RL 样本效率”。

安全性结论没有通过。Seed 1 的碰撞/unsafe AUC 更低，但 Seed 2、3 更高，三 seed 平均 collision AUC 反而增加 `0.05586`，unsafe AUC 增加 `0.05617`。不能声称方法稳定降低碰撞。

正式标签为：

```text
PASS_S6_ONLINE_SAMPLE_EFFICIENCY_SUPPORTED_RESIDUAL_NOT_SUPPORTED
CONDITIONAL_GO_PRIOR_EFFECT_ONLY
```

正式进度仍保持 `70%`，等待 controller 审核；本报告不自行提升到 85%，也不进入 S7。

## 完成与未完成

已完成：

- Pure PPO 与 Diffusion+PPO 各 3 个 seed、每个精确 500,000 TRAIN interactions。
- 每个 run 保存并在训练后统一评估 0/50k/.../500k 共 11 个 checkpoint。
- 6 个 run 共 66 个 checkpoint 的 VAL，以及每点 54 个逐任务结果。
- OPEN、BLOCK、SBEND 与 BLOCK+SBEND obstacle success 原始统计。
- primary/secondary AUC、paired deltas、bootstrap CI、安全性、稳定性和 residual-vs-prior 判断。
- H1 通过并冻结全部 VAL 后，一次性执行 development TEST；TEST 未参与选择或方法修改。
- 独立重算全部通过；完整 regression 为 `82 passed, 1 skipped, 4 warnings`。
- 验证后清理 66 个中间 checkpoint（114,357,840 bytes），每个 run 长期保留 best/last、哈希、曲线、summary 和 VAL/TEST 原始结果。

未完成或未证明：

- 未证明 PPO residual 改善 frozen Diffusion；实际结论为 `RESIDUAL_LEARNING_DEGRADES_PRIOR`。
- 未证明降低碰撞或 unsafe；`H3_SAFETY = NOT_SUPPORTED`。
- 未证明 total data efficiency；Diffusion 使用了 S2 offline GCOPTER expert data 和 S3 offline training。
- 未执行 S7、跨拓扑、真实飞行或部署验证。

## 冻结合同与身份

```text
TASK = S6-R0-THREE-SEED-CORE-SAMPLE-EFFICIENCY-VALIDATION-V2
START_HEAD = 9bfab829e697b1c862b2d2e9acd8fdb20e43cceb
END_HEAD = 429a9c658c14f1be2006819c861a0cb92be7c903
REMOTE_EVIDENCE_HEAD = 429a9c658c14f1be2006819c861a0cb92be7c903
BRANCH = agent/s6-three-seed-core-validation-v2
FORMAL_PROGRESS = 70%
SEEDS = 20260812, 20260813, 20260814
N_ENVS = 8
TRAIN_BUDGET = 500000 per run
DIFFUSION_RUNTIME = FAST_CUDA_GRAPH_EXACT_BATCH_ONE
HISTORICAL_BATCHED_DDIM = EXCLUDED
S5R1_Q_LCB = EXCLUDED
```

`END_HEAD` 与 `REMOTE_EVIDENCE_HEAD` 指向实现、6 个正式 run 和全部实测证据的不可变 commit。后续仅有一个 handoff metadata commit 写入这些 SHA，不改变实验或数值。

Diffusion checkpoint SHA-256：

```text
98de9a5d765ec1aeb48648497ac44ec98a6b6a071a607a01b4e49f3e13ab76cd
```

TRAIN dataset SHA-256：

```text
701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21
```

Reward contract SHA-256：

```text
92afefbf338d0ccbbcc42c12a572327413ca66f46bf1f061f637e68d125ae3ec
```

三个 residual run 的 step 0 都复现 `37/54, unsafe=9`；训练和评估 preflight 均确认 fast/reference 位级一致、Diffusion frozen、historical batched 未使用。旧 Seed 1 仅作为历史开发证据保留，不进入本轮任何统计。

## Primary：SUCCESS_AUC_0_500K

| Seed | Pure PPO | Diffusion+PPO | Paired delta |
|---|---:|---:|---:|
| 20260812 | 0.14630 | 0.48519 | +0.33889 |
| 20260813 | 0.13796 | 0.42963 | +0.29167 |
| 20260814 | 0.19352 | 0.45556 | +0.26204 |

```text
3/3 paired deltas > 0
mean delta = 0.2975308642
sample std = 0.0387600747
paired bootstrap 95% CI = [0.2620370370, 0.3388888889]
H1_ONLINE_SAMPLE_EFFICIENCY = SUPPORTED
```

0-200k secondary early-learning AUC 增量分别为 `+0.45139`、`+0.50000`、`+0.39815`。

## 障碍场景、回报与安全

BLOCK+SBEND obstacle-success AUC 增量分别为：

```text
+0.31528, +0.28472, +0.25278
mean = +0.28426
```

Pure PPO 在全部 seed、全部 checkpoint 上的成功都来自 OPEN，BLOCK+SBEND success 为 0；Diffusion prior 提供了实质性的障碍导航能力。

Return AUC paired 增量分别为 `+12.85443`、`+8.44525`、`+7.13553`，均为正。

Collision AUC paired 增量为：

```text
-0.05926, +0.12037, +0.10648
mean = +0.05586
```

Unsafe AUC paired 增量为：

```text
-0.05926, +0.12130, +0.10648
mean = +0.05617
H3_SAFETY = NOT_SUPPORTED
```

## Residual PPO 是否改善 prior

Frozen prior reference 为 `37/54`。三个 residual seed 的训练后最佳、warmup 后最佳和 final 如下：

| Seed | Best after training | Best post-warmup | Final 500k | AUC vs constant prior |
|---|---:|---:|---:|---:|
| 20260812 | 32 | 29 | 23 | -0.20000 |
| 20260813 | 33 | 33 | 5 | -0.25556 |
| 20260814 | 33 | 33 | 17 | -0.22963 |

```text
RESIDUAL_LEARNING_DEGRADES_PRIOR
```

H1 不能覆盖这一结论。样本效率优势来自强 prior 起点和早期保持，而不是 residual PPO 把 prior 训练得更好。

## 稳定性与 steps-to-threshold

50k、100k、200k 的三 seed success mean 均高于 Pure PPO，且按冻结判据，seed std 未超过 Pure std 加 0.05：

```text
H_STABILITY = SUPPORTED
```

Diffusion+PPO 因 step 0 已为 37/54，三个 seed 的 `steps_to_50_percent_success = 0`。Pure PPO 三个 seed 均为 `NOT_REACHED_WITHIN_500K`。

## Development TEST

TEST 只在 6 个 run 全部完成、checkpoint selection 与 VAL 分析冻结、H1 判定为 SUPPORTED 后执行一次。

| Method | Seed | Frozen selected step | TEST success | TEST unsafe |
|---|---:|---:|---:|---:|
| Pure PPO | 20260812 | 500k | 16/54 | 30 |
| Pure PPO | 20260813 | 300k | 16/54 | 8 |
| Pure PPO | 20260814 | 500k | 16/54 | 0 |
| Diffusion+PPO | 20260812 | 0 | 38/54 | 9 |
| Diffusion+PPO | 20260813 | 0 | 38/54 | 9 |
| Diffusion+PPO | 20260814 | 0 | 38/54 | 9 |

三个 Diffusion+PPO best 都是 step 0，因此 TEST 的 `38/54` 是 frozen Diffusion prior 的结果，不能归功于 residual PPO。

## 实测运行时间

| Method | Seed | Train | Post-hoc VAL | Total | Train steps/s |
|---|---:|---:|---:|---:|---:|
| Pure PPO | 20260812 | 12.03 min | 11.08 min | 23.11 min | 692.64 |
| Pure PPO | 20260813 | 5.94 min | 16.11 min | 22.05 min | 1402.04 |
| Pure PPO | 20260814 | 12.72 min | 23.53 min | 36.25 min | 655.21 |
| Diffusion+PPO | 20260812 | 12.03 min | 12.20 min | 24.24 min | 692.49 |
| Diffusion+PPO | 20260813 | 12.07 min | 12.71 min | 24.78 min | 690.48 |
| Diffusion+PPO | 20260814 | 12.18 min | 11.44 min | 23.62 min | 684.14 |

三个 Diffusion run 的完整实测均位于 S6-P0 预测的 20-30 分钟范围内。平均 GPU utilization 约 `45.1%-46.1%`，峰值 VRAM `2358-2570 MiB`。Pure Seed 3 VAL 保留了 300k/350k 的真实异常长尾（371.26 s/347.01 s），没有替换或放宽 deadline。

## 曲线与原始证据

- `artifacts/s6/success_vs_env_steps.pdf`
- `artifacts/s6/collision_vs_env_steps.pdf`
- `artifacts/s6/return_vs_env_steps.pdf`
- `artifacts/s6/obstacle_success_vs_env_steps.pdf`
- `artifacts/s6/aggregate_curves.csv`
- `artifacts/s6/paired_metrics.csv`
- `artifacts/s6/per_run_metrics.csv`
- `artifacts/s6/runs/*/val_rollouts.csv`
- `artifacts/s6/runs/*/test_rollouts.csv`
- `artifacts/s6/independent_verification.json`
- `artifacts/s6/storage_manifest.json`

## WHAT_WAS_PROVEN

Planning-guided frozen Diffusion prior 在相同 500k online interaction budget 下，对三个预注册 seed 均显著提高 Pure PPO 的 success AUC、obstacle-success AUC 与 return AUC；平均 success AUC 绝对提升约 0.298。

## WHAT_WAS_NOT_PROVEN

没有证明 PPO residual 改善 Diffusion prior，没有证明安全性提升，也没有证明 total data efficiency、跨分布适应、跨拓扑泛化或真实飞行能力。

```text
UNIQUE_NEXT_TASK = NONE - WAIT_FOR_CONTROLLER_REVIEW
```
