# S7 matched-BC and fresh-topology report

**Final label:** `PASS_S7_PRIOR_EFFECT_AND_FRESH_GENERALIZATION_SUPPORTED_DIFFUSION_SPECIFIC_THRESHOLD_NOT_MET`

## 先说人话

S7 完整回答了两个问题，但答案不是“Diffusion 全面胜出”。

第一，普通 matched BC 已经很强。它在旧 VAL 上单独达到 `31/54`，BC prior + PPO 的三随机种子 Success AUC 平均为 `0.41204`，远高于 Pure PPO 的 `0.15926`。因此 S6 的主要收益首先来自高质量离线 expert prior，而不能全部归功于 Diffusion。

第二，Diffusion 仍表现出一致的小优势：旧 VAL 上单独为 `37/54`，高于 BC 的 `31/54`；三 seed 的 `AUC_DIFFUSION - AUC_BC` 分别为 `+0.05463 / +0.03241 / +0.04722`。但均值 `+0.04475` 没达到预注册 `0.05` 门槛，所以正式分类必须是 `PRIOR_ADVANTAGE_NOT_DIFFUSION_SPECIFIC`，不能宣称 Diffusion-specific online advantage 已被证明。

在完全冻结的新拓扑上，Diffusion 为 `10/54`，BC 为 `4/54`，Pure PPO 三 seed 均为 `0/54`。这支持“Diffusion 在 fresh topology 上相对 BC/Pure 保留了有限优势”。但 Greedy-goal 是 `18/54`，Diffusion unsafe 为 `30/54`，而且 BC 与 Diffusion 相对旧 VAL 的 success-rate 都下降 `0.50`。因此泛化是有限的，不是稳健部署或安全性结论。

## Matched BC contract

- `34D -> H16 x 3` deterministic sequence；每一步使用 open-unit-ball squash，只执行第一步并 receding-horizon 重算。
- MLP：`34 -> 128 -> 128 -> 256`，四个 width-256 residual blocks，输出 48D。
- BC 参数 `592,688`；Diffusion denoiser 参数 `621,360`；比例 `95.38%`。
- SmoothL1 + AdamW，batch 512，lr `3e-4`，weight decay `1e-4`，10,000 updates，seed `20260810`。
- best update `10000`，VAL sequence loss `0.0001857527`。
- 训练窗口与 S3-R2 完全同预算：S2 TRAIN `125,678` + TRAIN-only recovery `66,939` = `192,617`；normalization 只来自 S2 TRAIN；S2 TEST 未参与。
- frozen BC checkpoint SHA256: `23c8ead92cc53851bdd295dfb32175cac62cca86b040ce0f8b3701f04c41015a`。

Offline dataset identity：

- S2 TRAIN SHA256: `701a1d36b767ff41347b1dac60868922ce5033ee8db27721daf891613125db21`
- S2 VAL SHA256: `c5687f812a958f28dce14c4a2742ae64a94bb330229747c7c64e85290134dbcc`
- S3-R2 recovery TRAIN SHA256: `ac87057d9363ee0557f8f02f2dcd2f1d4a27ae3bd956307800953eae2298f119`
- VAL windows: `27,127`；VAL 只用于 checkpoint selection。

## BC-only vs Diffusion-only on current VAL

| Method | Family | Tasks | Success | Collision | Unsafe | Timeout | Mean return |
|---|---|---:|---:|---:|---:|---:|---:|
| BC only | ALL | 54 | 31 | 12 | 12 | 11 | 13.354 |
| BC only | OPEN | 18 | 15 | 0 | 0 | 3 | 21.930 |
| BC only | BLOCK | 18 | 8 | 8 | 8 | 2 | 4.769 |
| BC only | SBEND | 18 | 8 | 4 | 4 | 6 | 13.364 |
| Diffusion only | ALL | 54 | 37 | 9 | 9 | 8 | 17.910 |
| Diffusion only | OPEN | 18 | 17 | 0 | 0 | 1 | 24.959 |
| Diffusion only | BLOCK | 18 | 9 | 8 | 8 | 1 | 6.038 |
| Diffusion only | SBEND | 18 | 11 | 1 | 1 | 6 | 22.731 |

BC 没有完全复制 Diffusion：总成功少 6 个任务，主要差距来自 OPEN 和 SBEND；Diffusion 的 collision/unsafe 也更少，但本表不足以推翻项目级 `H3_SAFETY = NOT_SUPPORTED`。

## Current-distribution online evidence

| Seed | Pure PPO AUC | BC+PPO AUC | Diffusion+PPO AUC | Diff - BC | BC - Pure |
|---:|---:|---:|---:|---:|---:|
| 20260812 | 0.14630 | 0.43056 | 0.48519 | +0.05463 | +0.28426 |
| 20260813 | 0.13796 | 0.39722 | 0.42963 | +0.03241 | +0.25926 |
| 20260814 | 0.19352 | 0.40833 | 0.45556 | +0.04722 | +0.21481 |

Mean AUC: Pure `0.15926`, BC+PPO `0.41204`, Diffusion+PPO `0.45679`。BC prior 的收益是 3/3 positive；Diffusion 相对 BC 也是 3/3 positive，但 mean delta `0.04475 < 0.05`。

BC+PPO selected steps 是 `150k / 50k / 0`；Diffusion+PPO 仍是 `0 / 0 / 0`。完整曲线显示两类 residual 在后期普遍侵蚀 prior，不能写成稳定 fine-tuning improvement。

### Three-seed success curves

下表每个单元格依次对应 `0/50k/100k/150k/200k/250k/300k/350k/400k/450k/500k`，数值是 54 个 VAL task 中的成功数。机器可读逐点证据为 `artifacts/s7/online_seed_curves.csv`。

| Method | Seed | Success count at 11 evaluation points |
|---|---:|---|
| Pure PPO | 20260812 | 0/0/6/10/12/11/6/7/7/13/14 |
| Pure PPO | 20260813 | 0/0/7/9/10/8/15/10/3/7/11 |
| Pure PPO | 20260814 | 0/1/11/16/11/13/10/10/11/13/17 |
| BC+PPO | 20260812 | 31/27/26/35/25/26/20/18/18/13/18 |
| BC+PPO | 20260813 | 31/32/27/28/29/21/15/13/15/11/16 |
| BC+PPO | 20260814 | 31/30/27/30/27/29/14/15/15/11/14 |
| Diffusion+PPO | 20260812 | 37/32/27/28/28/29/22/24/23/19/23 |
| Diffusion+PPO | 20260813 | 37/29/33/32/33/17/19/22/15/11/5 |
| Diffusion+PPO | 20260814 | 37/30/24/33/28/27/20/22/20/15/17 |

## Fresh topology holdout

- generation seed `20260816`；6 families，54 tasks，每 family 9 tasks。
- topology hash: `504a08fcbac4ac5bb20a411212586b63f80d26a89724170f74bcb8d66153870d`。
- task manifest hash: `4bcd49c082a1118e6fb6b8c47e2ace80675f18934772683eb63c75f5454c3ca3`。
- families: diagonal crossover, double alternating, triple chicane, narrow gate, wide lateral detour, offset twin gate。
- Manifest 在任何 policy evaluation 前冻结。GCOPTER `reference_pass` 只作为离线几何/轨迹 feasibility、collision-free、goal-reachability certificate；旧低层 reference playback 另存为诊断，不冒充 certification gate。

| Method | Success / 54 | Collision | Ground contact | Unsafe | Timeout | Mean return |
|---|---:|---:|---:|---:|---:|---:|
| Greedy goal | 18 | 36 | 0 | 36 | 0 | -0.416 |
| Pure PPO seed 1/2/3 | 0 / 0 / 0 | 19 / 3 / 2 | 0 / 0 / 0 | 19 / 3 / 2 | 35 / 51 / 52 | -5.299 / -0.383 / -0.066 |
| BC only | 4 | 20 | 0 | 20 | 30 | -0.798 |
| BC+PPO seed 1/2/3 | 6 / 6 / 4 | 26 / 24 / 20 | 1 / 0 / 0 | 27 / 24 / 20 | 21 / 24 / 30 | -2.567 / -0.877 / -0.798 |
| Diffusion only | 10 | 25 | 5 | 30 | 14 | -0.603 |
| Diffusion+PPO seed 1/2/3 | 10 / 10 / 10 | 25 / 25 / 25 | 5 / 5 / 5 | 30 / 30 / 30 | 14 / 14 / 14 | -0.603 |

### Per-family fresh success

每个数字均为 9 个 task 中的成功数；三 seed 方法按 `20260812/20260813/20260814` 排列。完整的 per-family collision、unsafe、timeout、return 位于 `artifacts/s7/fresh_metrics.csv`。

| Family | Greedy | Pure PPO 3 seeds | BC | BC+PPO 3 seeds | Diffusion | Diffusion+PPO 3 seeds |
|---|---:|---:|---:|---:|---:|---:|
| DIAGONAL_CROSSOVER | 0 | 0/0/0 | 0 | 0/0/0 | 0 | 0/0/0 |
| DOUBLE_ALTERNATING | 0 | 0/0/0 | 0 | 1/1/0 | 4 | 4/4/4 |
| NARROW_GATE | 9 | 0/0/0 | 1 | 3/1/1 | 3 | 3/3/3 |
| OFFSET_TWIN_GATE | 9 | 0/0/0 | 2 | 2/1/2 | 2 | 2/2/2 |
| TRIPLE_CHICANE | 0 | 0/0/0 | 0 | 0/0/0 | 0 | 0/0/0 |
| WIDE_LATERAL_DETOUR | 0 | 0/0/0 | 1 | 0/3/1 | 1 | 1/1/1 |

Current-to-fresh generalization gap：BC `31/54 -> 4/54`，success-rate delta `-0.50`；Diffusion `37/54 -> 10/54`，success-rate delta 同为 `-0.50`。Diffusion 在 fresh 上高于 BC 和 Pure PPO，但绝对成功率只有 `18.52%`，因此分类为 `DIFFUSION_GENERALIZATION_ADVANTAGE_LIMITED`，不能写成稳健泛化。

Diffusion+PPO 三个 selected checkpoint 都是 step 0；deterministic residual 被独立验证为 exact zero，因此其 fresh deployment rows 合法复用 frozen Diffusion-only 结果，并明确记录 provenance。这不是 residual 学习成果。

## Leakage, failures, storage, and regression

- S2 TEST 没有用于 S7 BC training、checkpoint selection、BC+PPO training 或 S7 analysis。S6 TEST 已是 development test，不再称 untouched final test。
- Fresh topology/task 没有参与训练、architecture/hyperparameter selection 或 checkpoint selection。
- 保留了四类工程失败：BC VAL schema 字段、SB3 type import、外部 120 秒中断的 100k partial run、Python 3.10 newline 参数；均未改变科学配置。
- Fresh generation 的 corridor/connectivity 与 vertical-reference feasibility failures 原样保留；scene/task manifest 只在 policy evaluation 前冻结一次。
- 独立验证完成后已删除 33 个正式中间 checkpoint，共 `57,178,920` bytes；删除前逐文件记录 SHA256。长期保留每个正式 run 的 best/last、BC best/last、曲线、raw rollout，以及被 120 秒外层超时中断的失败尝试 checkpoint。删除的正式中间权重不能从本地直接恢复，但其身份与清理清单保存在 `artifacts/s7/storage_manifest.json`。

Regression：完整测试为 `87 passed, 1 skipped, 4 warnings`；S7 独立复核重新计算 topology/task hash、648 条 fresh rollout、paired AUC、分类、step-0 residual 等价复用和 leakage contract，结果为 `PASS`。

## What was proven

- 高质量 expert prior 是 online environment-interaction sample-efficiency 的主要来源。
- Matched BC 无法完全复制 Diffusion：旧 VAL `31 < 37`，fresh `4 < 10`。
- Diffusion 相对 BC 的 online AUC 在 3/3 seeds 为正，但其 mean `0.04475` 未达到预注册 `0.05`，所以尚未证明 Diffusion-specific online advantage。
- Frozen Diffusion 在 fresh topologies 上仍有有限非零能力并高于 BC/Pure。

## What was not proven

- 没证明 Diffusion 对 BC 的 online 优势达到预注册强度。
- 没证明稳健跨拓扑泛化；两种 prior 的 success-rate gap 都是 `-0.50`。
- 没证明安全性改善；S7 继续保持 `H3_SAFETY = NOT_SUPPORTED`。
- 没证明 PPO residual 稳定改善任一 prior，也没证明 total-data efficiency、真实无人机或动态场景能力。

## Six required research answers

1. 普通 BC 能复制 expert-prior 的大部分在线收益，但不能完全复制 Diffusion：BC+PPO mean AUC `0.41204`，明显高于 Pure `0.15926`，但低于 Diffusion `0.45679`。
2. Diffusion 对 BC 有一致但未过预注册强度的额外价值：3/3 paired delta 为正，mean `0.04475 < 0.05`，所以不能宣称 diffusion-specific online advantage 已被证明。
3. Expert prior 本身贡献很大：`BC_PPO - Pure_PPO` mean AUC 为 `+0.25278`。
4. Diffusion 在未见 topology 上仍能工作，但只达到 `10/54`，属于有限泛化，不是可靠部署。
5. PPO residual 仍没有稳定 improvement；Diffusion 三 seed selected checkpoint 全是 step 0，BC residual 后期也普遍侵蚀 prior。
6. 当前最准确的论文定位是：**规划专家训练的离线动作先验显著提升 PPO 的在线环境交互样本效率；Diffusion 相对 matched deterministic BC 呈现一致但尚未达到预注册阈值的增益，并在全新拓扑上保留有限优势。**

**FORMAL_PROGRESS = 85% pending controller acceptance**

**UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW**

## Reproducibility handoff

- `START_HEAD = 0aea5782b8bdd8889327da45909127d402ffd8ef`
- `EVIDENCE_END_HEAD = 25b01cce83d47b356b33270b8e7d1a648dc8088d`
- `BRANCH = agent/s7-bc-generalization-v1`
- The final pushed branch head is recorded in the controller handoff message after this metadata-only commit.
