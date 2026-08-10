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

## Current-distribution online evidence

| Seed | Pure PPO AUC | BC+PPO AUC | Diffusion+PPO AUC | Diff - BC | BC - Pure |
|---:|---:|---:|---:|---:|---:|
| 20260812 | 0.14630 | 0.43056 | 0.48519 | +0.05463 | +0.28426 |
| 20260813 | 0.13796 | 0.39722 | 0.42963 | +0.03241 | +0.25926 |
| 20260814 | 0.19352 | 0.40833 | 0.45556 | +0.04722 | +0.21481 |

Mean AUC: Pure `0.15926`, BC+PPO `0.41204`, Diffusion+PPO `0.45679`。BC prior 的收益是 3/3 positive；Diffusion 相对 BC 也是 3/3 positive，但 mean delta `0.04475 < 0.05`。

BC+PPO selected steps 是 `150k / 50k / 0`；Diffusion+PPO 仍是 `0 / 0 / 0`。完整曲线显示两类 residual 在后期普遍侵蚀 prior，不能写成稳定 fine-tuning improvement。

## Fresh topology holdout

- 6 families，54 tasks，每 family 9 tasks。
- topology hash: `504a08fcbac4ac5bb20a411212586b63f80d26a89724170f74bcb8d66153870d`。
- task manifest hash: `4bcd49c082a1118e6fb6b8c47e2ace80675f18934772683eb63c75f5454c3ca3`。
- families: diagonal crossover, double alternating, triple chicane, narrow gate, wide lateral detour, offset twin gate。
- Manifest 在任何 policy evaluation 前冻结。GCOPTER `reference_pass` 只作为离线几何/轨迹 feasibility、collision-free、goal-reachability certificate；旧低层 reference playback 另存为诊断，不冒充 certification gate。

| Method | Success / 54 | Unsafe | Timeout | Mean return |
|---|---:|---:|---:|---:|
| Greedy goal | 18 | 36 | 0 | -0.416 |
| Pure PPO seed 1/2/3 | 0 / 0 / 0 | 19 / 3 / 2 | 35 / 51 / 52 | -5.299 / -0.383 / -0.066 |
| BC only | 4 | 20 | 30 | -0.798 |
| BC+PPO seed 1/2/3 | 6 / 6 / 4 | 27 / 24 / 20 | 21 / 24 / 30 | -2.567 / -0.877 / -0.798 |
| Diffusion only | 10 | 30 | 14 | -0.603 |
| Diffusion+PPO seed 1/2/3 | 10 / 10 / 10 | 30 / 30 / 30 | 14 / 14 / 14 | -0.603 |

Diffusion+PPO 三个 selected checkpoint 都是 step 0；deterministic residual 被独立验证为 exact zero，因此其 fresh deployment rows 合法复用 frozen Diffusion-only 结果，并明确记录 provenance。这不是 residual 学习成果。

## Leakage, failures, storage, and regression

- S2 TEST 没有用于 S7 BC training、checkpoint selection、BC+PPO training 或 S7 analysis。S6 TEST 已是 development test，不再称 untouched final test。
- Fresh topology/task 没有参与训练、architecture/hyperparameter selection 或 checkpoint selection。
- 保留了四类工程失败：BC VAL schema 字段、SB3 type import、外部 120 秒中断的 100k partial run、Python 3.10 newline 参数；均未改变科学配置。
- Fresh generation 的 corridor/connectivity 与 vertical-reference feasibility failures 原样保留；scene/task manifest 只在 policy evaluation 前冻结一次。
- 正式 BC+PPO checkpoint 中间权重暂时保留到独立验证完成；最终只需长期保留 best/last、哈希、曲线和 raw rollout。

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

**FORMAL_PROGRESS = 85% pending controller acceptance**

**UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW**

## Reproducibility handoff

- `START_HEAD = 0aea5782d6d6b4805e0af04eafdb052ac49ba51e`
- `EVIDENCE_END_HEAD = 25b01cce83d47b356b33270b8e7d1a648dc8088d`
- `BRANCH = agent/s7-bc-generalization-v1`
- The final pushed branch head is recorded in the controller handoff message after this metadata-only commit.
