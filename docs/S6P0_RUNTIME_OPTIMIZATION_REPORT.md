# S6-P0 Runtime Optimization Report

## 人话结论

本轮通过。原来的 96.5 分钟不是 RTX 5060 Ti 算力不足，而是大量 batch=1 的小 CUDA kernel、每个动作 10 次 DDIM、未请求却仍执行的 GPU 到 CPU 诊断同步，以及 PyBullet CPU stepping 串行交替造成的等待。原始 batch=1 DDIM 微基准中，约 99% 的 wall time 位于 10-step denoise 与同步路径。

本轮没有修改 Diffusion checkpoint、网络、bounded-x0、10-step DDIM、eta、seed、34D observation、normalization、reward、动作空间、48 Hz、episode horizon、PPO 超参数、TRAIN/VAL/TEST 或 `N_ENVS=8`。正式科研进度仍为 70%，没有重跑 S5-R1，没有启动 S6 多随机种子实验，也没有访问 TEST。

最终采用的 fast mode 不是有漂移的 batched DDIM，而是固定 shape 的 batch-one CUDA Graph。它把完全相同的 eager 运算图捕获后重复执行，保留每个样本独立 seed 和原始浮点运算。64-action 微基准对原始 oracle 的最大误差为 0，位级一致；完整 54-task VAL 仍为 `37/54, unsafe=9`，逐任务 outcome mismatch 为 0。

按固定 8192-step TRAIN-only smoke 和完整 54-task VAL 的实测组件外推，500k TRAIN-only 约 12.5 分钟，11 个 post-hoc VAL 点约 7.4 分钟，总计约 19.8 分钟，相对历史 5792.2 秒约 4.87×。这不是一次完整 500k wall-time 实测，因此正式排期建议使用更保守的 20–30 分钟。

## CPU/GPU 原来浪费在哪里

- 原始 `ddim_sample_x0()` 即使 `return_diagnostics=False`，每个 DDIM step 仍通过 `.item()` 把多个诊断量同步回 CPU，破坏异步 CUDA 流水。
- 小型 MLP 的 batch=1 kernel 太碎，GPU 经常等待 Python 发起下一次 kernel；CPU 同时等待 GPU 返回动作，PyBullet 无法持续推进。
- 训练过程中每 50k 暂停并进入完整 54-task exact evaluator，使训练和评测阶段反复切换。
- 增加 CPU worker 不是无限有效：12-task 基准中，16 workers 最快；20 workers 因 PyBullet、Python 调度和中央推理队列竞争而回落。
- 任务管理器中的 100% GPU 不是目标。修正零残差初始化后的 fast TRAIN smoke 中，CPU 平均利用率约 55.5%，24 个 logical processors 均进入活跃统计；GPU 平均利用率约 42.6%、峰值显存约 2610 MiB，同时 wall time 明显下降。

## 实施内容

1. 保留 `predict_original_reference()` 作为不改动的审计 oracle。
2. `predict_reference()` 缓存固定 DDIM schedule/timestep/scalar tensors，并移除未请求诊断导致的 host synchronization；64-action 与 oracle 位级一致。
3. `predict_fast()` 使用固定 shape CUDA Graph 捕获相同 batch-one recurrence，不 batching、不混合精度；支持 `reference` 与 `fast` 两种正式 runtime mode。
4. 增加单 CUDA owner 的中央 inference scheduler，使独立 PyBullet worker 不再创建私有 CUDA streams。
5. 基准测试 8/12/16/20 evaluation workers，本机冻结 16；更多 worker 并不更快。
6. 增加只保存 0/50k/.../500k checkpoint 的 deferred-evaluation callback，以及训练后按原规则评估和选择 checkpoint 的通用工具。序列化前后 RNG identity 已测试，无 early stopping。
7. 增加逐层 DDIM trace、action、seed、checkpoint selection、worker/runtime mode、数据边界与 frozen-config 回归测试。

## Batched DDIM 专项审计

batched 路径不能用。开环最终动作最大误差仅 `2.3841858e-7`，但中间 `epsilon` 最大误差达到 `9.5367432e-6`；闭环会放大该误差。完整 VAL 结果为 `32/54, unsafe=11`，相对 frozen reference 有 26 个 outcome 字段不一致。因此：

```text
FAST_BATCHED_DIFFUSION_INFERENCE = REJECTED
```

本轮验证通过的 fast path 是：

```text
FAST_CUDA_GRAPH_EXACT_BATCH_ONE = VERIFIED_EQUIVALENT
```

## Before / After

| 指标 | Before | After | 结论 |
|---|---:|---:|---|
| 64-action original reference | 0.6593 s | 0.04232 s | CUDA Graph 15.58× |
| Diffusion actions/s | 97.1 | 1512.3 | 15.58× |
| 54-task eager exact VAL | 221.29 s | 40.20 s | fast exact 5.50× |
| VAL tasks/min | 14.64 | 80.59 | 5.50× |
| 8192-step integrated TRAIN | 73.03 s reference | 12.26 s fast | 5.96× |
| Integrated TRAIN env steps/s | 112.2 reference | 668.4 fast | 5.96× |
| 历史 500k 总 wall time | 5792.2 s | 1190.4 s projected | 4.87× projected |

Pure PPO 的同预算 smoke 为约 1240.9 env steps/s；说明剩余差距主要仍是每个动作必须运行 frozen Diffusion，而非 PPO update 本身。

## Training / VAL 解耦合同

正式 runner 可连续训练并只保存以下冻结 checkpoint：

```text
0, 50k, 100k, 150k, 200k, 250k, 300k, 350k, 400k, 450k, 500k
```

训练结束后统一离线评估 11 个 checkpoint。训练不读取 VAL 结果、没有 early stopping；真实 SB3 PPO checkpoint 在 smoke 后完成序列化，其 Python/NumPy/Torch CPU/CUDA RNG 均保持不变。post-hoc selection 使用相同 deterministic key，所以 learning curve 和 selection rule 均保留。

## 工程候选处置

- `torch.compile(mode="reduce-overhead")`：当前 Windows PyTorch 环境缺少可用 Triton，拒绝作为依赖。
- `torch.compile(backend="cudagraphs")`：实测 64 actions 约 85.7 s，远慢于 eager，拒绝。
- 手工固定 shape CUDA Graph：位级一致且显著加速，接受。
- FP16/BF16：未采用；现有 exact graph 已达到目标，无需引入精度风险。
- batched DDIM：闭环 identity 失败，拒绝。

## 证据与回归

- `artifacts/s6p0/profile_baseline.json`
- `artifacts/s6p0/profile_optimized.json`
- `artifacts/s6p0/equivalence_audit.json`
- `artifacts/s6p0/runtime_benchmark.csv`
- `artifacts/s6p0/independent_verification.json`
- `artifacts/s6p0/summary.json`

独立 verifier 的全部 Gate 通过；完整项目 regression 为 `82 passed, 1 skipped, 4 warnings`。唯一 skip 是无 CUDA 的 CPU 测试环境中的 GPU preflight；CUDA Graph 数值与闭环 Gate 已在 RTX 5060 Ti 环境独立执行。

核心证据 hash：

```text
profile_baseline.json  = 53ec91a47659ae8f631501d2fe516509bcb8574ebfd28da27dad78a5c7164ade
profile_optimized.json = a9e36af69c841f63a923a7711131d784e8527f51abef5a0969c62ea3c27f6bc7
equivalence_audit.json = e5aab51f0b9572c4d900ec49a75ab0cf6183e89349259bc8f3702caa115c369a
runtime_benchmark.csv  = 6a8c8d2fe151ef9f47d8bb9172bbc9bd2c15e6db8eb224fe6d3e50be6940b9a1
```

六个 S6-P0 artifact 文件合计 53,459 bytes。

未来 runner 可用以下显式模式做工程 smoke：

```powershell
python scripts/run_s6p0_runtime.py --phase runtime-smoke --runtime-mode reference
python scripts/run_s6p0_runtime.py --phase runtime-smoke --runtime-mode fast
```

## Git 与边界

```text
START_HEAD = 435e9d58ba41f627195354a333b3f5051d451b4c
END_HEAD = 1b58c3651609f21ab10cffbf3b4f86c2366ee267
REMOTE_HEAD = 1b58c3651609f21ab10cffbf3b4f86c2366ee267
BRANCH = agent/s6p0-runtime-optimization-v1
FORMAL_PROGRESS = 70%
UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW
```

`END_HEAD/REMOTE_HEAD` 指向包含实现、完整实测证据和本报告主体的不可变提交；其后的 handoff-only commit 只补齐上述 SHA 元数据，不改变代码、指标或科学结论。

## WHAT_WAS_PROVEN

- optimized eager exact 与原始 oracle 位级一致。
- CUDA Graph fast mode 与原始 oracle 位级一致，并完整复现 frozen 54-task outcome。
- fast mode 在本机显著提高 batch-one Diffusion、VAL 和短训练吞吐。
- evaluation worker 选择、deferred checkpoint orchestration 和 RNG/selection invariants 有自动证据。

## WHAT_WAS_NOT_PROVEN

- 没有执行完整 500k S6 run；19.9 分钟是组件外推，不是正式 full-run wall time。
- 没有证明 mixed precision 或 batched DDIM 等价。
- 没有访问 TEST，没有启动多随机种子、消融、泛化或 S6 正式科研实验。
