# S8-R2 10K procedural GCOPTER expert dataset

## 先说人话

这次已经生成并冻结了一个新的专家数据基座：1000 张真正不同的程序化随机地图，每张地图 10 条通过 GCOPTER 和闭环安全验收的专家轨迹，共 10,000 条。旧 S2 数据没有覆盖，TEST 只作为数据集的一部分封存，尚未用于训练、checkpoint 选择或 BC/Diffusion 比较。

1. 最终有多少张真正不同的地图？

   1000 张：10 个 family，每个 family 100 张；map ID 唯一，geometry hash 也全部唯一。

2. 每张地图有多少任务？

   每张地图恰好 10 条 accepted trajectory。每张地图的 start 和 goal 均为 10 个不同点。

3. 地图随机化了什么？

   obstacle 的数量、位置、宽度、深度、高度、相对排列、gate/走廊结构和 detour 方向；每张 map 使用独立的几何 draw，而不是复制同一张地图后只换 start/goal。

4. start/goal 如何保证多样？

   10 个 task slot 固定覆盖沿 ±X、正反 lateral crossing、diagonal、3D 高度关系以及 long-range/high-detour；各 slot 再在对应几何类别内用 PCG64 随机采样。每张图至少 10 个不同 start 和 10 个不同 goal。

5. 10 个 family 是什么？

   `SINGLE_BLOCK`、`OFFSET_BLOCK`、`DOUBLE_BLOCK`、`ALTERNATING_BLOCKS`、`SBEND_RANDOM`、`CHICANE_RANDOM`、`NARROW_GATE`、`DOUBLE_GATE`、`LATERAL_DETOUR`、`MIXED_CLUTTER`。

6. 一共有多少 transition？

   6,120,725 条：TRAIN 4,896,168，VAL 613,153，TEST 611,404。最终文件是三个 concatenated NPZ，并保存 episode offsets/lengths，可恢复任意单条 trajectory。

7. GCOPTER 失败率是多少？

   共尝试 10,002 个 candidate，10,000 个 accepted，2 个 planner failure；candidate acceptance rate 为 99.9800%。失败 candidate 不计入正式数据集，也没有放宽 collision、goal 或动力学 gate。

8. 数据集是否完全无碰撞？

   是。正式 10,000 条轨迹的 collision、ground contact、nonfinite、action support violation 和 environment clipping 全部为 0；observation 为 34D，action 为 3D，所有 action 满足单位球约束。

9. TRAIN/VAL/TEST 是否有地图泄漏？

   没有。split 在 1000 张地图完成后按 family 使用 `SPLIT_SEED=20260826` 做 map-level 划分：TRAIN 800 张图/8000 条轨迹，VAL 100 张图/1000 条，TEST 100 张图/1000 条。三组 map/task overlap 均为 0，TEST 已标记 `TEST_SPLIT_SEALED=true`。

10. 与旧 S2 的差别是什么？

   旧 S2 是 9 张 frozen topology、360 条 trajectory、252 条 TRAIN trajectory、184,871 transitions；新 S8-R2 是 1000 张独立程序化地图、10,000 条 trajectory、8000 条 TRAIN trajectory、6,120,725 transitions。增长不只是 timestep 变多，而是 topology、family 和 map/task 组合都增加。

## 冻结配置与证据

- `MAP_GENERATION_SEED=20260824`，`TASK_GENERATION_SEED=20260825`，`SPLIT_SEED=20260826`，全部使用 NumPy PCG64。
- GCOPTER 使用现有正式 `GCOPTER_PolytopeSFC::setup+optimize` pipeline；没有修改 dynamics、collision margin、speed/acceleration limits 或 trajectory quality gate。
- acceptance 同时要求 planning success、finite rollout、goal reached、zero collision、zero ground contact、zero clipping、observation 34D、action 3D。
- 独立 verifier 共 44 项检查全部 PASS；10 个 family 各选 1 个 map/task 重跑两次，planner coefficient SHA256 全部一致，`DETERMINISM_AUDIT=PASS`。
- 最终 NPZ 约 1.024 GB，生成 cache 约 1.079 GB（均不提交）；三文件、map/task manifest 以及 `dataset_summary.sha256` 均有 SHA256 记录。

## Git 与阶段边界

```text
TASK = S8-R2-10K-PROCEDURAL-GCOPTER-EXPERT-DATASET-V2
START_HEAD = 2d78357c800cb846fb16a3513f728273a91b06bb
ORIGIN_MAIN = 2d78357c800cb846fb16a3513f728273a91b06bb
END_HEAD = 74a6b58 (execution evidence commit)
REMOTE_HEAD = PENDING_PUSH (package commit will be recorded after push)
BRANCH = agent/s8r2-10k-expert-dataset-v2
FORMAL_PROGRESS = 95%
UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW
```

本轮只生成和审计数据；没有训练 BC、MLP Diffusion、Temporal U-Net、PPO，也没有访问 TEST 做模型评估。下一步是否用这一个冻结的 10K 数据集做同 seed 的 BC vs 当前 MLP Diffusion 训练充分性审计，等待 controller 明确授权。
