# S8-R5 专家数据导师汇报可视化说明

## 一句话结论

这批图是给导师汇报直接使用的科研素材：它把冻结的 S8-R2 GCOPTER 专家数据从“地图长什么样、轨迹怎样飞、单个样本含什么、数据集有多大”依次讲清楚。它不是模型结果图，也不证明 BC、Diffusion 或 PPO 已经训练成功。

本轮已完成 8 张统一风格的 16:9 图片，每张同时提供 300 dpi PNG 和文字可编辑的 SVG。所有轨迹面板和单帧样本都来自 TRAIN；没有打开 TEST 轨迹载荷，没有运行训练或模型评测，没有修改科研合同。

## 数据基座与边界

- 冻结基座：S8-R2 10K procedural expert dataset。
- 规模：1,000 maps，10,000 expert trajectories，6,120,725 transitions。
- 结构：10 个 environment families，每个 family 100 maps，每张地图 10 个任务。
- 切分：TRAIN / VAL / TEST = 8,000 / 1,000 / 1,000 trajectories，按 map 隔离，map-level leakage = 0。
- 质量门：collision / ground / nonfinite / action-support violation / environment clipping 均为 0。
- TEST 边界：仅使用冻结 summary/manifest 中的汇总事实；本轮唯一打开的轨迹载荷是 `artifacts/s8r2_10k/train.npz`，没有打开 TEST NPZ。
- 算法边界：未训练 BC、Diffusion、PPO，未新增算法实验，未运行 sealed TEST 模型比较。

## 选样与可复现原则

选样完全确定，不使用随机数，因此 `random_seed = null`、`random_selection_used = false`。

1. Family 总览：先从 `task_manifest.csv` 硬筛 `split == train`；每个 family 中选择其障碍几何统计最接近 family 中位数的 TRAIN map（按障碍数量和尺寸的 IQR 标准化 L1 距离），再在该 map 中选择 path/straight ratio 最大的 TRAIN trajectory 作为示意。这个选择用于展示，不宣称它是性能最优或最难样本。
2. 同图十任务：固定使用 `DOUBLE_GATE_075`，该 map 的十个任务全部属于 TRAIN。
3. 3D 与时序主线：固定使用 `DOUBLE_GATE_075_TASK_08`，使地图、三维轨迹和时间序列叙事连续。该轨迹有双门结构、横向与高度变化，适合解释三维飞行；不把约 1.04 的路径比夸大为极端绕障。
4. 单帧观测：在上述 TRAIN map 的十个任务中，选择可见障碍 ray 数最多的完整 H=16 样本；并列时依次按最短 ray、最早时刻和 task ID 决定。最终样本是 `DOUBLE_GATE_075_TASK_09` 的 step 295，11/18 条 ray 在量程内命中障碍，最近 ray 距离约 0.964 m。

精确 episode offset、length、task ID、step、split 和选择规则记录在 [`selection_manifest.json`](../artifacts/s8r5_visualization/metadata/selection_manifest.json)。

## 八张图分别回答什么

| 图 | 导师能从图中直接得到的答案 | 真实数据来源与选择 | 交付文件 |
|---|---|---|---|
| 1. Family overview | 任务并非少数固定小场景，而是 10 类程序化障碍结构。 | 每个 family 的确定性 TRAIN 代表 map + 一条真实 rollout。 | [`expert_family_overview.png`](../artifacts/s8r5_visualization/figures/expert_family_overview.png) / [`SVG`](../artifacts/s8r5_visualization/figures/expert_family_overview.svg) |
| 2. Ten tasks, one map | 同一障碍几何上有十组不同 start-goal 和十条专家轨迹。 | `DOUBLE_GATE_075` 的 10 个 TRAIN tasks；start/goal 取 manifest 精确值。 | [`ten_tasks_same_map.png`](../artifacts/s8r5_visualization/figures/ten_tasks_same_map.png) / [`SVG`](../artifacts/s8r5_visualization/figures/ten_tasks_same_map.svg) |
| 3. 3D trajectory | 专家 rollout 是真实三维飞行路径，包含横向和高度变化并穿越门结构。 | `DOUBLE_GATE_075_TASK_08` 的 position 序列、manifest 障碍和精确 goal。 | [`single_expert_trajectory_3d.png`](../artifacts/s8r5_visualization/figures/single_expert_trajectory_3d.png) / [`SVG`](../artifacts/s8r5_visualization/figures/single_expert_trajectory_3d.svg) |
| 4. Time series | 专家数据不只是曲线图，还包含同步的位置、速度与动作时间序列。 | 与图 3 同一 TRAIN episode；48 Hz，共 589 control steps。 | [`expert_trajectory_timeseries.png`](../artifacts/s8r5_visualization/figures/expert_trajectory_timeseries.png) / [`SVG`](../artifacts/s8r5_visualization/figures/expert_trajectory_timeseries.svg) |
| 5. Sample format | 模型每步看到 34D observation，标签是 3D normalized world-frame target velocity，并可形成 H=16×3 动作窗。 | `DOUBLE_GATE_075_TASK_09` step 295 的真实 observation/action/H16 window。 | [`expert_sample_format.png`](../artifacts/s8r5_visualization/figures/expert_sample_format.png) / [`SVG`](../artifacts/s8r5_visualization/figures/expert_sample_format.svg) |
| 6. Dataset statistics | 数据集规模、轨迹长度、任务距离、路径比、TRAIN 动作范数和 family 平衡情况。 | 冻结 dataset metadata；动作范数仅从全部 TRAIN transitions 重算。 | [`expert_dataset_statistics.png`](../artifacts/s8r5_visualization/figures/expert_dataset_statistics.png) / [`SVG`](../artifacts/s8r5_visualization/figures/expert_dataset_statistics.svg) |
| 7. Observation rays | student 单帧获得局部 18-ray sensing、目标方向和自身状态，而不是完整障碍地图。 | 与图 5 同一真实 TRAIN frame；ray fraction × 3 m 得实际显示长度。 | [`observation_ray_visualization.png`](../artifacts/s8r5_visualization/figures/observation_ray_visualization.png) / [`SVG`](../artifacts/s8r5_visualization/figures/observation_ray_visualization.svg) |
| 8. Data-to-learning bridge | 冻结 rollout 如何切成 episode-bounded observation/action windows，再作为后续 BC/Diffusion 输入。 | 真实数组契约和 H=16 切窗规则的概念图；没有运行训练。 | [`expert_to_learning_pipeline.png`](../artifacts/s8r5_visualization/figures/expert_to_learning_pipeline.png) / [`SVG`](../artifacts/s8r5_visualization/figures/expert_to_learning_pipeline.svg) |

## 样本格式说明

真实 observation 是连续 34D 向量，字段顺序为：

| 索引 | 字段 | 维数 |
|---|---|---:|
| `[0:3]` | position | 3 |
| `[3:6]` | goal − position | 3 |
| `[6:9]` | linear velocity | 3 |
| `[9:13]` | quaternion (xyzw) | 4 |
| `[13:16]` | angular velocity | 3 |
| `[16:34]` | local obstacle-ray fractions | 18 |

真实 action 是 3D normalized world-frame target velocity，即 GCOPTER world velocity 除以 0.801 m/s，支持域为三维欧氏单位球。它不是旧式 4D command。

H=16 样本严格遵循 `observation[t] -> actions[t:t+16]`：不 padding、不跨 episode。冻结 TRAIN 提供 4,776,168 个有效 H=16 windows。图中只解释输入契约，不表示模型已学习到该映射。

## 关键统计

| 项目 | 冻结/复核结果 |
|---|---:|
| Maps | 1,000 |
| Expert trajectories | 10,000 |
| Total transitions | 6,120,725 |
| Families | 10 |
| Trajectories per family | 1,000 |
| TRAIN trajectories | 8,000 |
| TRAIN transitions | 4,896,168 |
| Valid TRAIN H=16 windows | 4,776,168 |
| Map-level overlaps | 0 |
| Collision / ground / nonfinite / action violation / clipping | 0 / 0 / 0 / 0 / 0 |

图 6 的 trajectory length、start-goal distance 和 geometric path ratio 来自冻结全数据 metadata；action norm 分布明确只使用 TRAIN 轨迹载荷。两类来源在图中分开标注，避免暗示打开了 TEST trajectory payload。

## PPT 插图建议表

| PPT 页 | 推荐图片 | 建议讲法 |
|---:|---|---|
| 1 | `expert_family_overview` | 先建立“10 类、1000 maps”的环境多样性直觉。 |
| 2 | `ten_tasks_same_map` | 解释一张地图对应十个不同 start-goal 任务。 |
| 3 | `single_expert_trajectory_3d` + `expert_trajectory_timeseries` | 左侧讲三维几何路径，右侧讲同步状态/动作序列。 |
| 4 | `expert_sample_format` | 解释模型看 34D、学 3D action/H16 window。 |
| 5 | `expert_dataset_statistics` | 集中说明规模、平衡、分布和质量门。 |
| 6 | `observation_ray_visualization` + `expert_to_learning_pipeline` | 从局部观测自然过渡到后续 BC/Diffusion 研究输入。 |

若 PPT 需要裁切、改标题或调整字体，优先插入 SVG；若只需快速汇报，使用 3999×2250 的 PNG。

## 工程产物与复现

- 生成脚本：[`scripts/generate_s8r5_expert_visualization.py`](../scripts/generate_s8r5_expert_visualization.py)
- 独立验证：[`scripts/verify_s8r5_expert_visualization.py`](../scripts/verify_s8r5_expert_visualization.py)
- 回归测试：[`tests/test_s8r5_visualization.py`](../tests/test_s8r5_visualization.py)
- 图目录：[`artifacts/s8r5_visualization/figures/`](../artifacts/s8r5_visualization/figures/)
- 元数据目录：[`artifacts/s8r5_visualization/metadata/`](../artifacts/s8r5_visualization/metadata/)
- 图及源文件哈希：[`figure_manifest.json`](../artifacts/s8r5_visualization/metadata/figure_manifest.json)
- 独立验证结果：[`verification.json`](../artifacts/s8r5_visualization/metadata/verification.json)

推荐复现命令：

```powershell
D:\anaconda\envs\dp_quad_py310\python.exe scripts\generate_s8r5_expert_visualization.py
D:\anaconda\envs\dp_quad_py310\python.exe scripts\verify_s8r5_expert_visualization.py
D:\anaconda\envs\dp_quad_py310\python.exe -m pytest -q tests\test_s8r5_visualization.py
```

生成器会再次校验冻结 manifest SHA-256、每个轨迹选择的 TRAIN split、action 单位球和 H=16 episode 边界。验证器检查 8 组 PNG/SVG、实际文件哈希、16:9 高分辨率、SVG 可编辑文本、TRAIN-only provenance 和冻结源哈希；不会读取 TEST trajectory archive。

## WHAT_WAS_PROVEN

- 8 张强制/建议图均由冻结真实数据和 manifest 可复现生成，文件齐全且风格统一。
- 所有具体 trajectory、observation、action 和 H=16 window 均来自 TRAIN。
- 每张 PNG 是 3999×2250、约 300 dpi 的 16:9 PPT 图；每张另有文字可编辑 SVG。
- 图示的数据规模、split、family balance、质量门和 observation/action/H16 接口与冻结证据一致。
- 独立验证可复核图文件哈希、源哈希、选样 split、TEST 边界和未训练边界。

## WHAT_WAS_NOT_PROVEN

- 没有证明任何 BC、Diffusion 或 PPO 模型已训练、收敛或优于基线。
- 没有进行 TEST 模型评测、泛化比较、消融或超参数实验。
- 代表性轨迹只用于解释真实数据形态，不证明全数据都具有同等绕障幅度，也不构成最优性比较。
- 局部 ray 图说明单帧 sensing 结构；absolute position 和 goal-relative position仍是 observation 的一部分，因此不应表述为“模型完全没有全局状态信息”。
- 数据规模与质量门说明后续研究的数据基础和接口可用，但不能单独证明后续学习方法一定成功。

## 最终状态

```text
TASK = S8-R5-EXPERT-DATA-VISUALIZATION-FOR-ADVISOR-BRIEFING-V1
FINAL_LABEL = PASS_S8R5_EXPERT_DATA_VISUALIZATION_READY
FORMAL_PROGRESS = 95%
UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW
BRANCH = agent/s8r5-expert-visualization-v1
START_HEAD = d7cfd36e8e508f7c68bfd463d203be6e10391d18
```

本报告不自引用尚未生成的提交哈希；最终 `END_HEAD` 与 `REMOTE_HEAD` 由交付消息给出。完成后停止，不自行开始下一科研阶段。
