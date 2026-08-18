# S8-R5B 渲染风格专家轨迹地图汇报素材

## 先说人话

这批图的目的，是给导师汇报时直观展示“专家地图和轨迹长什么样”。它们是展示素材，不是算法比较图，也不证明任何 BC、Diffusion 或 PPO 模型已经训练成功。

本轮只读取冻结的 S8-R2 10K 专家数据：障碍物位置/尺寸来自 `map_manifest.json`，轨迹顶点来自 `train.npz` 的 TRAIN episode。没有运行仿真、没有打开 TEST 轨迹载荷、没有训练模型，也没有修改科研合同。

## 选样

选样是固定的、可复现的，不使用随机抽样：

| 主图 | family | map_id | task_id | 选择理由 |
|---|---|---|---|---|
| Double Gate | `DOUBLE_GATE` | `DOUBLE_GATE_075` | `DOUBLE_GATE_075_TASK_08` | 双门结构清楚，轨迹有横向与高度变化，适合作为主视觉。 |
| S-Bend | `SBEND_RANDOM` | `SBEND_RANDOM_023` | `SBEND_RANDOM_023_TASK_05` | 三段交替障碍形成清晰 S 型绕障。 |
| Chicane | `CHICANE_RANDOM` | `CHICANE_RANDOM_090` | `CHICANE_RANDOM_090_TASK_02` | 连续左右偏置能够展示多次路线调整。 |
| Mixed Clutter | `MIXED_CLUTTER` | `MIXED_CLUTTER_078` | `MIXED_CLUTTER_078_TASK_04` | 障碍数量更多、排列更复杂，适合研究问题背景页。 |

补充图固定使用 `DOUBLE_GATE_075` 的 10 条 TRAIN 任务（`task_00`–`task_09`），以及 `MIXED_CLUTTER_078_TASK_04` 上的简洁无人机位置标识。

## 图像风格与数据边界

- 工具：Matplotlib 3D `Poly3DCollection` / `Line3DCollection`，总览图使用 PIL 拼接。
- 输出：单图 `2560×1440`，总览图 `3840×2160`，16:9，PNG。
- 非数据层美化：深色展示背景、地面底板和网格、障碍物面色/边缘高光、轨迹发光底线、轨迹渐变色、起终点标识和位于真实轨迹上的无人机标识。
- 保持不变：障碍物中心、尺寸、数量、轨迹点序列、起点和终点均来自冻结数据；没有手工描线、轨迹平滑、障碍物搬移或 PyBullet 截图。

## 交付图片

主图：

- [render_double_gate_hero.png](../artifacts/s8r5_rendered_maps/figures/render_double_gate_hero.png)
- [render_sbend_hero.png](../artifacts/s8r5_rendered_maps/figures/render_sbend_hero.png)
- [render_chicane_or_detour_hero.png](../artifacts/s8r5_rendered_maps/figures/render_chicane_or_detour_hero.png)
- [render_clutter_or_gate_hero.png](../artifacts/s8r5_rendered_maps/figures/render_clutter_or_gate_hero.png)
- [rendered_expert_map_overview.png](../artifacts/s8r5_rendered_maps/figures/rendered_expert_map_overview.png)

补充图：

- [render_same_map_multi_trajectories.png](../artifacts/s8r5_rendered_maps/figures/render_same_map_multi_trajectories.png)
- [render_single_scene_with_drone_marker.png](../artifacts/s8r5_rendered_maps/figures/render_single_scene_with_drone_marker.png)

## PPT 使用建议

1. 封面或方法背景页：`render_double_gate_hero.png`，双门结构最容易在一眼内讲清楚。
2. 专家数据介绍页：`render_same_map_multi_trajectories.png`，说明同一张地图上存在多个真实 start-goal 任务。
3. 研究问题背景页：`render_clutter_or_gate_hero.png`，突出复杂障碍环境下的空间绕障场景。
4. 一页总览：`rendered_expert_map_overview.png`，用于快速建立 family 多样性直觉。
5. 需要强调“无人机在场景中飞行”时：`render_single_scene_with_drone_marker.png`。

## 机器可读 provenance

- Metadata：[render_manifest.json](../artifacts/s8r5_rendered_maps/metadata/render_manifest.json)
- 独立验证：[verification.json](../artifacts/s8r5_rendered_maps/metadata/verification.json)
- 生成脚本：[generate_s8r5b_rendered_maps.py](../scripts/generate_s8r5b_rendered_maps.py)
- 验证脚本：[verify_s8r5b_rendered_maps.py](../scripts/verify_s8r5b_rendered_maps.py)

metadata 记录了每张图的 `family`、`map_id`、`task_id`、`split`、`selection_reason`、源文件 SHA256、输出分辨率和 `based_on_real_trajectory=true`。

## WHAT_WAS_PROVEN

- 7 张图片已经由冻结 TRAIN 专家轨迹和冻结地图几何生成。
- 5 张主图、2 张补充图、metadata 和说明文档齐全。
- 图片为 16:9 高分辨率 PNG，视觉抽查通过；障碍物、轨迹、起点和终点具有明确层次。
- 独立验证检查了文件哈希、像素尺寸、图像非空、TRAIN task provenance 和源文件哈希。
- 本轮没有训练模型，没有性能比较，没有访问 TEST 轨迹载荷，没有修改科研合同。

## WHAT_WAS_NOT_PROVEN

- 没有证明任何学习模型已经训练、收敛或优于基线。
- 没有做 TEST 评测、泛化比较、消融或新算法实验。
- 代表性样本只用于汇报展示，不代表最难、最优或全数据分布的极值。
- 图片的“渲染风格”是可视化表达，不改变真实地图几何和专家轨迹数据。

## 最终状态

```text
TASK = S8-R5B-RENDERED-EXPERT-TRAJECTORY-MAPS-FOR-ADVISOR-V1
FINAL_LABEL = PASS_S8R5B_RENDERED_EXPERT_MAPS_READY
START_HEAD = d7d11bb02fbcbf9e0dcfd2da54ae8c87b8664814
BRANCH = agent/s8r5b-rendered-expert-maps-v1
TEST_ACCESSED = false
MODEL_TRAINING_RUN_COUNT = 0
SCIENTIFIC_CONTRACT_CHANGED = false
FORMAL_PROGRESS = 95%
UNIQUE_NEXT_TASK = NONE — WAIT_FOR_CONTROLLER_REVIEW
```

最终 `END_HEAD`、`REMOTE_HEAD` 和提交 SHA 以交付消息为准。
