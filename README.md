# Quadrotor Diffusion PPO

本项目研究规划引导的四旋翼导航学习：固定 34 维可部署观测和 3 维归一化速度动作，以 GCOPTER 专家轨迹建立数据集，再比较 Diffusion、BC 和 PPO 路线。仓库中的模型、奖励、场景、数据划分和实验合同保持原科研定义；本次整理只改善安装、路径和证据组织，不训练模型、不重跑正式 TEST。

## 当前状态

最近正式科研证据是 S8-R4 Conditional 1-D U-Net 架构审计。它是单 seed、TRAIN/VAL-only 的架构审计，记录为 `PASS_S8R4_UNET_PROMISING`；它不证明三 seed 优势、参数量匹配优势、BC 优势，也没有启动 PPO。全部正式数据集继续保留；模型权重只长期保留冻结 VAL 规则选中的 S8-R4 U-Net `pass_10.pt`。历史实验过程、负结果和发展方向见 `docs/RESEARCH_HISTORY_AND_NEXT_DIRECTIONS.md`。

## 目录

```text
src/quadrotor_diffusion_ppo/   环境、专家数据、Diffusion、BC、PPO 和评估代码
scripts/                       训练/评估/审计脚本，以及安装验证和 Windows bootstrap
tests/                         单元与合同测试
configs/                       冻结科研合同
docs/                          阶段报告、环境/存储审计和数据清单
environment/                   最小运行依赖、开发依赖、Conda 文件和 PyTorch 安装脚本
third_party_manifest/          GCOPTER 与 gym-pybullet-drones 来源登记
artifacts/                     可提交的小型证据；大型数据和日志由 .gitignore 排除
checkpoints/                   本机科研 checkpoint，默认不进入 Git
```

## Neural Network Implementations

- MLP Diffusion：`src/quadrotor_diffusion_ppo/diffusion/model.py`
- Temporal 1-D U-Net：`src/quadrotor_diffusion_ppo/diffusion/unet1d.py`
- Behavioral Cloning：`src/quadrotor_diffusion_ppo/bc/model.py`
- PPO components：`src/quadrotor_diffusion_ppo/ppo/`

模型类、输入输出和科研用途见 `docs/NETWORK_ARCHITECTURE_INDEX.md`。

## 从零安装 Windows

推荐 Python 3.10（Python 3.9 也可运行历史 S8-R4 环境，但新环境使用 3.10）。先准备 PowerShell、Git 和一个可用的 Python/conda 环境，然后在仓库根目录执行：

```powershell
conda env create -f environment/environment-windows.yml
conda activate quadrotor-repro-test
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1 -Python python -TorchVariant cu128
```

The Conda file intentionally creates only Python and pip; the bootstrap script
then installs the pinned base/dev dependencies and the editable project in an
order that avoids downloading a CPU PyTorch wheel before the selected CUDA
wheel.

没有 NVIDIA CUDA 12.8 环境时，把最后一个参数改为 `-TorchVariant cpu`。CUDA wheel 不写入普通 PyPI 依赖；`environment/install_torch_windows.ps1` 负责选择官方 PyTorch wheel。

`pip install -e .` 是独立可执行的项目安装步骤：

```powershell
python -m pip install -e .
python scripts/verify_installation.py
```

## 第三方输入

`scripts/bootstrap_windows.ps1` 会把只读输入放在仓库外的 `.deps/`（该目录被 Git 忽略）：

- GCOPTER-Clean-Reproduction：`https://github.com/canimiliya/GCOPTER-Clean-Reproduction.git`，冻结 ref 见 `third_party_manifest/GCOPTER.md`。
- gym-pybullet-drones：`https://github.com/learnsyslab/gym-pybullet-drones.git`，固定 commit `e712698a05a80728b06572819dcf044596707754`。

也可以把它们放在任意位置并设置：

```powershell
$env:GCOPTER_CLEAN_REPRODUCTION = "<path-to-GCOPTER-Clean-Reproduction>"
$env:GYM_PYBULLET_DRONES_ROOT = "<path-to-gym-pybullet-drones>"
$env:GCOPTER_YAML_SCENE_PLANNER = "<path-to-gcopter_yaml_scene_planner>"
```

代码只读取 GCOPTER 的九个 `scenes/pybullet/*.yaml` 和已生成的解析轨迹；不复制第三方源码镜像，不把 planner、motor RPM 或 optimizer state 当作学生观测。

## 数据和 checkpoint

正式数据仍在本机原位置，未移动：

- S2：`artifacts/s2/dataset/{train,val,test}.npz`
- S3-R2 recovery：`artifacts/s3r2/recovery_dataset/`
- S8-R2 10K：`artifacts/s8r2_10k/{train,val,test}.npz`
- S7 fresh-topology holdout：`artifacts/s7/fresh_holdout/`
- S8-R3 TRAIN normalization：`artifacts/s8r3/train_obs_normalization.npz`
- 唯一长期保留权重：`checkpoints/s8r4/unet/pass_10.pt`

大小、SHA256、重建性和清理状态见 `docs/DATA_AND_CHECKPOINT_MANIFEST.md`。数据仍按科研阶段留在规范相对路径中，避免引入个人绝对路径或破坏脚本；本机 `process` 目录中的管理索引不复制大型数据，也不是运行依赖。

完整 pytest 的 frozen-asset 恢复

`python -m pytest -q` 会验证历史 S2/S3-R2 合同，因此除了代码和第三方源码，还需要从受控归档恢复以下外部输入；它们故意不进入 Git：

- `.deps/gcopter_reference/OPEN_00/reference_coefficients.csv`
- `artifacts/s2/dataset/{train,val,test}.npz`
- `checkpoints/s3r2/best.pt`

把归档中的同名文件复制到上述相对路径（不要移动正式源文件），然后按 manifest 中的 SHA256 校验。例如：

```powershell
$archive = "<path-to-frozen-asset-archive>"
Copy-Item "$archive\gcopter_reference\OPEN_00\reference_coefficients.csv" ".deps\gcopter_reference\OPEN_00\"
Copy-Item "$archive\s2\dataset\*.npz" "artifacts\s2\dataset\"
Copy-Item "$archive\s3r2\best.pt" "checkpoints\s3r2\"
Get-FileHash .deps\gcopter_reference\OPEN_00\reference_coefficients.csv -Algorithm SHA256
Get-FileHash artifacts\s2\dataset\train.npz,artifacts\s2\dataset\val.npz,artifacts\s2\dataset\test.npz,checkpoints\s3r2\best.pt -Algorithm SHA256
```

S3-R2 历史 checkpoint 已按所有者批准完成本地清理，只保留上表清单中的原始 SHA256 和实验结论。没有该 optional external asset 时，`verify_installation.py` 仍应通过，但依赖该历史 checkpoint 的测试会明确失败；如需复核该旧合同，必须从独立归档按 SHA256 恢复或按冻结流程重建，不得删除或跳过测试来伪造完整 pytest 通过。

## 验证、测试和运行

```powershell
python scripts/verify_installation.py       # 数十秒内的 import/shape/forward smoke test
python -m pytest -q                         # 完整单元/合同测试
```

典型科研命令（会读取本机外部资产和数据，可能产生大文件；不会被安装验证调用）：

```powershell
python scripts/run_s2_dataset.py
python scripts/run_s3_diffusion.py
python scripts/run_s7_bc.py
python scripts/run_s8r3_training_sufficiency.py
python scripts/run_s8r4_unet_audit.py
python scripts/run_s7_fresh_evaluation.py
```

正式 TEST 的访问边界由各阶段合同和报告控制；不要把这些命令当成“重现正式结果”的快捷方式。正式科研复现应先阅读对应 `docs/*_CONTRACT.md`、数据 manifest 和阶段报告。

## 为什么大文件不进 Git

NPZ 数据、训练 checkpoint、PyBullet 日志、缓存和 planner build products 体积大、变化频繁，并且很多是可由冻结输入重建的派生物。Git 只保存代码、合同、摘要、hash 和可审计的小型证据；本机保留全部正式数据集和唯一选定的 S8-R4 U-Net 权重，恢复/迁移前先核对 SHA256。

## 研究边界

本仓库是仿真研究代码，不是 PX4/ROS 2/真实飞行发布包。GCOPTER 的静态已知地图输入、gym-pybullet-drones 仿真、Diffusion/BC/PPO 训练和 S8-R4 架构审计是不同证据链；一个链路通过不自动证明其他链路通过。
