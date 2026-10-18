# TranAD M0–M4 消融实验

本工作流在原版 TranAD 上逐步增加局部因果嵌入、活性掩码稀疏图、周期性门控频域约束与动态趋势损失，以及峭度门控的标准化残差聚合。M0 保留原版两阶段重构结构；扩展模块独立放在 `src/tranad_progressive.py`。

## 变体定义

| 变体 | 算法 |
| --- | --- |
| M0 | 原版 TranAD 两阶段自条件重构损失；测试分数为全通道均方重构误差的均值。 |
| M1 | M0 加核尺寸 1、3、5 的并行深度可分离因果卷积。只在左侧填充，拼接后投影；LayerScale 初始化为 `1e-4`，再与输入投影相加并做 LayerNorm。 |
| M2 | M1 加活性方差掩码和软阈值余弦图注意力。时间方差低于 `1e-5` 的通道走残差恒等路径，不收发图消息；余弦邻接使用初值 `0.3` 的可学习阈值、ReLU 截断和逐行归一化。单变量数据直接绕过图模块。 |
| M3 | M2 加谱平坦度门控的 rFFT 幅度损失、Huber 时域损失和一阶差分趋势损失。频域权重为 `0.1 × sigmoid(10 × (0.3 − SFM))`，趋势损失权重为 `0.1`；不使用 Focal-MSE。 |
| M4 | M3 加训练集通道残差均值/标准差归一化，并以每个时间点的通道超额峭度门控通道均值与 Top-3 均值的混合。 |

FFT 和趋势约束沿有序训练批次的时间轴计算。谱平坦度先移除通道均值并排除 DC 项，再取原信号与一阶差分 SFM 的较大值：这能避免静态偏置主导周期性判断，并让单次阶跃的宽频差分降低频域权重。训练批次不打乱，频域损失会在每个小批次内计算；MPS 上实数 FFT 在 CPU 计算频谱，并把损失送回模型设备。

M4 使用零中心化 sigmoid 门控 `max(0, 2 × sigmoid(kurtosis) − 1)`。这是对原提案中 `sigmoid(kurtosis)` 的数学修正：超额峭度为 0 时普通 sigmoid 输出 0.5，不能像文字说明那样退化到均值分数；中心化门控会在峭度非正时令 Top-k 权重为 0，在正峭度增大时平滑趋近 1。

## 文件改动

- `src/tranad_progressive.py`：独立实现 M1 因果卷积嵌入、M2 活性掩码与图邻接、M3 自适应损失、M4 标准化峭度聚合；保留旧 `adaptive_topk_scores` 以兼容已有实验脚本。
- `src/models.py`：通过延迟加载钩子注册 `TranAD_M0`–`TranAD_M4`，保持原版模型主体和已有模型入口。
- `src/experiment_utils.py`：挂载渐进变体训练损失、M4 绝对重构残差及 M4 评分聚合；加载旧 MSDS 预处理文件时兼容标签的 5 倍采样对齐。
- `src/device.py`：自动选择 CUDA、可用的 MPS 或 CPU；MPS 初始化失败时自动回退 CPU。
- `preprocess.py`：MSDS 的 labels 与 train/test 使用同一个每 5 行采样步长。
- `experiments/run_progressive_ablation.py`：单数据集/变体/Seed 训练和测试，使用时间顺序验证集、早停、最佳检查点与指标记录。
- `run_ablation.sh`：遍历数据集、M0–M4 与五个随机种子；缺失数据集时以黄色警告跳过。
- `aggregate_metrics.py`：按数据集和变体汇总 Seed，输出论文表格格式的均值与样本标准差。
- `ABLATION_GUIDE.md`：本说明及复现实验命令。

## 训练与收敛记录

默认最多训练 15 个 Epoch，命令行 `--epochs` 可覆盖；学习率使用 `CosineAnnealingLR`。验证集取训练序列尾部 10%，Early Stopping 默认 `patience=3`、`min_delta=1e-4`。每个 Seed 只用验证重构损失最低的权重测试。

```text
checkpoints/
└── {dataset}/
    └── {M0|M1|M2|M3|M4}/
        └── seed_{seed}/
            ├── model_best.pt
            └── loss_curve.csv       # epoch, train_loss, val_loss
experiment_logs/
└── {dataset}/{variant}/seed_{seed}/
    ├── train_metrics.csv            # 完整训练诊断数据
    ├── test_metrics.json
    └── test_per_dim.csv
experiments_results/
├── ablation_raw_metrics.csv
└── ablation_summary_mean_std.csv   # Precision, Recall, F1, ROC-AUC 的 Mean ± Std
```

`loss_curve.csv` 每轮原子更新，包含 `epoch,train_loss,val_loss` 三列；`train_loss` 是当轮训练目标总损失，`val_loss` 是验证重构损失。M4 的通道标准化统计与 POT 阈值都由官方训练集残差拟合（包含从训练序列尾部留出的验证段），不使用测试数据或标签拟合统计量。

实验入口只读取 `processed/{dataset}` 中所需的 train/test/labels 文件。自动脚本也检查对应原始数据目录；任一缺失时打印 `[WARN] Dataset <NAME> not found, skip.` 后继续其他数据集。MSL/SMAP 共用 `data/SMAP_MSL` 原始目录。

MSDS 的仓库预处理脚本会生成全零 `labels.csv`，因为该数据包没有提供异常区间标注。这些零值只能作为占位标签，不能用于验证异常检测性能。MSDS 仍会训练并输出异常分数、POT 阈值和预测；当标签只有一个类别时，监督指标 F1、Precision、Recall、ROC-AUC 记为不可用（原始指标 CSV 为 `NaN`，JSON 为 `null`，摘要 CSV 显示 `--`）。`test_metrics.json` 和 `test_per_dim.csv` 中的 `evaluation_status` 会标明 `no_positive_labels`。补充真实标签并重新生成 `processed/MSDS/labels.npy` 后，监督指标才可计算。

## 运行全量实验

在仓库根目录运行：

```sh
sh run_ablation.sh
```

默认数据集为 `synthetic SMD SWaT SMAP MSL WADI MSDS UCR MBA NAB`，Seeds 为 `1 2 3 4 5`，变体为 `M0 M1 M2 M3 M4`。可通过环境变量覆盖列表和设备：

```sh
DEVICE=cpu EPOCHS=1 SEEDS="1" DATASETS="UCR" MODELS="M0 M4" sh run_ablation.sh
```

每次单 Seed 测试后追加到 `experiments_results/ablation_raw_metrics.csv`。全量脚本结束时自动运行聚合器；也可单独重建摘要：

```sh
python3 aggregate_metrics.py
```

聚合器按 `(dataset, model, seed)` 去重并使用样本标准差（`ddof=1`）；重复运行相同 Seed 时，摘要采用最新记录。

## 单独加载最佳权重复测

Linux/CUDA 单行命令：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device cuda --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```

macOS/MPS 单行命令：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device mps --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```

macOS 自动选择 MPS 并在初始化失败时退回 CPU：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device auto --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```
