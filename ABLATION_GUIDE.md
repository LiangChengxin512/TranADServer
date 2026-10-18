# TranAD M0–M4 消融实验

本流程以原版 TranAD 的两阶段重构目标为基线，运行 M0–M4。模型扩展位于 `src/tranad_progressive.py`，`src/models.py` 只注册 M0–M4 入口，并关闭 TranAD TransformerEncoder 的 nested tensor 快路径以消除 PyTorch 警告。

## 变体定义

| 变体 | 算法 |
| --- | --- |
| M0 | 原版 TranAD 架构、两阶段损失权重 `1/n` 与 `1−1/n`、全通道均值分数。M0 不附加扩展模块。 |
| M1 | M0 加 kernel 1、3、5 的并联因果深度可分离卷积，严格左填充；残差 LayerScale 初值为单变量 `0.02`、多变量 `0.1`。 |
| M2 | M1 加 16 个可学习正常原型的隐空间记忆库（残差系数初值 `0.1`），以及余弦软阈值 `0.3` 的稀疏活性通道交互。方差低于 `1e-4` 的通道不参与消息传递；D=1 时图模块短路。 |
| M3 | M2 的重构残差经因果 EMA（`beta=0.35`）和 5 步因果中值/MAD 门控滤波，并叠加小权重的局部/慢速基线偏移分数。训练集低方差通道被屏蔽；慢速基线衰减率为 `0.995`，偏移权重为 `0.05`。 |
| M4 | M3 的原始残差上尝试 `k=1..D` 的 Top-K 通道均值。只有候选的 POT 阈值以上测试尾部分离裕度优于全通道均值时才使用稀疏 k，否则回退到 `k=D`，即 M3 的均值聚合。全程不对通道残差做除以标准差的 Z-score。 |

M4 的防退化条件保证选择分数在定义的无标签尾部分离指标上不低于均值候选；它不能数学上保证真实测试标签上的 Precision、Recall 或 F1 不下降。请以真实标注的独立评估结果判断模型表现，不能把该选择规则解释成分类指标保证。

## M0 训练与早停

`run_progressive_ablation.py` 默认 `--epochs 100`，使用原版实验的 `AdamW` 与 `StepLR(step_size=5, gamma=0.9)`。训练使用全部 `trainD`，不截去尾部。早停验证对全部训练窗口加入固定随机种子的高斯扰动，标准差为 `0.05 * std(trainD)`，并以未扰动的最后时刻训练值计算去噪重构损失。默认 `min_epochs=5`、`patience=3`、相对改善阈值 `rel_delta=0.005`；另外当重构训练损失降至首轮的 `20%`（`--baseline-loss-ratio 0.2`）也会停止，避免继续过拟合重构。两种条件都只影响停止时机；仍只保存验证损失最低的 `model_best.pt`。

POT 的 `init_score` 使用训练集残差，并以每通道 Tukey 下界 `q25 - 1.5*IQR`、`0.05*std(trainD)` 对应的噪声误差下界和机器精度下界中的最大值保护，避免训练重构误差整体趋零时阈值失稳。这个保护只抬高低于下界的残差，不整体平移训练分布，也不读取测试标签。

历史 Seed 1 的 UCR 结果是在每个 Epoch 上按测试 F1 选择权重，因此它只能作历史参考，不能当作无偏验证基线。当前 `model_best.pt` 根据固定噪声去噪验证损失选择，指标可能与该测试集择优结果不同。UCR 历史最佳 epoch 的重构训练损失约为首轮的 `17%`，当前默认的 `20%` 水位用训练损失作停止信号，不读取测试标签。

每轮 `loss_curve.csv` 包含 `epoch,train_loss,val_loss,is_best`。`train_loss` 是原版 TranAD 两阶段训练损失（M3/M4 只改变推理残差处理，不额外改变 M0 的重构目标）。

```text
checkpoints/
└── {dataset}/
    └── {M0|M1|M2|M3|M4}/
        └── seed_{seed}/
            ├── model_best.pt
            └── loss_curve.csv
experiment_logs/
└── {dataset}/{variant}/seed_{seed}/
    ├── train_metrics.csv
    ├── test_metrics.json
    └── test_per_dim.csv
experiments_results/
├── ablation_raw_metrics.csv
└── ablation_summary_mean_std.csv
```

## MSDS 标签限制

MSDS 原始仓库的 `data/MSDS/clean.py` 用全零生成 `labels.csv`；该数据没有随仓库提供异常区间标注。预处理会按每 5 行同步采样数据与标签，加载旧版标签时也会在内存中对齐长度；本机现有的 `labels.npy` 全为 0，不能由切片修复出真实异常标签。

MSDS 仍可训练、生成异常分数和 POT 阈值。由于全零占位标签只有一个类别，F1、Precision、Recall、ROC-AUC 必须标记为不可计算（JSON 为 `null`，汇总表为 `--`），而不能声称得到非零监督指标。只有补入可信的 MSDS 异常标注后，才能验证这些指标。

## 全量脚本

`run_ablation.sh` 是 POSIX `sh` 脚本，默认仅跑 Seed 1、最多 100 Epoch，遍历 `synthetic SMD SWaT SMAP MSL WADI MSDS UCR MBA NAB` 和 M0–M4。缺少原始目录或预处理数组时会打印黄色警告并跳过。正式五种子实验可设置 `SEEDS="1 2 3 4 5"`。

```sh
sh run_ablation.sh
```

环境变量可覆盖设备、Epoch、Seed、数据集和模型；例如快速 CPU 验证：

```sh
DEVICE=cpu EPOCHS=8 MIN_EPOCHS=5 SEEDS="1" DATASETS="UCR" MODELS="M0 M1" sh run_ablation.sh
```

每轮单次测试追加到 `experiments_results/ablation_raw_metrics.csv`，全量脚本结束后自动汇总。单 Seed 时样本标准差显示为 `0.0000`；无效指标显示 `--`。

```sh
python3 aggregate_metrics.py
```

## 从最佳权重单独测试

Linux/CUDA：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device cuda --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```

macOS/MPS（不可用时可改为 `--device cpu`，或使用自动检测）：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device mps --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device auto --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```
