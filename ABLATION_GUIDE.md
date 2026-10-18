# TranAD M0–M4 消融实验

本流程以原版 TranAD 的两阶段重构目标为基线，运行 M0–M4。模型扩展位于 `src/tranad_progressive.py`，`src/models.py` 只注册 M0–M4 入口，并关闭 TranAD TransformerEncoder 的 nested tensor 快路径以消除 PyTorch 警告。

## 变体定义

| 变体 | 算法 |
| --- | --- |
| M0 | 原版 TranAD 架构、两阶段损失权重 `1/n` 与 `1−1/n`、全通道均值分数。M0 不附加扩展模块。 |
| M1 | M0 加 kernel 1、3、5 的并联因果深度可分离卷积，严格左填充；残差 LayerScale 初值为单变量 `0.02`、多变量 `0.1`。 |
| M2 | M1 加 16 个可学习正常原型的隐空间记忆库（残差系数初值 `0.1`），以及余弦软阈值 `0.3` 的稀疏活性通道交互。方差低于 `1e-4` 的通道不参与消息传递；D=1 时图模块短路。 |
| M3 | M2 加两阶段残差谐振：用 `E2 * clip((E2+eps)/(E1+eps), 0.85, 1.25)` 提纯残差，再仅对相对过去两步高出 1.5 倍的孤立峰值做 25% 次大值收缩。全程无 EMA、无 Shift 偏置，不会把当前高分写入未来时间步。另用训练序列前 80% 拟合 POT、后 20% 作无标签验证；只有候选变换的尾部分离裕度至少提升 `max(0.05, 基线裕度绝对值的 10%)` 且越界率不增加时才启用，否则回退到 M2 的平方残差分数。 |
| M4 | 在经 M3 训练侧保护后的残差上选 `k=1..D` 的 Top-K 均值。选择器只把训练序列前 80% 用作 POT 校准、后 20% 作无标签验证；只有分离裕度显著提高，且验证越界率至少降低 10% 时才采用稀疏 k，否则回退到 `k=D`。全程不使用测试标签选 k，也不做通道 Z-score。 |

M3/M4 的防退化条件只比较训练侧验证的尾部分离与越界率；这些无标签判据不能数学上保证真实测试标签上的 Precision、Recall 或 F1 单调递增。模型结构递进不等于分类指标必然递增，需用可信标签验证。若要求逐数据集强制 F1 单调，需要用带标签验证集来选择模型/阈值；不能通过测试集指标择优来冒充无泄漏泛化结果。

## 实现文件

- `src/tranad_progressive.py`：M1 因果卷积、M2 原型记忆/活性通道图、M3 训练侧谐振门控和 M4 Top-K 选择器。
- `src/models.py`：注册 M0–M4 变体，保留 M0 原始 TranAD 路径。
- `src/experiment_utils.py`：MSDS 运行时缓存自愈、M3/M4 分数生成与校准、POT 评估接口。
- `preprocess.py`：MSDS 标签列解析、正样本校验、时间对齐和安全重建缓存。
- `data/MSDS/clean.py`：标明该数据集导出的全零标签文件是占位文件，不是监督标签。
- `experiments/run_progressive_ablation.py`：单 Seed 训练、验证早停、最佳检查点、epoch-five 尺度参考和测试指标输出。
- `run_ablation.sh`：跨平台设备选择、数据/标签检查、Seed 与变体遍历。
- `aggregate_metrics.py`：按数据集和模型汇总指标均值与样本标准差。
- `ABLATION_GUIDE.md`：算法、验证限制、目录和复测说明。

## M0 训练与早停

`run_progressive_ablation.py` 默认 `--epochs 100`，使用原版实验的 `AdamW` 与 `StepLR(step_size=5, gamma=0.9)`。训练使用全部 `trainD`，不截去尾部。早停验证对全部训练窗口加入固定随机种子的高斯扰动，标准差为 `0.05 * std(trainD)`，并以未扰动的最后时刻训练值计算去噪重构损失。默认 `min_epochs=5`、`patience=3`、相对改善阈值 `rel_delta=0.005`；另外当重构训练损失降至首轮的 `20%`（`--baseline-loss-ratio 0.2`）也会停止，避免继续过拟合重构。两种条件都只影响停止时机；仍只保存验证损失最低的 `model_best.pt`。

训练第 5 轮后保存逐通道残差均值/标准差到 `score_reference_epoch5.json`。M3/M4 同时保存谐振候选与 M2 回退分数的尺度，评估时使用该检查点训练侧门控所选分支的统计量。若最佳检查点晚于第 5 轮，POT 的训练 `init_score` 按第 5 轮统计量线性锚定并与原分数逐点取最大值；测试残差不作尺度平移。第 5 轮或更早的检查点不锚定。

历史 Seed 1 的 UCR 结果是在每个 Epoch 上按测试 F1 选择权重，因此它只能作历史参考，不能当作无偏验证基线。当前 `model_best.pt` 根据固定噪声去噪验证损失选择，指标可能与该测试集择优结果不同。UCR 历史最佳 epoch 的重构训练损失约为首轮的 `17%`，当前默认的 `20%` 水位用训练损失作停止信号，不读取测试标签。

每轮 `loss_curve.csv` 包含 `epoch,train_loss,val_loss,is_best`。`train_loss` 是原版 TranAD 两阶段训练损失（M3/M4 只改变推理残差处理，不额外改变 M0 的重构目标）。

```text
checkpoints/
└── {dataset}/
    └── {M0|M1|M2|M3|M4}/
        └── seed_{seed}/
            ├── model_best.pt
            ├── loss_curve.csv
            └── score_reference_epoch5.json  # epoch 5 残差尺度（M3/M4 保存候选与回退两套）
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

MSDS 加载时会检查处理缓存；缓存标签为空时自动尝试重读 `data/MSDS/labels.csv` 并重建三个 `.npy` 文件。预处理会忽略索引/时间戳列、保留与特征数相同的逐通道标签，否则对标签列取逻辑 OR；下采样时按源区间 max-pool，避免漏掉短异常段。写入缓存前必须先确认源标签至少有一个正例。

当前仓库的 `data/MSDS/clean.py` 会生成全零占位 `labels.csv`，本地源标签没有任何正例。自愈会拒绝将占位文件写成监督标签，`run_ablation.sh` 会黄色警告并跳过 MSDS，不追加无效指标。要评估 MSDS 的分类指标，必须先放入可信异常标注。

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
