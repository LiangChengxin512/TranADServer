# TranAD M0–M4 消融实验

扩展实现集中在 `src/tranad_progressive.py`，原 TranAD 主体只通过 `src/models.py` 中的 M0–M4 工厂入口挂载扩展。M0 保留两阶段重构训练目标。指标只在评估阶段读取标签；M2 的通道回退和 M4 的 Top-K 选择均不读取测试标签。

## 变体与评分逻辑

| 变体 | 逻辑 |
| --- | --- |
| M0 | 原版 TranAD，两阶段重构损失和全通道均值残差。 |
| M1 | M0 加 kernel 1、3、5 并联的左侧因果深度可分离卷积。 |
| M2 | M1 加正常原型记忆和 hollow 跨通道预测器。训练数据中每通道只有 `nunique > 10` 且 `std > 1e-3` 才参与通道消息传递及原型记忆残差；邻接权重对角线恒为零，D=1 时跨通道分支短路。M2 及以上会按训练尾部逐通道比较 M1；回退门控只用最后 20% 训练窗口拟合，若 M2 更差则该通道训练、测试残差回退到 M1。 |
| M3 | M2 的残差评分。先用未处理的训练均值残差估计 POT 阈值并锁定；随后只对测试分数中持续不超过 2 点、峰值/阈值比小于 2.2 的短弱事件乘以 `0.45`。当特征数至少 20 时，因果 50 点均值高于训练滚动均值 95 分位且连续至少 30 点的测试区间乘以 `1.35`。阈值不会因测试分数处理而重新拟合。 |
| M4 | 在 `k ∈ {1,2,3,4,5,8,12,16,32,64,...,D}` 中用无标签事件轮廓比较 Top-K 均值；保留稳健事件的同时改善稳健事件突出度/SNR 或减少短弱事件才采用候选，否则回退到 `k=D`。每个候选都先用训练残差拟合 POT 阈值，再执行 M3 的短弱事件抑制与持续段增强。 |

这些无监督规则无法保证每个数据集的真实测试 F1 严格满足 M0≤M1≤…≤M4。若用测试标签强制排序或挑选模型，会造成测试集泄漏；结果应如实报告，不应据此调整测试分数。

## 改动文件

- `src/tranad_progressive.py`：M1 因果卷积、M2 hollow 通道预测与 M1 训练尾部保真门控、M3 事件处理、M4 无标签 Top-K 事件选择。
- `src/models.py`：通过工厂注册渐进变体，保留 M0 原模型路径。
- `src/experiment_utils.py`：原始残差推理、锁定阈值的 M3/M4 评估、MSDS/WADI 标签可用性检查。
- `experiments/run_progressive_ablation.py`：逐通道连续掩码、M1 残差参考、Epoch 5 权重快照和训练分数 q98 回退。
- `run_ablation.sh`：POSIX shell 数据检查、无标签数据跳过、Seed/模型遍历和汇总触发。
- `preprocess.py` 与 `data/MSDS/clean.py`：MSDS 标签读取、自愈及空标签保护。
- `aggregate_metrics.py`：按数据集和模型汇总均值与样本标准差。
- `ABLATION_GUIDE.md`：算法和运行说明。

## 检查点、日志与指标

当 `model_best.pt` 的训练均值残差 q98 低于 Epoch 5 检查点的 85% 时，测试推理改用 `model_epoch5.pt`。这个规则选择完整权重，不对 POT 的训练分数单独缩放，也不修改测试残差。指标记录同时保留验证最佳 epoch、实际推理 epoch 和回退诊断值。

```text
checkpoints/
└── {dataset}/{M0|M1|M2|M3|M4}/seed_{seed}/
    ├── model_best.pt       # 验证重构损失最低的权重
    ├── model_epoch5.pt     # Epoch 5 完整权重，用于训练分数收缩回退
    └── loss_curve.csv      # epoch, train_loss, val_loss, is_best
experiment_logs/{dataset}/{variant}/seed_{seed}/
├── train_metrics.csv
├── test_metrics.json       # 包含推理 checkpoint 与 epoch5 回退诊断
└── test_per_dim.csv
experiments_results/
├── ablation_raw_metrics.csv
└── ablation_summary_mean_std.csv
```

`run_ablation.sh` 默认 Seed 1、最多 100 Epoch，数据集为 `synthetic SMD SWaT SMAP MSL WADI MSDS UCR MBA NAB`，变体为 M0–M4。正式五种子实验可设置 `SEEDS="1 2 3 4 5"`。缺少所需数据、或 MSDS/WADI 没有正例标签时打印黄色警告并跳过，不会为这些数据集追加指标行。

```sh
sh run_ablation.sh
```

例如限制在 UCR、CPU 和少量 epoch：

```sh
DEVICE=cpu EPOCHS=8 MIN_EPOCHS=5 SEEDS="1" DATASETS="UCR" MODELS="M0 M1" sh run_ablation.sh
```

汇总也可单独执行：

```sh
python3 aggregate_metrics.py
```

## 从检查点重新测试

Linux/CUDA：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device cuda --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```

macOS/MPS（不支持时可将设备改为 `cpu` 或 `auto`）：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device mps --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```
