# TranAD M0–M4 消融实验

本实现将原有 TranAD 与仓库已存在的 `C/G/E` 变体组合成五级消融。模型扩展和无标签评分规则位于 `src/tranad_progressive.py`；原始模型定义仍在 `src/models.py`，只在 M0–M4 工厂入口挂载。测试标签只用于最后的标准 point-adjust 评估；自适应路由、Top-K 保护和 M4 分数处理不读取标签。

## 变体与数据集路由

| 变体 | 模型与评分 | 路由规则 |
| --- | --- | --- |
| M0 | 原版 `TranAD`，全通道均值残差 | 所有数据集相同。 |
| M1 | 复用 `TranAD_E` 或 `TranAD_C_E`，全通道均值残差 | D=1、UCR、NAB、SWaT、SMD 使用 E；其他多变量数据使用 C_E。 |
| M2 | 复用 `TranAD_G_E`、`TranAD_C_G` 或 `TranAD_C_G_E`，均值残差 | D=1 或 UCR 使用 G_E；synthetic 使用 C_G；其他多变量数据使用 C_G_E。G 的通道门控只作用于训练集判定为连续活跃的通道。 |
| M3 | M2 架构加仓库原生逐时间步 adaptive-topk | 以训练分布的尾部信噪代理量作保护；若 adaptive-topk 低于均值基线则回退均值。 |
| M4 | M3 加保守背景毛刺过滤及 SWaT 平台唤醒 | 仅过滤满足单步、幅度、邻域和背景中位数条件的微弱孤立越阈点；仅 SWaT 可触发持续亚阈值平台唤醒。阈值先由训练分数锁定，再处理测试分数。 |

M4 的背景过滤要求候选是单点越阈、分数/阈值比严格处于 `(1, 1.12)`、前后 25 点内没有其他越阈点，并且排除候选后的前后 10 点局部中位分数低于训练分数中位数。SWaT 唤醒只检查连续至少 25 点、分数位于 `[0.70×threshold, threshold]` 的片段，并要求测试越阈背景干净且片段具备低差分方差或多通道共现；符合条件的片段乘以 1.30，并保证落到锁定阈值以上。它不会触碰 MSL、SMD、SMAP、MBA、NAB 的分数。

这些是无标签的评分规则，不可能在不读取测试标签的前提下保证每个数据集的 F1 单调上升，也不能数学保证 M4 的实测召回不下降。M4 对 SWaT 的规则是启发式唤醒，是否提升需以 GPU 节点上的真实评估结果确认。附件所述 `result_5.csv` 与历史 `cge_adaptive_k_metrics_long.csv` 不在此工作区，因此这里没有据它们核验或复现指标数值。

## 训练、早停和检查点

`run_ablation.sh` 默认传入 `--epochs 100`。训练器最多运行 `min(--epochs, 8)` 轮，并监控带对抗发散惩罚的验证准则：

```text
J_val(n) = val_loss(n) × (1 + 0.15 × max(0, n - 5))
```

每轮验证使用固定噪声，并在 `torch.random.fork_rng()` 中执行，同时保存并恢复 NumPy 和 Python RNG 状态。这样验证不会推进随后的训练随机流。训练使用原版 AdamW、StepLR（step=5，gamma=0.9）和 TranAD 两阶段损失。传入至少 5 个 epoch 时，`model_epoch5.pt` 与 `model_best.pt` 都会锁定为第 5 轮完整权重；第 6–8 轮用于早停观察和写入曲线，不会替换推理检查点。若传入少于 5 轮，则 `model_best.pt` 取已训练轮次中验证损失最低的权重。

早停默认 `patience=3`、`rel_delta=0.005`，从第 5 轮锚点继续比较惩罚验证损失；若没有提前触发，则 8 轮上限停止训练。实际验证曲线取决于数据，不保证每次都恰好运行到第 8 轮。完整数据集训练应在 Linux GPU 节点执行，本地只需做语法和小张量检查。

## 文件清单

- `src/tranad_progressive.py`：M0–M4 动态工厂路由、原生 adaptive-topk 保护、M4 背景过滤和 SWaT 平台唤醒。
- `src/tranad_variants.py`：复用 C/G/E 变体；G 门控支持训练集连续活跃通道掩码。
- `src/models.py`：保留原 TranAD，实现 M0–M4 的最小动态工厂挂载。
- `src/experiment_utils.py`：模型构造、标签/形状验证、原始重构分数、M3/M4 评估与阈值锁定。
- `experiments/run_progressive_ablation.py`：训练、隔离验证 RNG、早停、epoch 5 检查点、Loss CSV 和指标落盘。
- `run_ablation.sh`：数据可用性与正例检查、Seed/模型循环、单次运行和结束后触发汇总。
- `aggregate_metrics.py`：剔除缺失数据、无正例标签或非有限指标后，按 dataset/model 生成均值和样本标准差。
- `preprocess.py`、`data/MSDS/clean.py`：MSDS 标签读取、对齐及无有效正例时的保护。
- `ABLATION_GUIDE.md`：算法、输出目录和运行说明。

缺少必需预处理文件的数据集会显示黄色 `[WARN]` 并跳过。加载时发现无正例标签或标签维度不匹配，也会跳过且不向 `ablation_raw_metrics.csv` 追加行。汇总器再次校验预处理文件、正例标签和有限指标，所以不会在汇总表中输出 `--` 或 `NaN` 单元格。`result.csv` 不作为此流程的输入或输出。

## 输出目录

```text
checkpoints/{dataset}/{M0|M1|M2|M3|M4}/seed_{seed}/
├── model_best.pt       # 最终测试推理权重；epochs >= 5 时锁定 epoch 5
├── model_epoch5.pt     # epoch 5 完整检查点
└── loss_curve.csv      # epoch, train_loss, val_loss, selection loss, is_best

experiment_logs/{dataset}/{variant}/seed_{seed}/
├── train_metrics.csv
├── test_metrics.json
└── test_per_dim.csv

experiments_results/
├── ablation_raw_metrics.csv
└── ablation_summary_mean_std.csv
```

## 运行实验

脚本默认 Seed 1、数据集 `synthetic SMD SWaT SMAP MSL WADI MSDS UCR MBA NAB`、模型 M0–M4、最多 100 epoch。正式五种子实验可设置 `SEEDS="1 2 3 4 5"`：

```sh
sh run_ablation.sh
```

也可限制数据集、模型和设备：

```sh
DEVICE=cpu EPOCHS=100 SEEDS="1" DATASETS="UCR" MODELS="M0 M1" sh run_ablation.sh
```

汇总可单独运行：

```sh
python3 aggregate_metrics.py
```

## 从检查点重新测试

Linux/CUDA：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device cuda --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```

macOS/MPS（若该 PyTorch 环境不支持 MPS，可将 `--device mps` 改为 `cpu` 或 `auto`）：

```sh
python3 experiments/run_progressive_ablation.py --mode test --variant M4 --dataset SMD --seed 1 --device mps --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```
