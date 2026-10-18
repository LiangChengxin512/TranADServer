# 任务目标：基于 TranAD 的渐进式消融实验（M0 到 M4）算法改进与全流程自动化

## 1. 背景与核心问题诊断

当前工程中基于 TranAD 堆叠的组件在单变量时序（如 UCR）上因图结构退化出现了性能崩塌，且组件间简单叠加引发了负迁移。为了在毕业论文中展示严谨的学术动机与稳定提升（1%~5%），现需将架构重构成模块化、可插拔且具备自适应防护的渐进式消融体系（$M_0 \to M_4$）。

## 2. 代码修改原则（强制遵守）

1. **增量式开发（Non-invasive & Modular）**：
   - 原有核心逻辑尽量不破坏。新增的模块、损失函数、聚合算法均放置在独立的扩展文件（如 `src/models/extensions.py` 或 `src/modules/`）中。
   - 通过统一的模型入口参数（例如 `--variant M0/M1/M2/M3/M4` 或分立布尔开关）完成无缝切换，保证完全向后兼容。
2. **多环境自动适配（CUDA / MPS / CPU）**：
   - 自动检测计算平台：优先使用 Nvidia GPU（CUDA），次选 Apple Silicon GPU（MPS），无可用 GPU 时回退到 CPU。
   - 保证所有张量运算（尤其是 FFT 与高级索引）兼容 PyTorch MPS 后端。

---

## 3. 算法改进与消融模型设计（M0 到 M4）

### $M_0$：基准模型（TranAD Baseline）

- 保持现有未改动的 TranAD 架构与标准均值聚合打分方式。

### $M_1$：$M_0$ + 多尺度因果卷积局部嵌入（Multi-Scale Causal Conv Embedding）

- **动机**：标准自注意力对局部细粒度波形和瞬态毛刺不敏感。
- **算法设计**：
  - 在输入 Embedding 层并联一组因果深度可分离卷积（Causal 1D Conv），卷积核大小设为 $k \in \{1, 3, 5\}$，严格使用因果 Padding 防止未来信息泄漏。
  - 引入 LayerScale 机制（初始化为 $1\times 10^{-4}$ 的可学习权重向量），将多尺度特征与原输入表征通过残差融合：  
    $X_{\text{emb}} = \text{LayerNorm}(W_{\text{proj}} X + \text{LayerScale}(\text{Concat}(\text{Conv}_1(X), \text{Conv}_3(X), \text{Conv}_5(X))))$

### $M_2$：$M_1$ + 自适应门控图注意力机制（Adaptive Gated Graph Attention）

- **动机**：显式捕获多通道关联，同时彻底解决单变量（如 UCR，变量数 $D=1$）或弱关联通道上的图退化/训练崩塌问题。
- **算法设计**：
  - 设计拓扑自适应门控因子 $\gamma \in [0, 1]$：  
    $\gamma = \text{Sigmoid}(\text{MLP}(\text{GlobalAvgPool}(X)))$
  - 当输入变量维度 $D=1$ 时，代码显式短路该模块（强制 $\gamma = 0$）；当 $D > 1$ 时，计算可学习节点邻接矩阵的图注意力（GAT / GraphConv）。
  - 输出与 $M_1$ 特征做门控残差连接：  
    $H_{\text{out}} = (1 - \gamma) \cdot H_{\text{in}} + \gamma \cdot \text{GAT}(H_{\text{in}})$

### $M_3$：$M_2$ + 频域一致性损失与焦点重构约束（FFT Loss & Focal-MSE）

- **动机**：原版 L2 损失容易平滑突发尖峰，且易分正常样本主导了梯度更新。
- **算法设计**：
  - **频域一致性损失（FFT Loss）**：对真实序列 $X$ 与重构序列 $\hat{X}$ 计算沿时间维度的实数快速傅里叶变换（`torch.fft.rfft`），约束频域振幅谱误差：  
    $\mathcal{L}_{\text{freq}} = \frac{1}{T} \sum \Vert{}\text{Abs}(\text{rfft}(X)) - \text{Abs}(\text{rfft}(\hat{X}))\Vert{}_1$
  - **动态 Focal 重构损失**：给误差偏大的困难样本自适应权重：  
    $\mathcal{L}_{\text{focal}} = (1 + \tanh(\vert{}X - \hat{X}\vert{})) \odot \vert{}X - \hat{X}\vert{}^2$
  - **联合损失**：在原 TranAD 的两阶段损失基础上加权结合：$\mathcal{L} = \mathcal{L}_{\text{TranAD}} + \alpha \mathcal{L}_{\text{freq}} + \beta \mathcal{L}_{\text{focal}}$（建议默认 $\alpha=0.1, \beta=0.1$）。

### $M_4$：$M_3$ + 动态自适应 Top-K 打分聚合（Adaptive Top-K Score Aggregation）

- **动机**：多变量系统中异常表现具有稀疏性，全通道平均打分会稀释异常。
- **算法设计**：
  - 在推理打分阶段，根据残差方差或动态排序策略，动态确定敏感通道数量 $k$（或沿用工程中已验证有效的自适应 Top-K 模块）。
  - 仅选取各时间步排名前 $k$ 的残差绝对值计算平均，生成最终时序异常分数。

---

## 4. 设备检测代码规范

在核心入口（如 `main.py` 或 `src/utils/device.py`）中实现统一的设备获取函数：

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # 兼容性测试，防止个别旧版 macOS 报错
            _ = torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    else:
        return torch.device("cpu")
```

## 5. 数据集完整性检测与跳过逻辑

实验目标覆盖以下 10 个数据集： `DATASETS=('synthetic' 'SMD' 'SWaT' 'SMAP' 'MSL' 'WADI' 'MSDS' 'UCR' 'MBA' 'NAB')`

- 在 Python 数据读取入口或 Bash 运行前增加前置检测。
- 若检测到对应数据集的原始文件或预处理 pickle 文件不存在，**终端打印清晰黄色警告（Warning: Dataset [NAME] not found, skipping...）并直接优雅跳过**，严禁阻断后续实验的正常运行。

## 6. 自动化消融运行脚本：`run_ablation.sh`

编写符合 POSIX 标准的 Bash 脚本，支持在 Linux 与 macOS 上一键执行：

1. **实验循环**：
   - 依次遍历 5 个不同种子：`SEEDS=(1 2 3 4 5)`。
   - 依次遍历所有存在的数据集。
   - 依次测试 5 种变体：`MODELS=(M0 M1 M2 M3 M4)`。
2. **模型权重保存规范**：
   - 训练产物规范化存储：`checkpoints/{dataset}/{model}/seed_{seed}/model.pt`。
3. **指标记录**：
   - 每次评测完成后，将单次运行的详细指标（`seed`, `dataset`, `model`, `precision`, `recall`, `f1`, `roc_auc` 等）追加写入 `experiments_results/ablation_raw_metrics.csv`。

## 7. 统计汇总脚本：`aggregate_metrics.py`

编写独立的数据汇总脚本：

1. 读取 `experiments_results/ablation_raw_metrics.csv`。
2. 按 `dataset` 和 `model` 进行分组，计算 5 个种子测试结果的均值（Mean）与样本标准差（Standard Deviation）。
3. 导出论文专用的格式表 `experiments_results/ablation_summary_mean_std.csv`，单元格文本格式化为： `Mean ± Std`（例如：`0.9234 ± 0.0045`），涵盖 Precision、Recall、F1、ROC-AUC 等指标。

## 8. 文档生成：`ABLATION_GUIDE.md`

请在仓库根目录生成或更新 `ABLATION_GUIDE.md`，内容包含：

1. **代码改动清单**：清楚记录新增了哪些文件、原有文件的哪些地方增加了钩子函数（Hook）。
2. **模型产物存放结构树**：清晰展现 Checkpoint 和日志目录结构。
3. **单步运行与独立复现指南**：
   - 给出在 Linux（CUDA）环境直接调用某一个指定 Checkpoint 进行推理的命令行示例。
   - 给出在 macOS（MPS/CPU）环境下验证单数据集、单模型的快速运行命令行示例。
