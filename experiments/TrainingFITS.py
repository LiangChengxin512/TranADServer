import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve
import time


# ==========================================
# 1. FITS 模型定义 (纯矩阵乘法版，完美适配 MPS)
# ==========================================
class FITS(nn.Module):
    def __init__(self, seq_len, enc_in, cut_freq):
        super(FITS, self).__init__()
        self.seq_len = seq_len
        self.cut_freq = cut_freq
        self.channels = enc_in

        # 预计算 DFT 与 iDFT 投影矩阵 (避免使用 torch.fft)
        n = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        k = torch.arange(cut_freq, dtype=torch.float32).unsqueeze(0)  # [1, K]
        theta = 2.0 * math.pi * n * k / seq_len  # [L, K]

        # 前向 DFT 变换矩阵: [L, K]
        dft_cos = torch.cos(theta)
        dft_sin = -torch.sin(theta)

        # 逆变换 iDFT 矩阵系数 (k=0 权重为 1/L, k>0 权重为 2/L)
        scale = torch.full((1, cut_freq), 2.0 / seq_len, dtype=torch.float32)
        scale[0, 0] = 1.0 / seq_len

        idft_cos = (scale * torch.cos(theta)).transpose(0, 1)  # [K, L]
        idft_sin = (scale * torch.sin(theta)).transpose(0, 1)  # [K, L]

        # 注册为 buffer，跟随模型移动到 MPS 设备且不参与梯度更新
        self.register_buffer('dft_cos', dft_cos)
        self.register_buffer('dft_sin', dft_sin)
        self.register_buffer('idft_cos', idft_cos)
        self.register_buffer('idft_sin', idft_sin)

        # 可学习频域复数权重 (实部与虚部)
        self.real_weight = nn.Parameter(torch.randn(enc_in, cut_freq, cut_freq) * 0.02)
        self.imag_weight = nn.Parameter(torch.randn(enc_in, cut_freq, cut_freq) * 0.02)

    def forward(self, x):
        # x shape: [B, L, D] -> [B, D, L]
        x = x.permute(0, 2, 1)

        # 1. 前向频域投影 (矩阵乘法替代 rfft)
        # [B, D, L] x [L, K] -> [B, D, K]
        real = torch.matmul(x, self.dft_cos)
        imag = torch.matmul(x, self.dft_sin)

        # 2. 复数线性变换 (FITS 核心滤波层)
        out_real = torch.einsum('bdc,dcf->bdf', real, self.real_weight) - \
                   torch.einsum('bdc,dcf->bdf', imag, self.imag_weight)
        out_imag = torch.einsum('bdc,dcf->bdf', real, self.imag_weight) + \
                   torch.einsum('bdc,dcf->bdf', imag, self.real_weight)

        # 3. 逆频域投影 (矩阵乘法替代 irfft)
        # [B, D, K] x [K, L] -> [B, D, L]
        x_rec = torch.matmul(out_real, self.idft_cos) - torch.matmul(out_imag, self.idft_sin)

        # [B, D, L] -> [B, L, D]
        return x_rec.permute(0, 2, 1)


# ==========================================
# 2. 数据加载与多实体兼容 (MBA/Synthetic/SMAP)
# ==========================================
def create_windows(data, window_size):
    """创建滑动窗口"""
    if len(data) < window_size:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32)
    windows = [data[i:i + window_size] for i in range(len(data) - window_size + 1)]
    return np.array(windows, dtype=np.float32)


def load_dataset_windows(data_dir, seq_len):
    """统一加载单文件或多实体数据集 (已修复多维标签问题)"""
    train_windows_list, test_windows_list, test_labels_list = [], [], []

    # 场景 A: 单文件数据集 (Synthetic, MBA)
    if os.path.exists(os.path.join(data_dir, 'train.npy')):
        print("   -> Detected Single-file Structure")
        train_data = np.load(os.path.join(data_dir, 'train.npy'))
        test_data = np.load(os.path.join(data_dir, 'test.npy'))
        labels = np.load(os.path.join(data_dir, 'labels.npy'))

        # 【核心修复】：如果标签是 (N, D) 的多维矩阵，按时间步取 max 压缩成 (N,)
        if labels.ndim > 1:
            labels = np.max(labels, axis=1)

        train_windows_list.append(create_windows(train_data, seq_len))
        test_windows_list.append(create_windows(test_data, seq_len))
        test_labels_list.append(labels[seq_len - 1:])
        num_features = train_data.shape[1]

    # 场景 B: 多文件数据集 (SMAP, MSL)
    else:
        print("   -> Detected Multi-entity Structure (SMAP)")
        prefixes = sorted(list(set([f.replace('_train.npy', '')
                                    for f in os.listdir(data_dir) if f.endswith('_train.npy')])))
        if not prefixes:
            raise ValueError(f"No valid data found in {data_dir}")

        for prefix in prefixes:
            train_data = np.load(os.path.join(data_dir, f"{prefix}_train.npy"))
            test_data = np.load(os.path.join(data_dir, f"{prefix}_test.npy"))
            labels = np.load(os.path.join(data_dir, f"{prefix}_labels.npy"))

            # 【核心修复】：防止多维标签导致错位
            if labels.ndim > 1:
                labels = np.max(labels, axis=1)

            if len(train_data) >= seq_len:
                train_windows_list.append(create_windows(train_data, seq_len))
            if len(test_data) >= seq_len:
                test_windows_list.append(create_windows(test_data, seq_len))
                test_labels_list.append(labels[seq_len - 1:])

        num_features = train_data.shape[1]

    train_windows = np.concatenate(train_windows_list, axis=0)
    test_windows = np.concatenate(test_windows_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)

    return train_windows, test_windows, test_labels, num_features


def get_metrics(scores, labels):
    """计算最佳 F1 和 AUC"""
    # 核心修复：强制展平为纯 1D 数组，并统一类型，防止 sklearn 误判为多标签任务
    scores = np.array(scores).flatten()
    labels = np.array(labels).flatten().astype(int)

    # 计算 AUC (增加异常捕捉，防止某一切片中全为 0/正常数据)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.0  # 如果标签仅包含单个类别（如全0），AUC 无定义，置为0

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    # 避免除以零
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)

    return precisions[best_f1_idx], recalls[best_f1_idx], f1_scores[best_f1_idx], auc


# ==========================================
# 3. 核心训练与测试
# ==========================================
def run_experiment():
    epochs = 10
    batch_size = 128
    seq_len = 100
    cut_freq = 25
    datasets = ['Synthetic', 'MBA', 'SMAP']

    # 优先使用 Apple Silicon GPU (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Device: Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using Device: CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ Using Device: CPU")

    for dataset_name in datasets:
        print("\n" + "=" * 50)
        print(f"📊 Dataset: {dataset_name}")
        print("=" * 50)

        data_dir = f'../processed/{dataset_name}'
        if not os.path.exists(data_dir):
            data_dir = f'data/{dataset_name}'
            if not os.path.exists(data_dir):
                print(f"❌ Cannot find directory: {data_dir}")
                continue

        print("⏳ Loading and preprocessing data...")
        try:
            train_windows, test_windows, test_labels, num_features = load_dataset_windows(data_dir, seq_len)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        print(f"   Train samples: {len(train_windows)} | Test samples: {len(test_windows)} | Features: {num_features}")

        train_tensor = torch.tensor(train_windows, dtype=torch.float32)
        test_tensor = torch.tensor(test_windows, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)

        model = FITS(seq_len=seq_len, enc_in=num_features, cut_freq=cut_freq).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        print(f"\n🚀 Training ({epochs} Epochs)...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            start_time = time.time()

            for batch_x in train_loader:
                batch_x = batch_x[0].to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            epoch_time = time.time() - start_time
            print(
                f"   Epoch [{epoch + 1}/{epochs}] | Loss: {train_loss / len(train_loader):.6f} | Time: {epoch_time:.2f}s")

        print("\n🔍 Evaluating...")
        model.eval()
        scores = []

        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x[0].to(device)
                output = model(batch_x)
                error = torch.mean((batch_x[:, -1, :] - output[:, -1, :]) ** 2, dim=-1)
                scores.extend(error.cpu().numpy())

        scores = np.array(scores)
        p, r, f1, auc = get_metrics(scores, test_labels)

        print(f"\n✅ Results for {dataset_name}:")
        print(f"   Precision : {p:.4f}")
        print(f"   Recall    : {r:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   AUC       : {auc:.4f}")


if __name__ == "__main__":
    run_experiment()