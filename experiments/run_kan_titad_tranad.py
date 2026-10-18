import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==========================================
# 0. 全局配置参数
# ==========================================
# 定义模型保存的根目录，你可以随时修改
SAVE_DIR = "./saved_checkpoints_XLinear_TiTAD"


# ==========================================
# 1. 模型库定义
# ==========================================

class SimpleKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3):
        super(SimpleKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, degree + 1) * 0.01)

    def forward(self, x):
        B = x.size(0)
        bases = torch.stack([x ** d for d in range(self.degree + 1)], dim=-1)
        out = torch.einsum('bid,jid->bj', bases, self.coeffs)
        return out


class KAN_AD(nn.Module):
    def __init__(self, seq_len, enc_in, hidden_dim=64):
        super(KAN_AD, self).__init__()
        self.seq_len = seq_len
        self.channels = enc_in
        self.input_dim = seq_len * enc_in
        self.encoder = SimpleKANLayer(self.input_dim, hidden_dim)
        self.decoder = SimpleKANLayer(hidden_dim, self.input_dim)

    def forward(self, x):
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        hidden = self.encoder(x_flat)
        x_rec_flat = self.decoder(hidden)
        return x_rec_flat.reshape(B, self.seq_len, self.channels)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TiTAD(nn.Module):
    def __init__(self, seq_len, enc_in, d_model=64, n_heads=4, e_layers=2):
        super(TiTAD, self).__init__()
        self.seq_len = seq_len
        self.channels = enc_in
        self.embedding = nn.Linear(enc_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.reconstructor = nn.Linear(d_model, enc_in)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        enc_out = self.transformer_encoder(x_emb)
        x_rec = self.reconstructor(enc_out)
        return x_rec


# ==========================================
# 2. 数据处理与评估
# ==========================================
def create_windows(data, window_size):
    if len(data) < window_size:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32)
    windows = [data[i:i + window_size] for i in range(len(data) - window_size + 1)]
    return np.array(windows, dtype=np.float32)


def safe_load_npy(path):
    data = np.load(path)
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    return data


def load_dataset_windows(data_dir, seq_len):
    train_windows_list, test_windows_list, test_labels_list = [], [], []

    if os.path.exists(os.path.join(data_dir, 'train.npy')):
        train_data = safe_load_npy(os.path.join(data_dir, 'train.npy'))
        test_data = safe_load_npy(os.path.join(data_dir, 'test.npy'))
        labels = safe_load_npy(os.path.join(data_dir, 'labels.npy'))

        if labels.ndim > 1: labels = np.max(labels, axis=1)

        train_windows_list.append(create_windows(train_data, seq_len))
        test_windows_list.append(create_windows(test_data, seq_len))
        test_labels_list.append(labels[seq_len - 1:])
        num_features = train_data.shape[1]
    else:
        prefixes = sorted(
            list(set([f.replace('_train.npy', '') for f in os.listdir(data_dir) if f.endswith('_train.npy')])))
        if not prefixes: raise ValueError(f"No valid data found in {data_dir}")

        for prefix in prefixes:
            train_data = safe_load_npy(os.path.join(data_dir, f"{prefix}_train.npy"))
            test_data = safe_load_npy(os.path.join(data_dir, f"{prefix}_test.npy"))
            labels = safe_load_npy(os.path.join(data_dir, f"{prefix}_labels.npy"))

            if labels.ndim > 1: labels = np.max(labels, axis=1)

            if len(train_data) >= seq_len: train_windows_list.append(create_windows(train_data, seq_len))
            if len(test_data) >= seq_len:
                test_windows_list.append(create_windows(test_data, seq_len))
                test_labels_list.append(labels[seq_len - 1:])

        num_features = train_data.shape[1]

    train_windows = np.concatenate(train_windows_list, axis=0)
    test_windows = np.concatenate(test_windows_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)

    return train_windows, test_windows, test_labels, num_features


def get_metrics(scores, labels):
    scores = np.array(scores).flatten()
    labels = np.array(labels).flatten().astype(int)

    if np.isnan(scores).any() or np.isinf(scores).any():
        scores = np.nan_to_num(scores, nan=0.0,
                               posinf=np.nanmax(scores[scores != np.inf]) if np.any(scores != np.inf) else 1.0)

    if np.all(scores == scores[0]):
        return 0.0, 0.0, 0.0, 0.5

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)

    return precisions[best_f1_idx], recalls[best_f1_idx], f1_scores[best_f1_idx], auc


def sync_device(device):
    """设备同步锁，用于精确计算 Latency"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        try:
            torch.mps.synchronize()
        except:
            pass


# ==========================================
# 3. 核心运行流
# ==========================================
def run_experiment():
    epochs = 100
    batch_size = 128
    seq_len = 100

    os.makedirs(SAVE_DIR, exist_ok=True)
    datasets = ['Synthetic', 'MBA', 'SMAP']
    # models_to_run = ['KAN_AD', 'TiTAD']
    models_to_run = ['TiTAD']

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🚀 Using Device: {device}\n" + "-" * 50)

    for model_name in models_to_run:
        print(f"\n==================================================")
        print(f"🌟 Starting Model: {model_name}")
        print(f"==================================================")

        for dataset_name in datasets:
            print(f"\n---> Dataset: {dataset_name}")
            data_dir = f'../processed/{dataset_name}'
            if not os.path.exists(data_dir):
                data_dir = f'data/{dataset_name}'
                if not os.path.exists(data_dir): continue

            try:
                train_windows, test_windows, test_labels, num_features = load_dataset_windows(data_dir, seq_len)
            except Exception as e:
                continue

            # --- 划分 90% 训练集, 10% 验证集以筛选最优模型 ---
            full_train_tensor = torch.tensor(train_windows, dtype=torch.float32)
            val_size = max(1, int(len(full_train_tensor) * 0.1))
            train_size = len(full_train_tensor) - val_size

            train_ds, val_ds = random_split(TensorDataset(full_train_tensor), [train_size, val_size])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(torch.tensor(test_windows, dtype=torch.float32)),
                                     batch_size=batch_size, shuffle=False)

            # 初始化模型
            if model_name == 'KAN_AD':
                model = KAN_AD(seq_len=seq_len, enc_in=num_features).to(device)
            elif model_name == 'TiTAD':
                model = TiTAD(seq_len=seq_len, enc_in=num_features).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            model_save_path = os.path.join(SAVE_DIR, f"{model_name}_{dataset_name}_best.pth")

            print("   [Training & Checkpointing]")
            train_start_time = time.time()
            for epoch in range(epochs):
                model.train()
                for batch_x in train_loader:
                    batch_x = batch_x[0].to(device)
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_x)
                    loss.backward()
                    # 梯度裁剪非常关键，防 KAN/TiTAD 梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # --- 验证阶段 ---
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x in val_loader:
                        batch_x = batch_x[0].to(device)
                        output = model(batch_x)
                        val_loss += criterion(output, batch_x).item()
                val_loss /= len(val_loader)

                # 保存最优模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)

            total_train_time = time.time() - train_start_time
            print(f"   ✓ Best model saved to: {model_save_path}")

            # ==========================================
            # 加载最优模型进行测试与性能基准评估
            # ==========================================
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
            scores = []
            latencies_ms = []  # 保存每个 batch 的延迟(ms)
            throughputs = []  # 保存每个 batch 的吞吐量(samples/s)

            print("   [Testing & Profiling]")
            with torch.no_grad():
                for i, batch_x in enumerate(test_loader):
                    batch_x = batch_x[0].to(device)
                    current_batch_size = batch_x.size(0)

                    # 硬件同步开始计时
                    sync_device(device)
                    start_t = time.perf_counter()

                    output = model(batch_x)

                    # 硬件同步结束计时
                    sync_device(device)
                    end_t = time.perf_counter()

                    # 计算耗时 (略过前 5 个 batch 的预热期)
                    if i > 5:
                        latency_seconds = end_t - start_t
                        latencies_ms.append(latency_seconds * 1000.0)
                        throughputs.append(current_batch_size / latency_seconds)

                    error = torch.mean((batch_x[:, -1, :] - output[:, -1, :]) ** 2, dim=-1)
                    scores.extend(error.cpu().numpy())

            # 聚合测试结果
            p, r, f1, auc = get_metrics(scores, test_labels)

            # 计算平均值与标准差 (± Mean格式)
            avg_latency = np.mean(latencies_ms) if latencies_ms else 0.0
            std_latency = np.std(latencies_ms) if latencies_ms else 0.0
            avg_throughput = np.mean(throughputs) if throughputs else 0.0
            std_throughput = np.std(throughputs) if throughputs else 0.0

            print(f"\n   ✅ {model_name} on {dataset_name} (Epochs: {epochs}):")
            print(f"      ⏱️ Training Time: {total_train_time:.2f} s")
            print(f"      ⚡ Latency      : {avg_latency:.2f} ± {std_latency:.2f} ms / batch")
            print(f"      🚀 Throughput   : {avg_throughput:.2f} ± {std_throughput:.2f} samples / s")
            print(f"      🎯 Accuracy     : P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n")


if __name__ == "__main__":
    run_experiment()