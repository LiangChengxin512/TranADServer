from pathlib import Path
import os
import csv

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tranad_matplotlib")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


DATA = {
    "SMD": {
        "TranAD_E": {"f1": 0.9133648819747217, "precision": 0.9951370424618167, "recall": 0.8440194943981271},
        "TranAD": {"f1": 0.9981188236746776, "precision": 0.9988734472441853, "recall": 0.9973753243442994},
    },
    "MSL": {
        "TranAD_E": {"f1": 0.9494589959667135, "precision": 0.9037900611139924, "recall": 0.9999999677419366},
        "TranAD": {"f1": 0.9494589959667135, "precision": 0.9037900611139924, "recall": 0.9999999677419366},
    },
    "Synthetic": {
        "TranAD_E": {"f1": 0.9836015385903053, "precision": 0.9677419159729453, "recall": 0.9999999791666672},
        "TranAD": {"f1": 0.9756047392681091, "precision": 0.9523809334845054, "recall": 0.9999999791666672},
    },
    "SWaT": {
        "TranAD_E": {"f1": 0.81508255249909, "precision": 0.9999999774266372, "recall": 0.6878881880762704},
        "TranAD": {"f1": 0.81508255249909, "precision": 0.9999999774266372, "recall": 0.6878881880762704},
    },
    "NAB": {
        "TranAD_E": {"f1": 0.9411711188269714, "precision": 0.8888885596709039, "recall": 0.9999995833335069},
        "TranAD": {"f1": 0.9411711188269714, "precision": 0.8888885596709039, "recall": 0.9999995833335069},
    },
    "MBA": {
        "TranAD_E": {"f1": 0.9792793675976177, "precision": 0.9594095905556841, "recall": 0.999999996153846},
        "TranAD": {"f1": 0.9789106612131879, "precision": 0.958702061361718, "recall": 0.999999996153846},
    },
}


def style_axes(ax):
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_f1_comparison():
    datasets = list(DATA.keys())
    tranad = np.array([DATA[d]["TranAD"]["f1"] for d in datasets])
    enhanced = np.array([DATA[d]["TranAD_E"]["f1"] for d in datasets])
    x = np.arange(len(datasets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.bar(x - width / 2, tranad, width, label="TranAD", color="#4C78A8")
    ax.bar(x + width / 2, enhanced, width, label="TranAD-Ensemble + top-k", color="#F58518")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.75, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f1_comparison.pdf")
    plt.close(fig)


def save_metric_deltas():
    datasets = list(DATA.keys())
    metrics = ["f1", "precision", "recall"]
    colors = ["#54A24B", "#B279A2", "#E45756"]
    x = np.arange(len(datasets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for idx, metric in enumerate(metrics):
        delta = np.array([DATA[d]["TranAD_E"][metric] - DATA[d]["TranAD"][metric] for d in datasets])
        ax.bar(x + (idx - 1) * width, delta, width, label=metric.capitalize(), color=colors[idx])
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_ylabel("TranAD-Ensemble - TranAD")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metric_deltas.pdf")
    plt.close(fig)


def box(ax, xy, text, width=1.6, height=0.55, color="#E8EEF7"):
    rect = plt.Rectangle(xy, width, height, facecolor=color, edgecolor="#2F3A4A", linewidth=1.1)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9)
    return rect


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.1, color="#2F3A4A"))


def save_architecture_tranad():
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    box(ax, (0.2, 1.35), "Input window\nW x F")
    box(ax, (2.0, 1.35), "Self-conditioning\nc = 0")
    box(ax, (3.8, 1.35), "Encoder\nTransformer")
    box(ax, (5.6, 2.1), "Decoder 1\nx1")
    box(ax, (5.6, 0.6), "Decoder 2\nx2")
    box(ax, (7.5, 1.35), "MSE anomaly\nscore")

    arrow(ax, (1.8, 1.62), (2.0, 1.62))
    arrow(ax, (3.6, 1.62), (3.8, 1.62))
    arrow(ax, (5.4, 1.75), (5.6, 2.35))
    arrow(ax, (6.4, 2.1), (6.4, 1.15))
    ax.text(6.55, 1.55, "c = (x1 - input)^2", fontsize=8, va="center")
    arrow(ax, (5.4, 1.45), (5.6, 0.85))
    arrow(ax, (7.2, 0.88), (7.5, 1.62))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture_tranad.pdf")
    plt.close(fig)


def save_architecture_tranad_e_topk():
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    box(ax, (0.2, 1.45), "Input window\nW x F")
    box(ax, (1.9, 1.45), "Baseline\nTranAD core", width=1.7)
    box(ax, (4.0, 2.25), "Phase 1\nx1")
    box(ax, (4.0, 0.75), "Phase 2\nx2")
    box(ax, (6.0, 1.45), "Learnable blend\nx = (1-a)x1 + ax2", width=2.1, color="#FFF2D8")
    box(ax, (8.5, 1.45), "Top-k score\nmean largest k dims", width=1.8, color="#FCE8E6")

    arrow(ax, (1.8, 1.72), (1.9, 1.72))
    arrow(ax, (3.6, 1.95), (4.0, 2.5))
    arrow(ax, (3.6, 1.5), (4.0, 1.0))
    arrow(ax, (5.6, 2.5), (6.0, 2.0))
    arrow(ax, (5.6, 1.0), (6.0, 1.65))
    arrow(ax, (8.1, 1.72), (8.5, 1.72))
    ax.text(6.05, 0.45, "Structural change: phase-output ensemble\nScoring change: top-k aggregation over feature errors",
            fontsize=8.5, ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture_tranad_e_topk.pdf")
    plt.close(fig)


def save_prism_csv():
    rows = []
    delta_rows = []
    for dataset, values in DATA.items():
        base = values["TranAD"]
        enhanced = values["TranAD_E"]
        for model_name, metrics in values.items():
            rows.append({
                "Dataset": dataset,
                "Model": "TranAD-Ensemble + top-k" if model_name == "TranAD_E" else model_name,
                "F1": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
            })
        delta_rows.append({
            "Dataset": dataset,
            "Delta_F1": enhanced["f1"] - base["f1"],
            "Delta_Precision": enhanced["precision"] - base["precision"],
            "Delta_Recall": enhanced["recall"] - base["recall"],
        })

    with open(DATA_DIR / "metrics_for_prism.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Model", "F1", "Precision", "Recall"])
        writer.writeheader()
        writer.writerows(rows)

    with open(DATA_DIR / "metric_deltas_for_prism.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Delta_F1", "Delta_Precision", "Delta_Recall"])
        writer.writeheader()
        writer.writerows(delta_rows)


def main():
    save_prism_csv()
    save_f1_comparison()
    save_metric_deltas()
    save_architecture_tranad()
    save_architecture_tranad_e_topk()


if __name__ == "__main__":
    main()
