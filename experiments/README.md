# TranAD Variant Experiments

This directory contains standalone experiment entrypoints for TranAD, composable
TranAD variants, and the trainable repository baselines. The baseline `main.py`
is still available; these scripts are for controlled runs with epoch-level
metrics.

## Variant Tokens

Model names are composed as `TranAD_<tokens>`, where tokens can be combined in any
order:

- `C`: local temporal convolution adapter before the TranAD encoder.
- `G`: feature gate based on per-window feature statistics.
- `E`: learnable phase output blend between TranAD phase-1 and phase-2 outputs.
- `R`: RevIN-style reversible instance normalization from DCdetector.
- `V`: variable-temporal attention mixer inspired by VTT/DTAAD dual attention.
- `A`: association-discrepancy auxiliary loss inspired by Anomaly Transformer.
- `M`: sparse memory refinement inspired by MEMTO.

Use `C/G/E` as the primary improvement candidates. `R/V/A/M` are retained for
research comparison, but they are higher-risk and may underperform the baseline.

Examples:

```bash
TranAD_R
TranAD_V
TranAD_A
TranAD_M
TranAD_C
TranAD_G
TranAD_E
TranAD_C_G_E
TranAD_R_V
TranAD_R_A_M
TranAD_R_V_A_M
```

The original `TranAD` remains unchanged and can be used as the baseline.

## Training

```bash
conda run -n AnomalyDetection python experiments/train_tranad_variants.py \
  --model TranAD_R_V_A_M \
  --dataset synthetic \
  --epochs 5 \
  --batch-size 128 \
  --device auto \
  --score-agg topk \
  --score-topk 3
```

Outputs:

- `experiment_checkpoints/<model>_<dataset>/latest.ckpt`
- `experiment_checkpoints/<model>_<dataset>/best.ckpt`
- `experiment_logs/<model>_<dataset>/metrics.csv`
- `experiment_logs/<model>_<dataset>/per_dim_epoch_<epoch>.csv`
- `experiment_logs/<model>_<dataset>/config.json`

The metrics CSV records loss, learning rate, precision, recall, F1, ROC/AUC, and
auxiliary losses after each evaluated epoch.

Trainable baselines use the same entrypoint and write the same metric files:

- `OmniAnomaly`
- `GDN`
- `DAGMM`
- `MSCRED`
- `LSTM_AD` for the paper label LSTM-NDT
- `MAD_GAN`
- `USAD`
- `MTAD_GAT`
- `CAE_M`

```bash
conda run -n AnomalyDetection python experiments/train_tranad_variants.py \
  --model OmniAnomaly \
  --dataset MSL \
  --epochs 5 \
  --device auto
```

`LSTM-NDT` and `LSTM_NDT` CLI aliases are accepted and are normalized to
`LSTM_AD`. Paper-style `MAD-GAN`, `MTAD-GAT`, and `CAE-M` names are also
accepted and normalized to their underscored repository class names. `MERLIN`
is implemented as a parameter-free detector in this repository, so it has no
train step or checkpoint.

## Testing

```bash
conda run -n AnomalyDetection python experiments/test_tranad_variants.py \
  --model TranAD_R_V_A_M \
  --dataset synthetic \
  --device auto \
  --score-agg topk \
  --score-topk 3
```

The same test entrypoint evaluates the listed baselines. `MERLIN` is test-only:

```bash
conda run -n AnomalyDetection python experiments/test_tranad_variants.py \
  --model MERLIN \
  --dataset NAB
```

Outputs:

- `experiment_logs/<model>_<dataset>/test_metrics.json`
- `experiment_logs/<model>_<dataset>/test_per_dim.csv`

## Paper-Style Figures

Generate anomaly prediction and focus/attention figures from the selected model
checkpoint:

```bash
conda run -n AnomalyDetection python experiments/visualize_paper_figures.py \
  --model TranAD_C_G_E \
  --dataset SMD \
  --score-agg topk \
  --score-topk 3
```

Outputs:

- `experiment_visualizations/<model>_<dataset>/figure2_anomaly_prediction.png`
- `experiment_visualizations/<model>_<dataset>/figure3_focus_attention.png`
- `experiment_visualizations/<model>_<dataset>/visualization_metrics.json`

TranAD and TranAD variants expose their encoder temporal attention in the
Figure 3-style plot. Models without transformer attention, including
`OmniAnomaly`, show a loss-contribution heatmap in the attention panel and keep
the feature/time focus panels unchanged.

## Table 2-Style Tables

After running test jobs, render metric tables from their `test_metrics.json`
files:

```bash
conda run -n AnomalyDetection python experiments/render_table2_results.py \
  --models TranAD TranAD_C_G_E OmniAnomaly \
  --datasets NAB UCR MBA SMAP MSL SWaT WADI SMD MSDS
```

The script writes table images for F1, precision, recall, and AUC plus CSV and
LaTeX tables under `experiment_tables/table2/`. Missing test logs remain blank
unless `--strict` is supplied.

## One-Click TranAD_E Benchmark

Run the requested TranAD_E comparison workflow:

```bash
bash experiments/run_tranad_e_table2_benchmark.sh
```

The shell script trains `TranAD_E`, `TranAD`, `GDN`, `DAGMM`, `MSCRED`,
LSTM-NDT (`LSTM_AD`), MAD-GAN, `USAD`, MTAD-GAT, and CAE-M with batch size 128
and 10 epochs on `NAB`, `MBA`, `SWaT`, `synthetic`, `UCR`, `MSL`, `SMAP`, and
`SMD`. It applies top-k score aggregation only to `TranAD_E`, renders the
TranAD_E paper-style figures for every listed dataset, tests all listed models
plus the test-only `MERLIN`, and then renders strict Table 2-style tables.

The benchmark uses `NAB`, not `NBA`, because that is the dataset name in the
TranAD repository. The `MERLIN` test stage can be slow on long sequences because
it performs subsequence search rather than checkpoint inference.

Useful overrides:

```bash
DEVICE=cpu EPOCHS=1 EVAL_EVERY=0 \
DATASETS_OVERRIDE="synthetic" \
TRAIN_MODELS_OVERRIDE="TranAD_E DAGMM" \
TEST_MODELS_OVERRIDE="TranAD_E DAGMM MERLIN" \
bash experiments/run_tranad_e_table2_benchmark.sh
```

Set `DRY_RUN=1` to print generated commands without running them. Set
`KEEP_GOING=1` to continue after a failed step. `TABLE_STRICT=1` is the default
and rejects a final table when a requested `test_metrics.json` file is missing.

## Test-Only Report Workflow

After training has already finished, run only validation, TranAD_E paper-style
figures, and Table 2-style tables:

```bash
bash experiments/run_tranad_e_test_report.sh
```

This script does not train or overwrite checkpoints. It tests the existing
checkpoints for `TranAD_E`, `TranAD`, `GDN`, `DAGMM`, `MSCRED`, LSTM-NDT
(`LSTM_AD`), MAD-GAN, `USAD`, MTAD-GAT, CAE-M, and the test-only `MERLIN` on
`NAB`, `MBA`, `SWaT`, `synthetic`, `UCR`, `MSL`, `SMAP`, and `SMD`. It applies
top-k score aggregation only to `TranAD_E`, then writes English paper-style
figures and English Table 2-style metric tables.

Useful test-only overrides:

```bash
DRY_RUN=1 bash experiments/run_tranad_e_test_report.sh

DATASETS_OVERRIDE="SMAP SMD" \
TEST_MODELS_OVERRIDE="TranAD_E TranAD MERLIN" \
bash experiments/run_tranad_e_test_report.sh
```

The script defaults to `KEEP_GOING=1` and `TABLE_STRICT=0` so that one failed
model or one missing checkpoint does not block later report artifacts. Set
`TABLE_STRICT=1` when you want the final table step to fail if any requested
metric file is missing.

## TranAD_E Top-K Ablation

Run the four-stage ablation from the original TranAD baseline to TranAD_E with
top-k score aggregation:

```bash
bash experiments/run_tranad_e_topk_ablation.sh
```

The script trains only the two missing ablation settings:

- `TranAD+topk`: `TranAD` trained and tested with `--score-agg topk --score-topk 3`.
- `TranAD_E`: `TranAD_E` trained and tested with the default mean score aggregation.

It does not retrain the two already available settings:

- `TranAD`: read from `experiment_logs/TranAD_<dataset>/test_metrics.json`.
- `TranAD_E+topk`: read from `experiment_logs/TranAD_E_<dataset>/test_metrics.json`.

Default datasets:

```text
NAB UCR MBA SMD SMAP MSL SWaT synthetic
```

Default training parameters:

```text
--epochs 10
--batch-size 128
--device auto
```

Outputs:

- `experiment_logs/ablation_tranad_e_topk/tranad_topk/<model>_<dataset>/`
- `experiment_logs/ablation_tranad_e_topk/tranad_e_mean/<model>_<dataset>/`
- `experiment_tables/tranad_e_ablation/ablation_metrics_long.csv`
- `experiment_tables/tranad_e_ablation/ablation_precision.csv`
- `experiment_tables/tranad_e_ablation/ablation_recall.csv`
- `experiment_tables/tranad_e_ablation/ablation_f1.csv`
- `experiment_tables/tranad_e_ablation/ablation_roc_auc.csv`

Use `DRY_RUN=1` to print commands only, `FORCE=1` to rerun existing ablation
outputs, and `DATASETS_OVERRIDE="SMAP SMD"` to restrict datasets.

## Adaptive Top-K Multi-Seed Benchmark

`TranAD_E` can now use validation-selected adaptive top-k aggregation:

```bash
python experiments/train_tranad_variants.py \
  --model TranAD_E \
  --dataset SMD \
  --score-agg topk \
  --score-topk auto
```

With `--score-topk auto`, the evaluation code enumerates valid k values for the
current feature dimension count, selects the k with the best F1 score, and
records `score_topk: auto` plus `selected_topk` in `metrics.csv` and
`test_metrics.json`. Fixed top-k behavior is unchanged; scripts that pass
`--score-topk 3`, including `run_tranad_e_test_report.sh`, still use fixed k=3.

Run the five-seed benchmark for `TranAD_E` adaptive-k and the requested
baselines:

```bash
bash experiments/run_adaptive_k_multiseed_benchmark.sh
```

Default settings:

```text
Seeds: 1 2 3 4 5
Datasets: NAB MBA SWaT synthetic UCR MSL SMAP SMD
Models: TranAD_E TranAD GDN DAGMM MSCRED MAD_GAN USAD MTAD_GAT CAE_M MERLIN
TranAD_E scoring: --score-agg topk --score-topk auto
Baselines scoring: default mean aggregation
Training: --epochs 10 --batch-size 128 --device auto
```

Outputs:

- `experiment_checkpoints/multiseed_adaptive_k/seed_<seed>/`
- `experiment_logs/multiseed_adaptive_k/seed_<seed>/<model>_<dataset>/test_metrics.json`
- `experiment_visualizations/multiseed_adaptive_k/seed_1/TranAD_E_<dataset>/`
- `experiment_tables/multiseed_adaptive_k/multiseed_metrics_long.csv`
- `experiment_tables/multiseed_adaptive_k/multiseed_summary_long.csv`
- `experiment_tables/multiseed_adaptive_k/multiseed_precision_mean_std.csv`
- `experiment_tables/multiseed_adaptive_k/multiseed_recall_mean_std.csv`
- `experiment_tables/multiseed_adaptive_k/multiseed_f1_mean_std.csv`
- `experiment_tables/multiseed_adaptive_k/multiseed_roc_auc_mean_std.csv`
- `experiment_tables/multiseed_adaptive_k/multiseed_*_mean_std.png`
- `experiment_tables/multiseed_adaptive_k/multiseed_tables.tex`

The table cells are formatted as `mean±std`, for example `0.8524±0.0012`.
Useful controls:

```bash
DRY_RUN=1 bash experiments/run_adaptive_k_multiseed_benchmark.sh

SEEDS_OVERRIDE="1 2" \
DATASETS_OVERRIDE="SMD SMAP" \
TRAIN_MODELS_OVERRIDE="TranAD_E TranAD" \
TEST_MODELS_OVERRIDE="TranAD_E TranAD MERLIN" \
bash experiments/run_adaptive_k_multiseed_benchmark.sh

SKIP_TRAIN=1 bash experiments/run_adaptive_k_multiseed_benchmark.sh
FORCE=1 bash experiments/run_adaptive_k_multiseed_benchmark.sh
SUMMARY_STRICT=1 bash experiments/run_adaptive_k_multiseed_benchmark.sh
```

## Adaptive Top-K Ablation With Multi-Seed Results

After `run_adaptive_k_multiseed_benchmark.sh` has already produced the
multi-seed `TranAD` and `TranAD_E+adaptive-k` results, run the missing ablation
settings only:

```bash
bash experiments/run_adaptive_k_ablation_multiseed.sh
```

This script keeps the same default experimental parameters:

```text
Seeds: 1 2 3 4 5
Datasets: NAB MBA SWaT synthetic UCR MSL SMAP SMD
Training: --epochs 10 --batch-size 128 --device auto
Existing results read from: experiment_logs/multiseed_adaptive_k
```

The ablation columns are:

- `TranAD`: existing multi-seed baseline.
- `TranAD+adaptive-topk`: new `TranAD` run with `--score-agg topk --score-topk auto`.
- `TranAD+learnable-ensemble`: new `TranAD_E` run with default mean aggregation.
- `TranAD_E+adaptive-topk`: existing final model from the multi-seed benchmark.

Outputs:

- `experiment_checkpoints/adaptive_k_ablation/seed_<seed>/`
- `experiment_logs/adaptive_k_ablation/seed_<seed>/`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_metrics_long.csv`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_summary_long.csv`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_precision_mean_std.csv`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_recall_mean_std.csv`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_f1_mean_std.csv`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_roc_auc_mean_std.csv`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_*_mean_std.png`
- `experiment_tables/adaptive_k_ablation/adaptive_k_ablation_tables.tex`

Useful controls:

```bash
DRY_RUN=1 bash experiments/run_adaptive_k_ablation_multiseed.sh

SEEDS_OVERRIDE="1 2" \
DATASETS_OVERRIDE="SMD SMAP" \
bash experiments/run_adaptive_k_ablation_multiseed.sh

SKIP_TRAIN=1 bash experiments/run_adaptive_k_ablation_multiseed.sh
FORCE=1 bash experiments/run_adaptive_k_ablation_multiseed.sh
SUMMARY_STRICT=1 bash experiments/run_adaptive_k_ablation_multiseed.sh
```

## Recommended Ablation Order

Run the baseline first, then the low-risk single-token variants:

```bash
TranAD
TranAD_C
TranAD_G
TranAD_E
```

Then test pairwise combinations:

```bash
TranAD_C_G
TranAD_C_E
TranAD_G_E
```

Finally test the main combined candidate:

```bash
TranAD_C_G_E
```

## Full C/G/E + Adaptive Top-K Ablation

Run every `C/G/E` subset with both default mean score aggregation and
validation-selected adaptive top-k aggregation (16 settings in total):

```bash
bash experiments/run_cge_adaptive_k_ablation_multiseed.sh
```

By default this runs five seeds on every currently processed, supported dataset:

```text
MBA MSDS NAB MSL SMAP SMD SWaT UCR WADI synthetic
```

The eight model configurations are `TranAD`, `TranAD_C`, `TranAD_G`,
`TranAD_E`, `TranAD_C_G`, `TranAD_C_E`, `TranAD_G_E`, and `TranAD_C_G_E`.
For each one, the adaptive setting adds `--score-agg topk --score-topk auto` to
both training validation and final testing. Results are kept separate from the
mean setting, so they never share a checkpoint or a metric file.

The final reports include precision, recall, F1, and ROC/AUC for every
model/dataset/seed, followed by mean+-standard-deviation CSV, PNG, and LaTex
tables in `experiment_tables/cge_adaptive_k_ablation/`. The long CSV also
records the selected adaptive `k` for each test run.

Useful scoped runs:

```bash
DRY_RUN=1 bash experiments/run_cge_adaptive_k_ablation_multiseed.sh

SEEDS_OVERRIDE="1 2" \
DATASETS_OVERRIDE="SMAP SMD" \
MODELS_OVERRIDE="TranAD TranAD_C_G_E" \
bash experiments/run_cge_adaptive_k_ablation_multiseed.sh

# Run only final-model adaptive top-k testing after its checkpoints already exist.
SKIP_TRAIN=1 \
MODELS_OVERRIDE="TranAD_C_G_E" \
SCORE_SETTINGS_OVERRIDE="adaptive_topk" \
bash experiments/run_cge_adaptive_k_ablation_multiseed.sh
```

After the `C/G/E` family is characterized, optionally compare against the
research-heavy tokens:

```bash
TranAD_R
TranAD_V
TranAD_A
TranAD_M
TranAD_R_V_A_M
```
