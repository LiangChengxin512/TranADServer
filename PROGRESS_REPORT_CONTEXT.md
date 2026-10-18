# Progress Report Context for ChatGPT Recovery

This file summarizes the current project context for continuing the progress report in ChatGPT.

## Research Topic

Research direction:

> Using artificial intelligence methods and the theory of critical events for critical transitions forecasting in complex systems.

Current experimental work is based on the TranAD time-series anomaly detection project. The progress report should be written in English and can use the current experimental outputs as evidence.

## Main Report Focus

The current report should focus on:

- `TranAD` as the baseline model.
- `TranAD_E` as the current improved variant.
- `TranAD_E` means `TranAD-Ensemble`: it introduces a learnable ensemble/blending of the two TranAD reconstruction phases.
- `TranAD_E` uses `topk` score aggregation in testing and visualization:
  - `--score-agg topk`
  - `--score-topk 3`
- Other old exploratory variants such as `TranAD_R`, `TranAD_V`, `TranAD_R_V_A_M`, and `TranAD_C_G_E` should not be emphasized in the current report unless needed as background.

Existing English LaTeX report:

- `reports/tranad_e_topk/report.tex`

Existing report data and figures:

- `reports/tranad_e_topk/data/metrics_for_prism.csv`
- `reports/tranad_e_topk/data/metric_deltas_for_prism.csv`
- `reports/tranad_e_topk/figures/f1_comparison.pdf`
- `reports/tranad_e_topk/figures/metric_deltas.pdf`
- `reports/tranad_e_topk/figures/architecture_tranad.pdf`
- `reports/tranad_e_topk/figures/architecture_tranad_e_topk.pdf`
- `reports/tranad_e_topk/plot_results.py`

## Newly Added Experiment Scripts

The current experiment workflow uses these scripts:

- `experiments/train_tranad_variants.py`
  - Trains TranAD variants and implemented repository baselines.
  - Supports `--model`, `--dataset`, `--epochs`, `--batch-size`, `--device`, `--score-agg`, `--score-topk`.
- `experiments/test_tranad_variants.py`
  - Tests trained checkpoints and writes `test_metrics.json` and `test_per_dim.csv`.
  - Also supports test-only `MERLIN`.
- `experiments/visualize_paper_figures.py`
  - Generates paper-style visualizations:
    - `figure2_anomaly_prediction.png`
    - `figure3_focus_attention.png`
  - For TranAD/TranAD variants, Figure 3 includes temporal self-attention, focus over features, and focus over time.
  - For models without Transformer attention, the attention panel falls back to a loss-contribution heatmap.
- `experiments/render_table2_results.py`
  - Reads `experiment_logs/<model>_<dataset>/test_metrics.json`.
  - Generates Table 2-style CSV, PNG, and LaTeX tables for:
    - F1
    - Precision
    - Recall
    - ROC/AUC
- `experiments/run_tranad_e_table2_benchmark.sh`
  - Full workflow: train, test, render TranAD_E figures, render Table 2 tables.
- `experiments/run_tranad_e_test_report.sh`
  - Test/report-only workflow after training has finished.
  - This is the recommended script now because training has already been completed.
  - It does not train models or overwrite checkpoints.

Recommended command for report artifact generation:

```bash
bash experiments/run_tranad_e_test_report.sh
```

Useful reduced rerun command:

```bash
DATASETS_OVERRIDE="SMAP SMD" \
TEST_MODELS_OVERRIDE="TranAD_E TranAD MERLIN" \
bash experiments/run_tranad_e_test_report.sh
```

Dry run command:

```bash
DRY_RUN=1 bash experiments/run_tranad_e_test_report.sh
```

## Hardware and Environment

User environment:

- MacBook Air M4
- macOS 26.5
- Conda environment: `AnomalyDetection`
- Python: 3.10
- Device policy:
  - `--device auto` selects CUDA, MPS, or CPU depending on availability.
  - Floating point precision is kept as `float32`.
  - DGL models (`GDN`, `MTAD_GAT`) fall back to CPU on MPS because DGL does not support MPS.

Relevant device utility:

- `src/device.py`

## Models Included in the Latest Benchmark

Trainable/testable models:

- `TranAD_E`
- `TranAD`
- `GDN`
- `DAGMM`
- `MSCRED`
- `MAD_GAN` / paper label `MAD-GAN`
- `USAD`
- `MTAD_GAT` / paper label `MTAD-GAT`
- `CAE_M` / paper label `CAE-M`

Test-only model:

- `MERLIN`

Important naming note:

- The paper name `LSTM-NDT` maps to repository class `LSTM_AD`.
- In the most recent benchmark shell script, `LSTM_AD` may be commented out or omitted depending on the local edited model list. Check `experiments/run_tranad_e_table2_benchmark.sh` and `experiments/run_tranad_e_test_report.sh` before claiming LSTM-NDT is included in the final table.

Current default test/report script model list:

```text
TranAD_E
TranAD
GDN
DAGMM
MSCRED
LSTM_AD
MAD_GAN
USAD
MTAD_GAT
CAE_M
MERLIN
```

## Datasets Included

The current test/report workflow uses eight datasets:

```text
NAB
MBA
SWaT
synthetic
UCR
MSL
SMAP
SMD
```

Important naming note:

- The correct dataset name is `NAB`, not `NBA`.
- `NAB` means Numenta Anomaly Benchmark.
- The original long-term plan also mentioned WADI and MSDS, but the current generated Table 2-style report artifacts are for the eight datasets listed above.

## Current Figure Outputs

Table 2-style report figures:

- `experiment_tables/tranad_e_report/table2_f1.png`
- `experiment_tables/tranad_e_report/table2_precision.png`
- `experiment_tables/tranad_e_report/table2_recall.png`
- `experiment_tables/tranad_e_report/table2_roc_auc.png`

Table 2-style source data:

- `experiment_tables/tranad_e_report/table2_metrics_long.csv`
- `experiment_tables/tranad_e_report/table2_f1.csv`
- `experiment_tables/tranad_e_report/table2_precision.csv`
- `experiment_tables/tranad_e_report/table2_recall.csv`
- `experiment_tables/tranad_e_report/table2_roc_auc.csv`
- `experiment_tables/tranad_e_report/table2_tables.tex`

TranAD_E paper-style figures, generated for each dataset:

- `experiment_visualizations/tranad_e_report/TranAD_E_NAB/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_NAB/figure3_focus_attention.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_MBA/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_MBA/figure3_focus_attention.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_SWaT/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_SWaT/figure3_focus_attention.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_synthetic/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_synthetic/figure3_focus_attention.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_UCR/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_UCR/figure3_focus_attention.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_MSL/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_MSL/figure3_focus_attention.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_SMAP/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_SMAP/figure3_focus_attention.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_SMD/figure2_anomaly_prediction.png`
- `experiment_visualizations/tranad_e_report/TranAD_E_SMD/figure3_focus_attention.png`

Each visualization folder also contains:

- `visualization_metrics.json`

Example:

- `experiment_visualizations/tranad_e_report/TranAD_E_SMAP/visualization_metrics.json`

Example SMAP visualization metadata:

```json
{
  "model": "TranAD_E",
  "dataset": "SMAP",
  "checkpoint": "experiment_checkpoints/TranAD_E_SMAP/best.ckpt",
  "device": "mps",
  "dtype": "torch.float32",
  "start": 2049,
  "length": 400,
  "dimensions": [0, 19, 2, 3],
  "focus_index": 2188,
  "attention_available": true,
  "f1": 0.9110790723089599,
  "precision": 0.8366890286723823,
  "recall": 0.9999999866310163,
  "roc_auc": 0.9905891452881269,
  "score_agg": "topk"
}
```

## Current Training and Testing Output Files

Training outputs are under:

- `experiment_logs/<model>_<dataset>/metrics.csv`
- `experiment_logs/<model>_<dataset>/per_dim_epoch_<epoch>.csv`
- `experiment_checkpoints/<model>_<dataset>/best.ckpt`
- `experiment_checkpoints/<model>_<dataset>/latest.ckpt`

Test outputs are under:

- `experiment_logs/<model>_<dataset>/test_metrics.json`
- `experiment_logs/<model>_<dataset>/test_per_dim.csv`

Current observed output counts:

- `metrics.csv`: 72 files
- `test_metrics.json`: 80 files

These numbers may change if tests are rerun or if missing checkpoints are generated later.

## Training CSV Structure

Example file:

- `experiment_logs/TranAD_E_SMAP/metrics.csv`

Columns include:

- `epoch`
- `train_loss`
- `recon_loss`
- `aux_loss`
- `lr`
- `epoch_time_sec`
- `f1`
- `precision`
- `recall`
- `roc_auc`
- `threshold`
- `score_agg`
- `train_score_mean`
- `train_score_std`
- `test_score_mean`
- `test_score_std`
- `dim_f1_mean`
- `dim_precision_mean`
- `dim_recall_mean`

Meaning:

- One row per evaluated epoch.
- For the main benchmark, epoch count is usually 10.
- `score_agg` is `topk` for `TranAD_E` and usually `mean` for other models unless explicitly overridden.
- `f1`, `precision`, `recall`, and `roc_auc` are final aggregated time-series metrics after POT thresholding.
- `dim_*` metrics summarize per-feature evaluation.

Example first rows from `TranAD_E_SMAP/metrics.csv`:

```csv
epoch,train_loss,recon_loss,aux_loss,lr,epoch_time_sec,f1,precision,recall,roc_auc,threshold,score_agg,train_score_mean,train_score_std,test_score_mean,test_score_std,dim_f1_mean,dim_precision_mean,dim_recall_mean
0,0.0979463309859452,0.0979463309859452,0.0,0.001,0.894873857498169,0.8862509783860092,0.795744672385695,0.9999999866310163,0.987624081474797,0.5467799781643469,topk,...
1,0.03556851645850617,0.03556851645850617,0.0,0.001,0.4697301387786865,0.8920642187727628,0.8051668374040168,0.9999999866310163,0.9883331184736367,0.9997833769731193,topk,...
```

## Test JSON Structure

Example file:

- `experiment_logs/TranAD_E_SMAP/test_metrics.json`

Fields include:

- `checkpoint`
- `device`
- `dtype`
- `f1`
- `precision`
- `recall`
- `roc_auc`
- `threshold`
- `score_agg`
- `train_score_mean`
- `train_score_std`
- `test_score_mean`
- `test_score_std`
- `dim_f1_mean`
- `dim_precision_mean`
- `dim_recall_mean`

For `MERLIN`, fields differ slightly because it is parameter-free:

- `checkpoint`: `null`
- `device`: `cpu`
- `dtype`: `numpy`
- `score_agg`: `MERLIN`
- `merlin_status`
- `discord_start`
- `discord_length`
- `discord_distance`

Important recent issue and fix:

- During testing, `MERLIN on SMAP` failed with `ValueError: attempt to get argmax of an empty sequence`.
- The fix was added in `src/experiment_utils.py`.
- `evaluate_merlin()` now catches this specific empty-discord condition and writes valid metrics with:

```json
"merlin_status": "fallback_no_discord"
```

Example:

- `experiment_logs/MERLIN_SMAP/test_metrics.json`

```json
{
  "checkpoint": null,
  "device": "cpu",
  "dtype": "numpy",
  "f1": 0.12261678240953185,
  "precision": 0.10139860051225,
  "recall": 0.15508021183047846,
  "roc_auc": 0.5112773765148471,
  "score_agg": "MERLIN",
  "merlin_status": "fallback_no_discord",
  "threshold": null,
  "discord_start": null,
  "discord_length": null,
  "discord_distance": null
}
```

## Table 2-Style CSV Structure

Long-format table file:

- `experiment_tables/tranad_e_report/table2_metrics_long.csv`

Columns:

- `dataset`
- `model`
- `path`
- `f1`
- `precision`
- `recall`
- `roc_auc`

Example rows:

```csv
dataset,model,path,f1,precision,recall,roc_auc
NAB,TranAD_E,experiment_logs/TranAD_E_NAB/test_metrics.json,0.9411711188269714,0.8888885596709039,0.9999995833335069,0.999625748502994
NAB,TranAD,experiment_logs/TranAD_NAB/test_metrics.json,0.9411711188269714,0.8888885596709039,0.9999995833335069,0.999625748502994
NAB,GDN,experiment_logs/GDN_NAB/test_metrics.json,0.7441807680154827,0.8421048199448317,0.6666663888890046,0.8329590818363273
NAB,DAGMM,experiment_logs/DAGMM_NAB/test_metrics.json,0.7441807680154827,0.8421048199448317,0.6666663888890046,0.8329590818363273
```

Wide-format table files:

- `table2_f1.csv`
- `table2_precision.csv`
- `table2_recall.csv`
- `table2_roc_auc.csv`

These can be used directly for Prism or LaTeX report tables.

## Important Caveats for the Report

- The current benchmark is not identical to the original TranAD paper Table 2 because:
  - It uses current local code and current processed dataset selections.
  - It uses `TranAD_E` with top-k aggregation as the proposed/current method.
  - `MERLIN` is parameter-free and can use fallback behavior on SMAP if no discord candidate is found.
- `MERLIN_SMAP` should be described carefully:
  - It completed after the fallback fix.
  - `merlin_status = fallback_no_discord`.
  - This means the original MERLIN discord search did not produce a valid discord candidate for the configured SMAP window range, so the metric reflects the fallback check-based prediction path.
- `NAB`, not `NBA`, is the correct dataset name.
- WADI and MSDS are not part of the current generated `tranad_e_report` table unless rerun with explicit dataset overrides and valid processed data/checkpoints.

## Suggested Progress Report Structure

Suggested English report sections:

1. Introduction
   - Complex systems and critical transition forecasting.
   - Time-series anomaly detection as an operational proxy for early warning signals.
2. Baseline and Improved Method
   - TranAD baseline.
   - TranAD-Ensemble (`TranAD_E`) and top-k score aggregation.
3. Experimental Setup
   - Hardware/software environment.
   - Datasets: NAB, MBA, SWaT, synthetic, UCR, MSL, SMAP, SMD.
   - Compared methods: TranAD, GDN, DAGMM, MSCRED, MAD-GAN, USAD, MTAD-GAT, CAE-M, MERLIN, and optionally LSTM-NDT if final logs include it.
4. Metrics
   - Precision, Recall, F1 Score, ROC/AUC.
   - POT thresholding and top-k aggregation for `TranAD_E`.
5. Results
   - Use `table2_f1.png`, `table2_precision.png`, `table2_recall.png`, `table2_roc_auc.png`.
   - Discuss where `TranAD_E` improves, matches, or decreases compared with `TranAD`.
6. Visualization Analysis
   - Use `figure2_anomaly_prediction.png` to show anomaly prediction regions.
   - Use `figure3_focus_attention.png` to show temporal attention and focus over features/time.
7. Current Limitations and Next Steps
   - Need broader reruns if including WADI/MSDS.
   - Need careful handling of MERLIN fallback cases.
   - Need repeated runs/seeds for statistical reliability.

## Files Most Useful for GPT Continuation

Open these first in a resumed GPT session:

1. `reports/tranad_e_topk/report.tex`
2. `experiment_tables/tranad_e_report/table2_metrics_long.csv`
3. `experiment_tables/tranad_e_report/table2_f1.csv`
4. `experiment_tables/tranad_e_report/table2_precision.csv`
5. `experiment_tables/tranad_e_report/table2_recall.csv`
6. `experiment_tables/tranad_e_report/table2_roc_auc.csv`
7. `experiment_visualizations/tranad_e_report/TranAD_E_SMAP/visualization_metrics.json`
8. `experiment_logs/MERLIN_SMAP/test_metrics.json`
9. `experiments/README.md`

## Current Code Changes Relevant to the Report

- `src/experiment_utils.py`
  - Shared data preparation, training, prediction, and evaluation dispatch for TranAD variants and baselines.
  - `evaluate_merlin()` fallback for empty discord sequences.
- `src/dlutils.py`
  - Optional attention capture in Transformer encoder/decoder layers.
  - Used only for visualization, not intended to change normal training behavior.
- `experiments/visualize_paper_figures.py`
  - Generates paper-style Figure 2 and Figure 3.
- `experiments/render_table2_results.py`
  - Generates English Table 2-style results.
- `experiments/run_tranad_e_test_report.sh`
  - Main test/report-only workflow after training.

