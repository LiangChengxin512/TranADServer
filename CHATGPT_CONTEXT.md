# TranAD Project Context for ChatGPT Continuation

This file summarizes the current project state, implemented changes, experimental direction, and report artifacts so the conversation can be resumed in ChatGPT.

## Project and Environment

- Project root: `/Users/Liang/PycharmProjects/TranAD`
- Conda environment: `AnomalyDetection`
- Python: 3.10
- Main goal: adapt TranAD for hardware-aware CPU/CUDA/MPS execution and explore lightweight improvements for critical transition forecasting in complex systems.
- Research direction: `Using artificial intelligence methods and the theory of critical events for critical transitions forecasting in complex systems`.

## Hardware / Device Support Changes

The original project was CPU-oriented. It has been modified to support automatic device selection:

- Device priority: `CUDA -> MPS -> CPU`
- CLI flag added: `--device auto|cpu|cuda|mps`
- All devices now use `torch.float32` for comparable experiments.
- DGL-based models (`GDN`, `MTAD_GAT`) fall back to CPU on MPS because DGL does not support MPS.

Key file:

- `src/device.py`

Important functions:

- `get_best_device(requested='auto', model_name=None)`
- `get_dtype_for_device(device)` returns `torch.float32` for all devices.
- `move_optimizer_to_device(optimizer, device, dtype)`

## Main Code Fixes

### Test-stage hang fix

The process appeared to hang after:

```python
print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
```

The actual issue was not model inference. It was in `pot_eval()`:

```python
while True:
    try:
        ...
    except:
        lms = lms * 0.999
```

When SPOT initialization failed on degenerate score distributions, this loop could run indefinitely.

Fixes:

- Removed infinite retry.
- Added finite initialization levels.
- Added fallback percentile threshold.
- Cleaned NaN/Inf scores before POT.
- Added POT progress bar in main evaluation.

Files:

- `src/pot.py`
- `main.py`

### TranAD test batching fix

TranAD test previously used full test set as one batch:

```python
bs = len(data)
```

This was changed to use `model.batch` during testing as well.

Also added `torch.no_grad()` during test and train-score evaluation.

File:

- `main.py`

### PyTorch 2.x compatibility

Custom Transformer layers now accept causal keyword arguments:

- `is_causal`
- `tgt_is_causal`
- `memory_is_causal`

File:

- `src/dlutils.py`

### Pandas 2 compatibility

Old `DataFrame.append()` was replaced with list accumulation and DataFrame construction.

Files:

- `main.py`
- `src/experiment_utils.py`

### Matplotlib cache/style fixes

Added writable Matplotlib cache path:

```python
MPLCONFIGDIR=/private/tmp/tranad_matplotlib
```

Files:

- `src/plotting.py`
- `src/spot.py`
- `reports/tranad_e_topk/plot_results.py`

## Dependency Notes

Current `requirements.txt` pins versions for macOS/Python 3.10 compatibility:

- `torch==2.2.2`
- `torchvision==0.17.2`
- `torchaudio==2.2.2`
- `torchdata==0.7.1`
- `numpy<2`
- `dgl`
- `pyyaml`
- `pydantic`

DGL 2.2.0 on macOS requires a PyTorch version with matching GraphBolt dynamic libraries. PyTorch 2.12 caused DGL import failure. PyTorch 2.2.2 worked.

## Research Directory

The user added `Research/`, containing multiple external research projects:

- `Anomaly-Transformer`
- `KDD2023-DCdetector`
- `MEMTO`
- `DTAAD`
- `GTA`
- `TransDe`
- `AnomalyBERT`
- `KBS2024-VTT`

These projects are treated as reference material only. They do not need to run. Main TranAD project code outside `Research/` is the target.

## TranAD Variants

Dynamic TranAD variants were added.

Main files:

- `src/tranad_variants.py`
- `src/models.py`

The original `TranAD` baseline remains unchanged. Variant names are dynamically resolved through `src.models.__getattr__`.

### Initial research-heavy tokens

These were added but are not currently recommended as the primary direction:

- `R`: RevIN-style normalization from DCdetector.
- `V`: variable-temporal attention mixer inspired by VTT/DTAAD/DCdetector.
- `A`: association discrepancy auxiliary loss inspired by Anomaly Transformer.
- `M`: sparse memory refinement inspired by MEMTO.

Observation:

- `TranAD_R_V_A_M` and related combinations often underperformed TranAD or showed no useful improvement.

### Recommended lightweight tokens

The better current direction is:

- `C`: local temporal convolution adapter before the TranAD encoder.
- `G`: feature gate based on per-window feature statistics.
- `E`: learnable phase-output ensemble between TranAD phase 1 and phase 2.

Recommended ablation order:

```text
TranAD
TranAD_C
TranAD_G
TranAD_E
TranAD_C_G
TranAD_C_E
TranAD_G_E
TranAD_C_G_E
```

Important note:

- `E` means `Ensemble`.
- In code, model name is `TranAD_E`.
- In report, it is described as `TranAD-Ensemble`.

## Experiment Scripts

New standalone scripts:

- `experiments/train_tranad_variants.py`
- `experiments/test_tranad_variants.py`
- `experiments/README.md`

These scripts avoid the old global parser issue by parsing their own arguments first and shielding imports from the original project parser.

### Training example

```bash
conda run -n AnomalyDetection python experiments/train_tranad_variants.py \
  --model TranAD_C_G_E \
  --dataset SMD \
  --epochs 10 \
  --batch-size 128 \
  --device auto \
  --score-agg topk \
  --score-topk 3
```

### Testing example

```bash
conda run -n AnomalyDetection python experiments/test_tranad_variants.py \
  --model TranAD_C_G_E \
  --dataset SMD \
  --device auto \
  --score-agg topk \
  --score-topk 3
```

### Checkpoint choice

Default test script uses:

```text
experiment_checkpoints/<model>_<dataset>/best.ckpt
```

To test last epoch:

```bash
--checkpoint experiment_checkpoints/<model>_<dataset>/latest.ckpt
```

This matters because metrics may not improve monotonically with epoch.

### Output logs

Training writes:

```text
experiment_logs/<model>_<dataset>/metrics.csv
experiment_logs/<model>_<dataset>/per_dim_epoch_<epoch>.csv
experiment_logs/<model>_<dataset>/config.json
```

Checkpoints:

```text
experiment_checkpoints/<model>_<dataset>/latest.ckpt
experiment_checkpoints/<model>_<dataset>/best.ckpt
```

## Score Aggregation

Experiment scripts support:

```bash
--score-agg mean|max|p95|topk
--score-topk 3
```

Meaning:

- `mean`: average reconstruction error over all dimensions. This matches original-style final aggregation.
- `max`: use maximum feature-wise error at each timestamp.
- `p95`: use 95th percentile feature error at each timestamp.
- `topk`: average the largest `k` feature errors at each timestamp.

`--score-topk 3` means for each timestamp, take the top 3 feature reconstruction errors and average them.

Important experimental guidance:

- To compare model architecture only, keep `--score-agg mean`.
- To evaluate scoring strategy, compare `mean` vs `topk` separately.
- `topk` may help localized anomalies but may hurt distributed weak anomalies.

## Observed Experimental Behavior

### Synthetic quick probe

Small quick checks on `synthetic --less --epochs 3` showed:

- Baseline `TranAD` mean aggregation best F1 around `0.979`.
- `TranAD_C` mean aggregation reached around `0.986`.
- `TranAD_G` mean aggregation reached around `0.989`.
- `TranAD_C_G_E` mean aggregation reached around `0.995` at best early epoch.
- `TranAD_C_G_E` with `topk` or `max` reached around `0.990`.

These are quick probes, not final conclusions.

### Issue with unchanged metrics

If `train_loss`, `recon_loss`, or raw score statistics change but F1/Precision/Recall do not, this can happen because POT thresholding and point-adjusted evaluation are discrete.

If different models have exactly identical metrics across all epochs, check:

- whether test script is loading `best.ckpt` from early epoch;
- whether the same checkpoint path is being reused;
- whether score aggregation is fixed or changed;
- whether metrics are dominated by post-processing thresholds.

## Current Report

The user requested an English LaTeX report to be run in Prism and used for future presentations.

Report directory:

```text
reports/tranad_e_topk/
```

Main files:

- `reports/tranad_e_topk/report.tex`
- `reports/tranad_e_topk/plot_results.py`

Figures:

- `reports/tranad_e_topk/figures/architecture_tranad.pdf`
- `reports/tranad_e_topk/figures/architecture_tranad_e_topk.pdf`
- `reports/tranad_e_topk/figures/f1_comparison.pdf`
- `reports/tranad_e_topk/figures/metric_deltas.pdf`

Prism CSV files:

- `reports/tranad_e_topk/data/metrics_for_prism.csv`
- `reports/tranad_e_topk/data/metric_deltas_for_prism.csv`

Regenerate figures and CSV:

```bash
conda run -n AnomalyDetection python reports/tranad_e_topk/plot_results.py
```

Compile LaTeX:

```bash
cd reports/tranad_e_topk
pdflatex report.tex
```

The report is fully English and uses standard `article`, not `ctexart`.

## Report Focus

The report intentionally focuses only on:

- `TranAD`
- `TranAD_E + topk`

Other combinations are ignored in the report.

In the report:

- `TranAD_E` is explained as implementation name.
- `E` stands for `Ensemble`.
- The model is presented as `TranAD-Ensemble`.

## Report Performance Data

Data used in report:

### SMD

TranAD-Ensemble + top-k:

- F1: `0.9133648819747217`
- Precision: `0.9951370424618167`
- Recall: `0.8440194943981271`

TranAD:

- F1: `0.9981188236746776`
- Precision: `0.9988734472441853`
- Recall: `0.9973753243442994`

### MSL

TranAD-Ensemble + top-k:

- F1: `0.9494589959667135`
- Precision: `0.9037900611139924`
- Recall: `0.9999999677419366`

TranAD:

- F1: `0.9494589959667135`
- Precision: `0.9037900611139924`
- Recall: `0.9999999677419366`

### Synthetic

TranAD-Ensemble + top-k:

- F1: `0.9836015385903053`
- Precision: `0.9677419159729453`
- Recall: `0.9999999791666672`

TranAD:

- F1: `0.9756047392681091`
- Precision: `0.9523809334845054`
- Recall: `0.9999999791666672`

### SWaT

TranAD-Ensemble + top-k:

- F1: `0.81508255249909`
- Precision: `0.9999999774266372`
- Recall: `0.6878881880762704`

TranAD:

- F1: `0.81508255249909`
- Precision: `0.9999999774266372`
- Recall: `0.6878881880762704`

### NAB

TranAD-Ensemble + top-k:

- F1: `0.9411711188269714`
- Precision: `0.8888885596709039`
- Recall: `0.9999995833335069`

TranAD:

- F1: `0.9411711188269714`
- Precision: `0.8888885596709039`
- Recall: `0.9999995833335069`

### MBA

TranAD-Ensemble + top-k:

- F1: `0.9792793675976177`
- Precision: `0.9594095905556841`
- Recall: `0.999999996153846`

TranAD:

- F1: `0.9789106612131879`
- Precision: `0.958702061361718`
- Recall: `0.999999996153846`

## Report Interpretation

The report does not claim universal improvement.

Correct interpretation:

- SMD: clear degradation, mainly recall drop.
- MSL, SWaT, NAB: unchanged.
- Synthetic, MBA: small improvement.
- Overall: partially effective but not robust across datasets.

Important phrasing:

- “preliminary stage result”
- “partially effective”
- “not yet robust across all datasets”
- “top-k may help localized anomalies but can miss distributed weak anomalies”

## Git / Workspace Notes

Untracked or modified items include:

- `Research/` added by user.
- `paper.pdf` added by user.
- `.idea/` untracked.
- `reports/tranad_e_topk/` added for report.
- `experiments/` added for new experiment scripts.
- `src/device.py`, `src/experiment_utils.py`, `src/tranad_variants.py` added.

Do not delete or revert user-added `Research/`.

