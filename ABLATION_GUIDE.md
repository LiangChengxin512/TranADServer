# Progressive TranAD Ablation Guide (M0–M4)

This workflow compares a fixed TranAD baseline with four incremental additions.
It uses the repository's `processed/{dataset}/*.npy` files and automatically
selects CUDA, then MPS, then CPU when `--device auto` is used.

## Variants

| Variant | Components |
| --- | --- |
| M0 | Original TranAD and mean residual aggregation. |
| M1 | M0 plus parallel causal depthwise separable convolutions with kernels 1, 3, and 5. The normalized local branch is fused through LayerScale initialized to `1e-4`, preserving the input path at initialization. |
| M2 | M1 plus gated feature graph attention. The graph module returns its input unchanged when the feature count is one. |
| M3 | M2 plus two-phase FFT amplitude and focal-MSE reconstruction losses, each weighted by `0.1`. |
| M4 | M3 plus per-timestep adaptive top-k aggregation based on residual dispersion across features. It does not use test labels to select k. |

M0 is backed by the original TranAD implementation; the new modules are only
attached to the `TranAD_M0`–`TranAD_M4` classes.

## Code changes and hooks

- `src/tranad_progressive.py` adds the causal multi-scale embedding, gated graph
  attention, M3 loss functions, adaptive top-k scoring, and the M0–M4 classes.
- `src/models.py` resolves `TranAD_M0`–`TranAD_M4` lazily so the existing
  baseline imports and checkpoints remain available.
- `src/experiment_utils.py` recognizes the variants, adds M3 loss terms to the
  existing two-phase TranAD loss, applies M4's adaptive scoring, and uses basic
  time-ordered tensor slices for TranAD batches on MPS.
- `src/device.py` probes MPS allocation before selecting it and falls back to
  CPU in automatic mode if the probe fails.
- `experiments/run_progressive_ablation.py` trains or evaluates one
  dataset/variant/seed, reserves a temporal validation tail, applies early
  stopping, keeps the lowest-validation-loss checkpoint, and appends final
  evaluation metrics.
- `run_ablation.sh` runs the five seeds, available datasets, and all variants.
- `aggregate_metrics.py` groups raw metrics and writes mean ± sample standard
  deviation values.

## Output layout

```text
checkpoints/
└── {dataset}/
    └── {M0|M1|M2|M3|M4}/
        └── seed_{seed}/
            └── model_best.pt
experiment_logs/
└── {dataset}/{variant}/seed_{seed}/
    ├── train_metrics.csv
    ├── test_metrics.json
    └── test_per_dim.csv
experiments_results/
├── ablation_raw_metrics.csv
└── ablation_summary_mean_std.csv
```

## Run the full ablation

From the repository root:

```sh
sh run_ablation.sh
```

The defaults are seeds `1 2 3 4 5`, the ten datasets from `instruction.md`,
all five variants, 10 epochs, batch size 128, a 10% temporal validation tail,
and automatic device selection. Early stopping uses `patience=3` and
`min_delta=1e-4`; only the lowest-validation-loss checkpoint is kept at
`model_best.pt` and loaded for final test-set evaluation. Set `EPOCHS=10`–`15`
for a longer run, or pass `--epochs` to the single-run entrypoint.
The runner checks for the required processed train/test/label files. A missing
dataset prints a yellow warning and is skipped while later datasets continue.
After successful runs, the raw and summary CSVs are written automatically.

You can override the device, epochs, validation fraction, patience, minimum
improvement, seeds, datasets, or variants using environment variables. For
example:

```sh
DEVICE=cpu EPOCHS=1 SEEDS="1" DATASETS="UCR" MODELS="M0 M4" sh run_ablation.sh
```

The summary can also be regenerated independently:

```sh
python3 aggregate_metrics.py
```

## Reproduce one checkpoint

On Linux with CUDA, evaluate a saved M4 checkpoint directly:

```sh
python3 experiments/run_progressive_ablation.py \
  --mode test \
  --variant M4 \
  --dataset SMD \
  --seed 1 \
  --device cuda \
  --checkpoint checkpoints/SMD/M4/seed_1/model_best.pt
```

On macOS, run one quick UCR/M4 experiment with one epoch. `auto` probes MPS
and falls back to CPU if MPS cannot be initialized:

```sh
DEVICE=auto EPOCHS=1 SEEDS="1" DATASETS="UCR" MODELS="M4" sh run_ablation.sh
```

To explicitly choose MPS or CPU for a single model:

```sh
python3 experiments/run_progressive_ablation.py \
  --mode train-test --variant M4 --dataset UCR --seed 1 \
  --epochs 1 --batch-size 128 --device mps
```

Replace `mps` with `cpu` when using a CPU-only PyTorch installation.
