#!/bin/sh
# Portable M0--M4 multi-seed experiment runner (POSIX shell).

set -u
set -f

ROOT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR" || exit 1

PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-auto}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
MIN_EPOCHS=${MIN_EPOCHS:-5}
PATIENCE=${PATIENCE:-3}
REL_DELTA=${REL_DELTA:-0.005}
BASELINE_LOSS_RATIO=${BASELINE_LOSS_RATIO:-0.2}
# 正式跑五折实验时可手动修改为 SEEDS=(1 2 3 4 5)
# POSIX sh represents that list as the space-separated string below.
SEEDS=${SEEDS:-"1"}
DATASETS=${DATASETS:-"synthetic SMD SWaT SMAP MSL WADI MSDS UCR MBA NAB"}
MODELS=${MODELS:-"M0 M1 M2 M3 M4"}
RUNNER=experiments/run_progressive_ablation.py
AVAILABLE_DATASETS=
FAILURES=0

dataset_prefix() {
	case "$1" in
		SMD) printf '%s' machine-1-1_ ;;
		SMAP) printf '%s' P-1_ ;;
		MSL) printf '%s' C-1_ ;;
		UCR) printf '%s' 136_ ;;
		NAB) printf '%s' ec2_request_latency_system_failure_ ;;
		*) printf '%s' ;;
	esac
}

dataset_raw_dir() {
	case "$1" in
		MSL|SMAP) printf '%s' data/SMAP_MSL ;;
		*) printf 'data/%s' "$1" ;;
	esac
}

msds_labels_available() {
	"$PYTHON" -c 'import sys; from preprocess import msds_has_positive_labels; sys.exit(0 if msds_has_positive_labels() else 1)' >/dev/null 2>&1
}

dataset_available() {
	dataset=$1
	prefix=$(dataset_prefix "$dataset")
	folder="processed/$dataset"
	raw_folder=$(dataset_raw_dir "$dataset")
	if [ "$dataset" = MSDS ]; then
		[ -f "$raw_folder/train.csv" ] &&
		[ -f "$raw_folder/test.csv" ] &&
		[ -f "$raw_folder/labels.csv" ] &&
		msds_labels_available
		return $?
	fi
	[ -d "$raw_folder" ] &&
	[ -f "$folder/${prefix}train.npy" ] &&
	[ -f "$folder/${prefix}test.npy" ] &&
	[ -f "$folder/${prefix}labels.npy" ]
}

for dataset in $DATASETS; do
	if dataset_available "$dataset"; then
		AVAILABLE_DATASETS="$AVAILABLE_DATASETS $dataset"
	else
		printf '\033[93m[WARN] Dataset %s not found, skip.\033[0m\n' "$dataset" >&2
	fi
done

printf 'Device: %s | Max epochs: %s | Min epochs: %s | Batch size: %s | Patience: %s | Relative delta: %s | Baseline loss ratio: %s\n' \
	"$DEVICE" "$EPOCHS" "$MIN_EPOCHS" "$BATCH_SIZE" "$PATIENCE" "$REL_DELTA" "$BASELINE_LOSS_RATIO"
printf 'Seeds: %s\nDatasets: %s\nModels: %s\n' "$SEEDS" "$AVAILABLE_DATASETS" "$MODELS"

for seed in $SEEDS; do
	for dataset in $AVAILABLE_DATASETS; do
		for model in $MODELS; do
			printf '\n[seed=%s dataset=%s model=%s]\n' "$seed" "$dataset" "$model"
			"$PYTHON" "$RUNNER" \
				--mode train-test \
				--variant "$model" \
				--dataset "$dataset" \
				--seed "$seed" \
				--epochs "$EPOCHS" \
				--batch-size "$BATCH_SIZE" \
				--min-epochs "$MIN_EPOCHS" \
				--patience "$PATIENCE" \
				--rel-delta "$REL_DELTA" \
				--baseline-loss-ratio "$BASELINE_LOSS_RATIO" \
				--device "$DEVICE"
			status=$?
			if [ "$status" -ne 0 ]; then
				FAILURES=$((FAILURES + 1))
				printf 'Failed with exit status %s: %s %s seed %s\n' "$status" "$dataset" "$model" "$seed" >&2
			fi
		done
	done
done

if [ -f experiments_results/ablation_raw_metrics.csv ]; then
	"$PYTHON" aggregate_metrics.py
	status=$?
	if [ "$status" -ne 0 ]; then
		printf 'Metric aggregation failed with exit status %s.\n' "$status" >&2
		FAILURES=$((FAILURES + 1))
	fi
fi

if [ "$FAILURES" -ne 0 ]; then
	printf '\nCompleted with %s failed steps.\n' "$FAILURES" >&2
	exit 1
fi

printf '\nM0--M4 ablation runs completed.\nMetrics: experiments_results/ablation_raw_metrics.csv\n'
