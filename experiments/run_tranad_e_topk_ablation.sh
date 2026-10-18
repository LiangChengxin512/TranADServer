#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-AnomalyDetection}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
EVAL_EVERY="${EVAL_EVERY:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiment_checkpoints/ablation_tranad_e_topk}"
LOG_DIR="${LOG_DIR:-experiment_logs/ablation_tranad_e_topk}"
EXISTING_LOG_DIR="${EXISTING_LOG_DIR:-experiment_logs}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_tables/tranad_e_ablation}"
KEEP_GOING="${KEEP_GOING:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/tranad_mpl}"

PYTHON_RUN=(conda run --no-capture-output -n "$CONDA_ENV" python)
TRAIN_SCRIPT="experiments/train_tranad_variants.py"
TEST_SCRIPT="experiments/test_tranad_variants.py"
COLLECT_SCRIPT="experiments/collect_tranad_e_ablation_results.py"

# Use the repository dataset name `synthetic`; `Synthetic` is a display name.
DATASETS=(NAB UCR MBA SMD SMAP MSL SWaT synthetic)

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
	read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
fi

FAILURES=()

print_command() {
	printf '>'
	for arg in "$@"; do
		printf ' %q' "$arg"
	done
	printf '\n'
}

run_step() {
	local description="$1"
	shift
	printf '\n[%s]\n' "$description"
	print_command "$@"
	if [[ "$DRY_RUN" == "1" ]]; then
		return 0
	fi
	"$@"
	local status=$?
	if [[ $status -ne 0 ]]; then
		FAILURES+=("$description")
		printf 'FAILED (%s): %s\n' "$status" "$description" >&2
		if [[ "$KEEP_GOING" != "1" ]]; then
			exit "$status"
		fi
	fi
}

maybe_run_step() {
	local output_file="$1"
	local description="$2"
	shift 2
	if [[ "$FORCE" != "1" && -f "$output_file" ]]; then
		printf '\n[%s]\n' "$description"
		printf 'Skip existing output: %s\n' "$output_file"
		return 0
	fi
	run_step "$description" "$@"
}

train_and_test_tranad_topk() {
	local dataset="$1"
	local ckpt_root="$CHECKPOINT_DIR/tranad_topk"
	local log_root="$LOG_DIR/tranad_topk"
	local test_json="$log_root/TranAD_${dataset}/test_metrics.json"
	local train_command=(
		"${PYTHON_RUN[@]}" "$TRAIN_SCRIPT"
		--model TranAD
		--dataset "$dataset"
		--epochs "$EPOCHS"
		--batch-size "$BATCH_SIZE"
		--eval-every "$EVAL_EVERY"
		--device "$DEVICE"
		--checkpoint-dir "$ckpt_root"
		--log-dir "$log_root"
		--score-agg topk
		--score-topk 3
	)
	local test_command=(
		"${PYTHON_RUN[@]}" "$TEST_SCRIPT"
		--model TranAD
		--dataset "$dataset"
		--batch-size "$BATCH_SIZE"
		--device "$DEVICE"
		--checkpoint-dir "$ckpt_root"
		--output-dir "$log_root"
		--score-agg topk
		--score-topk 3
	)
	maybe_run_step "$test_json" "Train TranAD+topk on $dataset" "${train_command[@]}"
	maybe_run_step "$test_json" "Test TranAD+topk on $dataset" "${test_command[@]}"
}

train_and_test_tranad_e_mean() {
	local dataset="$1"
	local ckpt_root="$CHECKPOINT_DIR/tranad_e_mean"
	local log_root="$LOG_DIR/tranad_e_mean"
	local test_json="$log_root/TranAD_E_${dataset}/test_metrics.json"
	local train_command=(
		"${PYTHON_RUN[@]}" "$TRAIN_SCRIPT"
		--model TranAD_E
		--dataset "$dataset"
		--epochs "$EPOCHS"
		--batch-size "$BATCH_SIZE"
		--eval-every "$EVAL_EVERY"
		--device "$DEVICE"
		--checkpoint-dir "$ckpt_root"
		--log-dir "$log_root"
	)
	local test_command=(
		"${PYTHON_RUN[@]}" "$TEST_SCRIPT"
		--model TranAD_E
		--dataset "$dataset"
		--batch-size "$BATCH_SIZE"
		--device "$DEVICE"
		--checkpoint-dir "$ckpt_root"
		--output-dir "$log_root"
	)
	maybe_run_step "$test_json" "Train TranAD_E without topk on $dataset" "${train_command[@]}"
	maybe_run_step "$test_json" "Test TranAD_E without topk on $dataset" "${test_command[@]}"
}

printf 'Root: %s\n' "$ROOT_DIR"
printf 'Conda env: %s\n' "$CONDA_ENV"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Device: %s, batch size: %s, epochs: %s\n' "$DEVICE" "$BATCH_SIZE" "$EPOCHS"
printf 'New ablation checkpoints: %s\n' "$CHECKPOINT_DIR"
printf 'New ablation logs: %s\n' "$LOG_DIR"
printf 'Existing baseline/final logs used for collection: %s\n' "$EXISTING_LOG_DIR"

for dataset in "${DATASETS[@]}"; do
	train_and_test_tranad_topk "$dataset"
	train_and_test_tranad_e_mean "$dataset"
done

collect_command=(
	"${PYTHON_RUN[@]}" "$COLLECT_SCRIPT"
	--datasets "${DATASETS[@]}"
	--existing-log-dir "$EXISTING_LOG_DIR"
	--ablation-log-dir "$LOG_DIR"
	--output-dir "$OUTPUT_DIR"
)
run_step "Collect ablation P/R/F1/AUC CSV files" "${collect_command[@]}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
	printf '\nCompleted with %s failed steps:\n' "${#FAILURES[@]}" >&2
	printf ' - %s\n' "${FAILURES[@]}" >&2
	exit 1
fi

printf '\nAblation workflow completed.\n'
printf 'Ablation metrics CSV: %s/ablation_metrics_long.csv\n' "$OUTPUT_DIR"
printf 'Wide metric CSVs: %s/ablation_precision.csv, %s/ablation_recall.csv, %s/ablation_f1.csv, %s/ablation_roc_auc.csv\n' "$OUTPUT_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR"
