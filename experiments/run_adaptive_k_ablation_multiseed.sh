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
SEEDS=(1 2 3 4 5)
DATASETS=(NAB MBA SWaT synthetic UCR MSL SMAP SMD)

EXISTING_LOG_DIR="${EXISTING_LOG_DIR:-experiment_logs/multiseed_adaptive_k}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiment_checkpoints/adaptive_k_ablation}"
LOG_DIR="${LOG_DIR:-experiment_logs/adaptive_k_ablation}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_tables/adaptive_k_ablation}"
KEEP_GOING="${KEEP_GOING:-1}"
SUMMARY_STRICT="${SUMMARY_STRICT:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/tranad_mpl}"

PYTHON_RUN=(conda run --no-capture-output -n "$CONDA_ENV" python)
TRAIN_SCRIPT="experiments/train_tranad_variants.py"
TEST_SCRIPT="experiments/test_tranad_variants.py"
COLLECT_SCRIPT="experiments/collect_adaptive_k_ablation_results.py"

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
	read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
fi
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
	return "$status"
}

run_or_skip_step() {
	local output_path="$1"
	local description="$2"
	shift 2
	if [[ "$DRY_RUN" != "1" && "$FORCE" != "1" && -e "$output_path" ]]; then
		printf '\n[Skip existing] %s\n' "$description"
		printf 'Output exists: %s\n' "$output_path"
		return 0
	fi
	run_step "$description" "$@"
}

train_and_test_variant() {
	local seed="$1"
	local dataset="$2"
	local variant_key="$3"
	local model="$4"
	local score_mode="$5"

	local seed_checkpoint_dir="$CHECKPOINT_DIR/seed_${seed}/${variant_key}"
	local seed_log_dir="$LOG_DIR/seed_${seed}/${variant_key}"
	local latest_checkpoint="$seed_checkpoint_dir/${model}_${dataset}/latest.ckpt"
	local test_metrics="$seed_log_dir/${model}_${dataset}/test_metrics.json"

	local train_command=(
		"${PYTHON_RUN[@]}" "$TRAIN_SCRIPT"
		--model "$model"
		--dataset "$dataset"
		--batch-size "$BATCH_SIZE"
		--epochs "$EPOCHS"
		--eval-every "$EVAL_EVERY"
		--device "$DEVICE"
		--checkpoint-dir "$seed_checkpoint_dir"
		--log-dir "$seed_log_dir"
		--seed "$seed"
	)
	local test_command=(
		"${PYTHON_RUN[@]}" "$TEST_SCRIPT"
		--model "$model"
		--dataset "$dataset"
		--batch-size "$BATCH_SIZE"
		--device "$DEVICE"
		--checkpoint-dir "$seed_checkpoint_dir"
		--output-dir "$seed_log_dir"
		--seed "$seed"
	)
	if [[ "$score_mode" == "adaptive_topk" ]]; then
		train_command+=(--score-agg topk --score-topk auto)
		test_command+=(--score-agg topk --score-topk auto)
	fi

	if [[ "$SKIP_TRAIN" != "1" ]]; then
		run_or_skip_step "$latest_checkpoint" "Train $model/$variant_key on $dataset with seed $seed" "${train_command[@]}"
	fi
	if [[ "$SKIP_TEST" != "1" ]]; then
		run_or_skip_step "$test_metrics" "Test $model/$variant_key on $dataset with seed $seed" "${test_command[@]}"
	fi
}

printf 'Root: %s\n' "$ROOT_DIR"
printf 'Conda env: %s\n' "$CONDA_ENV"
printf 'Seeds: %s\n' "${SEEDS[*]}"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Device: %s, batch size: %s, epochs: %s\n' "$DEVICE" "$BATCH_SIZE" "$EPOCHS"
printf 'Existing TranAD and TranAD_E adaptive-k logs: %s\n' "$EXISTING_LOG_DIR"
printf 'New ablation checkpoints: %s\n' "$CHECKPOINT_DIR"
printf 'New ablation logs: %s\n' "$LOG_DIR"
printf 'Ablation table output: %s\n' "$OUTPUT_DIR"

for seed in "${SEEDS[@]}"; do
	printf '\n=== Seed %s ===\n' "$seed"
	for dataset in "${DATASETS[@]}"; do
		train_and_test_variant "$seed" "$dataset" "tranad_adaptive_topk" "TranAD" "adaptive_topk"
		train_and_test_variant "$seed" "$dataset" "tranad_e_mean" "TranAD_E" "mean"
	done
done

collect_command=(
	"${PYTHON_RUN[@]}" "$COLLECT_SCRIPT"
	--datasets "${DATASETS[@]}"
	--seeds "${SEEDS[@]}"
	--existing-log-dir "$EXISTING_LOG_DIR"
	--ablation-log-dir "$LOG_DIR"
	--output-dir "$OUTPUT_DIR"
)
if [[ "$SUMMARY_STRICT" == "1" ]]; then
	collect_command+=(--strict)
fi
run_step "Collect adaptive-k ablation mean±std tables" "${collect_command[@]}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
	printf '\nCompleted with %s failed steps:\n' "${#FAILURES[@]}" >&2
	printf ' - %s\n' "${FAILURES[@]}" >&2
	printf '\nPartial artifacts may still be available in:\n' >&2
	printf ' - Checkpoints: %s\n' "$CHECKPOINT_DIR" >&2
	printf ' - Metrics: %s\n' "$LOG_DIR" >&2
	printf ' - Tables: %s\n' "$OUTPUT_DIR" >&2
	exit 1
fi

printf '\nAdaptive-k ablation workflow completed.\n'
printf 'Ablation metrics: %s/adaptive_k_ablation_metrics_long.csv\n' "$OUTPUT_DIR"
printf 'Ablation summary: %s/adaptive_k_ablation_summary_long.csv\n' "$OUTPUT_DIR"
printf 'Mean±std tables: %s/adaptive_k_ablation_*_mean_std.csv and .png\n' "$OUTPUT_DIR"
