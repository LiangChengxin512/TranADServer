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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiment_checkpoints/multiseed_adaptive_k}"
LOG_DIR="${LOG_DIR:-experiment_logs/multiseed_adaptive_k}"
VISUALIZATION_DIR="${VISUALIZATION_DIR:-experiment_visualizations/multiseed_adaptive_k}"
TABLE_DIR="${TABLE_DIR:-experiment_tables/multiseed_adaptive_k}"
KEEP_GOING="${KEEP_GOING:-1}"
SUMMARY_STRICT="${SUMMARY_STRICT:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"
FIGURE_SEED="${FIGURE_SEED:-1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/tranad_mpl}"

PYTHON_RUN=(conda run --no-capture-output -n "$CONDA_ENV" python)
TRAIN_SCRIPT="experiments/train_tranad_variants.py"
TEST_SCRIPT="experiments/test_tranad_variants.py"
FIGURE_SCRIPT="experiments/visualize_paper_figures.py"
SUMMARY_SCRIPT="experiments/summarize_multiseed_results.py"

SEEDS=(1 2 3 4 5)
DATASETS=(NAB MBA SWaT synthetic UCR MSL SMAP SMD)
TRAIN_MODELS=(TranAD_E TranAD GDN DAGMM MSCRED MAD_GAN USAD MTAD_GAT CAE_M)
TEST_MODELS=(TranAD_E TranAD GDN DAGMM MSCRED MAD_GAN USAD MTAD_GAT CAE_M MERLIN)

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
	read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
fi
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
	read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
fi
if [[ -n "${TRAIN_MODELS_OVERRIDE:-}" ]]; then
	read -r -a TRAIN_MODELS <<< "$TRAIN_MODELS_OVERRIDE"
fi
if [[ -n "${TEST_MODELS_OVERRIDE:-}" ]]; then
	read -r -a TEST_MODELS <<< "$TEST_MODELS_OVERRIDE"
fi

FAILURES=()

canonical_model() {
	case "$1" in
		LSTM-NDT|LSTM_NDT) printf 'LSTM_AD' ;;
		MAD-GAN) printf 'MAD_GAN' ;;
		MTAD-GAT) printf 'MTAD_GAT' ;;
		CAE-M) printf 'CAE_M' ;;
		*) printf '%s' "$1" ;;
	esac
}

canonical_models=()
for model in "${TEST_MODELS[@]}"; do
	canonical_models+=("$(canonical_model "$model")")
done

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

printf 'Root: %s\n' "$ROOT_DIR"
printf 'Conda env: %s\n' "$CONDA_ENV"
printf 'Seeds: %s\n' "${SEEDS[*]}"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Training models: %s\n' "${TRAIN_MODELS[*]}"
printf 'Table models: %s\n' "${canonical_models[*]}"
printf 'Device: %s, batch size: %s, epochs: %s\n' "$DEVICE" "$BATCH_SIZE" "$EPOCHS"
printf 'TranAD_E uses adaptive top-k via: --score-agg topk --score-topk auto\n'

for seed in "${SEEDS[@]}"; do
	seed_checkpoint_dir="$CHECKPOINT_DIR/seed_${seed}"
	seed_log_dir="$LOG_DIR/seed_${seed}"
	printf '\n=== Seed %s ===\n' "$seed"

	if [[ "$SKIP_TRAIN" != "1" ]]; then
		for dataset in "${DATASETS[@]}"; do
			for model in "${TRAIN_MODELS[@]}"; do
				canonical="$(canonical_model "$model")"
				latest_checkpoint="$seed_checkpoint_dir/${canonical}_${dataset}/latest.ckpt"
				train_command=(
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
				if [[ "$canonical" == "TranAD_E" ]]; then
					train_command+=(--score-agg topk --score-topk auto)
				fi
				run_or_skip_step "$latest_checkpoint" "Train $canonical on $dataset with seed $seed" "${train_command[@]}"
			done
		done
	fi

	if [[ "$SKIP_TEST" != "1" ]]; then
		for dataset in "${DATASETS[@]}"; do
			for model in "${TEST_MODELS[@]}"; do
				canonical="$(canonical_model "$model")"
				test_metrics="$seed_log_dir/${canonical}_${dataset}/test_metrics.json"
				test_command=(
					"${PYTHON_RUN[@]}" "$TEST_SCRIPT"
					--model "$model"
					--dataset "$dataset"
					--batch-size "$BATCH_SIZE"
					--device "$DEVICE"
					--checkpoint-dir "$seed_checkpoint_dir"
					--output-dir "$seed_log_dir"
					--seed "$seed"
				)
				if [[ "$canonical" == "TranAD_E" ]]; then
					test_command+=(--score-agg topk --score-topk auto)
				fi
				run_or_skip_step "$test_metrics" "Test $canonical on $dataset with seed $seed" "${test_command[@]}"
			done
		done
	fi
done

if [[ "$SKIP_FIGURES" != "1" ]]; then
	figure_checkpoint_dir="$CHECKPOINT_DIR/seed_${FIGURE_SEED}"
	figure_output_dir="$VISUALIZATION_DIR/seed_${FIGURE_SEED}"
	for dataset in "${DATASETS[@]}"; do
		figure_png="$figure_output_dir/TranAD_E_${dataset}/figure3_focus_attention.png"
		figure_command=(
			"${PYTHON_RUN[@]}" "$FIGURE_SCRIPT"
			--model TranAD_E
			--dataset "$dataset"
			--batch-size "$BATCH_SIZE"
			--device "$DEVICE"
			--checkpoint-dir "$figure_checkpoint_dir"
			--output-dir "$figure_output_dir"
			--score-agg topk
			--score-topk auto
			--seed "$FIGURE_SEED"
		)
		run_or_skip_step "$figure_png" "Render TranAD_E adaptive-k paper-style figures on $dataset from seed $FIGURE_SEED" "${figure_command[@]}"
	done
fi

summary_command=(
	"${PYTHON_RUN[@]}" "$SUMMARY_SCRIPT"
	--log-dir "$LOG_DIR"
	--output-dir "$TABLE_DIR"
	--seeds "${SEEDS[@]}"
	--datasets "${DATASETS[@]}"
	--models "${canonical_models[@]}"
)
if [[ "$SUMMARY_STRICT" == "1" ]]; then
	summary_command+=(--strict)
fi
run_step "Render multi-seed mean±std metric tables" "${summary_command[@]}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
	printf '\nCompleted with %s failed steps:\n' "${#FAILURES[@]}" >&2
	printf ' - %s\n' "${FAILURES[@]}" >&2
	printf '\nPartial artifacts may still be available in:\n' >&2
	printf ' - Checkpoints: %s\n' "$CHECKPOINT_DIR" >&2
	printf ' - Metrics: %s\n' "$LOG_DIR" >&2
	printf ' - Figures: %s\n' "$VISUALIZATION_DIR" >&2
	printf ' - Tables: %s\n' "$TABLE_DIR" >&2
	exit 1
fi

printf '\nAdaptive-k multi-seed benchmark completed.\n'
printf 'Checkpoints: %s\n' "$CHECKPOINT_DIR"
printf 'Metrics: %s\n' "$LOG_DIR"
printf 'TranAD_E adaptive-k figures: %s\n' "$VISUALIZATION_DIR"
printf 'Mean±std tables: %s\n' "$TABLE_DIR"
