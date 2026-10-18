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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiment_checkpoints}"
LOG_DIR="${LOG_DIR:-experiment_logs}"
VISUALIZATION_DIR="${VISUALIZATION_DIR:-experiment_visualizations}"
TABLE_DIR="${TABLE_DIR:-experiment_tables/tranad_e_table2}"
KEEP_GOING="${KEEP_GOING:-0}"
TABLE_STRICT="${TABLE_STRICT:-1}"
DRY_RUN="${DRY_RUN:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/tranad_mpl}"

PYTHON_RUN=(conda run --no-capture-output -n "$CONDA_ENV" python)
TRAIN_SCRIPT="experiments/train_tranad_variants.py"
TEST_SCRIPT="experiments/test_tranad_variants.py"
FIGURE_SCRIPT="experiments/visualize_paper_figures.py"
TABLE_SCRIPT="experiments/render_table2_results.py"

# TranAD uses the Numenta Anomaly Benchmark name NAB. There is no NBA dataset
# in the repository.
DATASETS=(NAB MBA SWaT synthetic UCR MSL SMAP SMD)
#TRAIN_MODELS=(TranAD_E TranAD GDN DAGMM MSCRED LSTM-NDT MAD-GAN USAD MTAD-GAT CAE-M)
TRAIN_MODELS=(TranAD_E TranAD GDN DAGMM MSCRED MAD-GAN USAD MTAD-GAT CAE-M)

# Test/table logs use the canonical repository class names after CLI aliases are
# normalized by the train/test scripts.
#TEST_MODELS=(TranAD_E TranAD GDN DAGMM MSCRED LSTM_AD MAD_GAN USAD MTAD_GAT CAE_M MERLIN)
TEST_MODELS=(TranAD_E TranAD GDN DAGMM MSCRED MAD_GAN USAD MTAD_GAT CAE_M MERLIN)

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

printf 'Root: %s\n' "$ROOT_DIR"
printf 'Conda env: %s\n' "$CONDA_ENV"
printf 'Datasets: %s\n' "${DATASETS[*]}"
printf 'Training models: %s\n' "${TRAIN_MODELS[*]}"
printf 'Table models: %s\n' "${TEST_MODELS[*]}"
printf 'Device: %s, batch size: %s, epochs: %s\n' "$DEVICE" "$BATCH_SIZE" "$EPOCHS"

for dataset in "${DATASETS[@]}"; do
	for model in "${TRAIN_MODELS[@]}"; do
		train_command=(
			"${PYTHON_RUN[@]}" "$TRAIN_SCRIPT"
			--model "$model"
			--dataset "$dataset"
			--batch-size "$BATCH_SIZE"
			--epochs "$EPOCHS"
			--eval-every "$EVAL_EVERY"
			--device "$DEVICE"
			--checkpoint-dir "$CHECKPOINT_DIR"
			--log-dir "$LOG_DIR"
		)
		if [[ "$model" == "TranAD_E" ]]; then
			train_command+=(--score-agg topk --score-topk 3)
		fi
		run_step "Train $model on $dataset" "${train_command[@]}"
	done
done

for dataset in "${DATASETS[@]}"; do
	figure_command=(
		"${PYTHON_RUN[@]}" "$FIGURE_SCRIPT"
		--model TranAD_E
		--dataset "$dataset"
		--batch-size "$BATCH_SIZE"
		--device "$DEVICE"
		--checkpoint-dir "$CHECKPOINT_DIR"
		--output-dir "$VISUALIZATION_DIR"
		--score-agg topk
		--score-topk 3
	)
	run_step "Render TranAD_E paper-style figures on $dataset" "${figure_command[@]}"
done

for dataset in "${DATASETS[@]}"; do
	for model in "${TEST_MODELS[@]}"; do
		test_command=(
			"${PYTHON_RUN[@]}" "$TEST_SCRIPT"
			--model "$model"
			--dataset "$dataset"
			--batch-size "$BATCH_SIZE"
			--device "$DEVICE"
			--checkpoint-dir "$CHECKPOINT_DIR"
			--output-dir "$LOG_DIR"
		)
		if [[ "$model" == "TranAD_E" ]]; then
			test_command+=(--score-agg topk --score-topk 3)
		fi
		run_step "Test $model on $dataset" "${test_command[@]}"
	done
done

table_command=(
	"${PYTHON_RUN[@]}" "$TABLE_SCRIPT"
	--log-dir "$LOG_DIR"
	--output-dir "$TABLE_DIR"
	--models "${TEST_MODELS[@]}"
	--datasets "${DATASETS[@]}"
)
if [[ "$TABLE_STRICT" == "1" ]]; then
	table_command+=(--strict)
fi
run_step "Render Table 2-style metric tables" "${table_command[@]}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
	printf '\nCompleted with %s failed steps:\n' "${#FAILURES[@]}" >&2
	printf ' - %s\n' "${FAILURES[@]}" >&2
	exit 1
fi

printf '\nBenchmark workflow completed.\n'
printf 'Checkpoints: %s\n' "$CHECKPOINT_DIR"
printf 'Test metrics: %s\n' "$LOG_DIR"
printf 'TranAD_E figures: %s\n' "$VISUALIZATION_DIR"
printf 'Table 2-style tables: %s\n' "$TABLE_DIR"
