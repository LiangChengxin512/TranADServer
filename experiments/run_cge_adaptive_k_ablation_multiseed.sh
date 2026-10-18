#!/usr/bin/env bash

# Full C/G/E module and adaptive top-k ablation: 8 model configurations x 2 score settings.
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
# These are every dataset currently available under processed/ and supported by src/constants.py.
DATASETS=(MBA MSDS NAB MSL SMAP SMD SWaT UCR WADI synthetic)
MODELS=(TranAD TranAD_C TranAD_G TranAD_E TranAD_C_G TranAD_C_E TranAD_G_E TranAD_C_G_E)
SCORE_SETTINGS=(mean adaptive_topk)

CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiment_checkpoints/cge_adaptive_k_ablation}"
LOG_DIR="${LOG_DIR:-experiment_logs/cge_adaptive_k_ablation}"
TABLE_DIR="${TABLE_DIR:-experiment_tables/cge_adaptive_k_ablation}"
KEEP_GOING="${KEEP_GOING:-1}"
SUMMARY_STRICT="${SUMMARY_STRICT:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/tranad_mpl}"

PYTHON_RUN=(conda run --no-capture-output -n "$CONDA_ENV" python)
TRAIN_SCRIPT="experiments/train_tranad_variants.py"
TEST_SCRIPT="experiments/test_tranad_variants.py"
SUMMARY_SCRIPT="experiments/collect_cge_adaptive_k_ablation_results.py"

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then read -r -a SEEDS <<< "$SEEDS_OVERRIDE"; fi
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then read -r -a DATASETS <<< "$DATASETS_OVERRIDE"; fi
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
if [[ -n "${SCORE_SETTINGS_OVERRIDE:-}" ]]; then read -r -a SCORE_SETTINGS <<< "$SCORE_SETTINGS_OVERRIDE"; fi

FAILURES=()

print_command() {
	printf '>'
	for arg in "$@"; do printf ' %q' "$arg"; done
	printf '\n'
}

run_step() {
	local description="$1"
	shift
	printf '\n[%s]\n' "$description"
	print_command "$@"
	if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
	"$@"
	local status=$?
	if [[ $status -ne 0 ]]; then
		FAILURES+=("$description")
		printf 'FAILED (%s): %s\n' "$status" "$description" >&2
		if [[ "$KEEP_GOING" != "1" ]]; then exit "$status"; fi
	fi
	return "$status"
}

run_or_skip_step() {
	local output_path="$1"
	local description="$2"
	shift 2
	if [[ "$DRY_RUN" != "1" && "$FORCE" != "1" && -e "$output_path" ]]; then
		printf '\n[Skip existing] %s\nOutput exists: %s\n' "$description" "$output_path"
		return 0
	fi
	run_step "$description" "$@"
}

experiment_id() {
	local model_lower
	model_lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
	printf '%s_%s' "$model_lower" "$2"
}

printf 'Root: %s\nConda env: %s\n' "$ROOT_DIR" "$CONDA_ENV"
printf 'Seeds: %s\nDatasets: %s\nModels: %s\nScore settings: %s\n' "${SEEDS[*]}" "${DATASETS[*]}" "${MODELS[*]}" "${SCORE_SETTINGS[*]}"
printf 'Each model is evaluated once with mean aggregation and once with --score-agg topk --score-topk auto.\n'
printf 'Device: %s, batch size: %s, epochs: %s\n' "$DEVICE" "$BATCH_SIZE" "$EPOCHS"

for seed in "${SEEDS[@]}"; do
	for dataset in "${DATASETS[@]}"; do
		for model in "${MODELS[@]}"; do
			for score_setting in "${SCORE_SETTINGS[@]}"; do
				experiment="$(experiment_id "$model" "$score_setting")"
				checkpoint_root="$CHECKPOINT_DIR/seed_${seed}/$experiment"
				log_root="$LOG_DIR/seed_${seed}/$experiment"
				latest_checkpoint="$checkpoint_root/${model}_${dataset}/latest.ckpt"
				test_metrics="$log_root/${model}_${dataset}/test_metrics.json"
				train_command=("${PYTHON_RUN[@]}" "$TRAIN_SCRIPT" --model "$model" --dataset "$dataset" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" --device "$DEVICE" --checkpoint-dir "$checkpoint_root" --log-dir "$log_root" --seed "$seed")
				test_command=("${PYTHON_RUN[@]}" "$TEST_SCRIPT" --model "$model" --dataset "$dataset" --batch-size "$BATCH_SIZE" --device "$DEVICE" --checkpoint-dir "$checkpoint_root" --output-dir "$log_root" --seed "$seed")
				if [[ "$score_setting" == "adaptive_topk" ]]; then
					train_command+=(--score-agg topk --score-topk auto)
					test_command+=(--score-agg topk --score-topk auto)
				fi
				if [[ "$SKIP_TRAIN" != "1" ]]; then
					run_or_skip_step "$latest_checkpoint" "Train $model ($score_setting) on $dataset, seed $seed" "${train_command[@]}"
				fi
				if [[ "$SKIP_TEST" != "1" ]]; then
					run_or_skip_step "$test_metrics" "Test $model ($score_setting) on $dataset, seed $seed" "${test_command[@]}"
				fi
			done
		done
	done
done

summary_command=("${PYTHON_RUN[@]}" "$SUMMARY_SCRIPT" --log-dir "$LOG_DIR" --output-dir "$TABLE_DIR" --seeds "${SEEDS[@]}" --datasets "${DATASETS[@]}" --models "${MODELS[@]}" --score-settings "${SCORE_SETTINGS[@]}")
if [[ "$SUMMARY_STRICT" == "1" ]]; then summary_command+=(--strict); fi
run_step 'Render precision, recall, F1, and AUC mean+-std tables' "${summary_command[@]}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
	printf '\nCompleted with %s failed steps:\n' "${#FAILURES[@]}" >&2
	printf ' - %s\n' "${FAILURES[@]}" >&2
	exit 1
fi

printf '\nC/G/E plus adaptive-top-k ablation completed.\n'
printf 'Checkpoints: %s\nMetrics: %s\nTables: %s\n' "$CHECKPOINT_DIR" "$LOG_DIR" "$TABLE_DIR"
