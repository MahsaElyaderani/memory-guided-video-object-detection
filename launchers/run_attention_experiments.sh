#!/usr/bin/env bash
set -eo pipefail

ACTION="${1:-all}"
if [[ $# -gt 1 || ! "$ACTION" =~ ^(all|train|eval)$ ]]; then
  echo "Usage: bash launchers/run_attention_experiments.sh [all|train|eval]" >&2
  exit 2
fi

STOP_OPTIONS=()
case "${EARLY_STOPPING:-}" in
  1) STOP_OPTIONS=(--early-stopping) ;;
  0) STOP_OPTIONS=(--no-early-stopping) ;;
  '') ;;
  *) echo "EARLY_STOPPING must be 0 or 1" >&2; exit 2 ;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUNS="$(python utils/path.py --get RUNS_DIR)"
ROOT="$RUNS/p4_student_imagenet_no_projection"

experiment_dir() {
  local name="$1" old=none
  [[ "$name" != attention_memory ]] || old=p4
  if [[ ! -d "$ROOT/$name" && -d "$ROOT/$old" ]]; then
    printf '%s\n' "$ROOT/$old"
  else
    printf '%s\n' "$ROOT/$name"
  fi
}

if [[ "$ACTION" != eval ]]; then
  for MODE in no_memory attention_memory; do
    DIR="$(experiment_dir "$MODE")"
    SOURCE=()
    if [[ -f "$DIR/detection.pt" ]]; then
      SOURCE=(--checkpoint "$DIR/detection.pt")
    fi
    printf 'Training %s using %s\n' "$MODE" "${SOURCE[1]:-ImageNet backbone weights}"
    python -u train_attention_models.py train --memory "$MODE" --run-dir "$DIR" ${SOURCE[@]+"${SOURCE[@]}"} \
      ${STOP_OPTIONS[@]+"${STOP_OPTIONS[@]}"} --device cuda
  done
fi

if [[ "$ACTION" != train ]]; then
  RESULTS="$(python utils/path.py --get RESULTS_DIR)"
  mkdir -p "$RESULTS"
  OUT="$(mktemp -d "$RESULTS/attention_eval_XXXXXX")"
  for MODE in no_memory attention_memory; do
    BEST_STUDENT="$(experiment_dir "$MODE")/detection_best.pt"
    [[ -f "$BEST_STUDENT" ]] || { echo "Missing best checkpoint: $BEST_STUDENT" >&2; exit 1; }
    printf 'Evaluating %s using %s\n' "$MODE" "$BEST_STUDENT"
    python -u train_attention_models.py eval --memory "$MODE" --checkpoint "$BEST_STUDENT" --device cuda \
      --output-dir "$OUT/$MODE" --visuals
  done
  printf 'Evaluation results: %s\n' "$OUT"
fi
