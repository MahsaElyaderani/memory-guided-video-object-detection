#!/usr/bin/env bash
set -eo pipefail

ACTION="${1:-all}"
INITIALIZATION="${2:-detector}"
if [[ $# -gt 2 || ! "$ACTION" =~ ^(all|train|eval)$ || ! "$INITIALIZATION" =~ ^(detector|imagenet)$ ]]; then
  echo "Usage: bash launchers/run_attention_experiments.sh [all|train|eval] [detector|imagenet]" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUNS="$(python path.py --get RUNS_DIR)"
ROOT="$RUNS/p4_student"
[[ "$INITIALIZATION" != imagenet ]] || ROOT="$RUNS/p4_student_imagenet"
BEST="$RUNS/efficientnet_tinynet_subset25_class5_linear_det100_cosine/seed_0/joint_efficientnet_b0_tinynet_d/detection_best.pt"
INIT="$ROOT/joint_init.pt"

if [[ "$ACTION" != eval ]]; then
  for MODE in none p4; do
    DIR="$ROOT/$MODE"
    SOURCE=()
    if [[ -f "$DIR/detection.pt" ]]; then
      SOURCE=(--checkpoint "$DIR/detection.pt")
    elif [[ "$INITIALIZATION" == detector ]]; then
      # The joint trainer replaces its best file atomically. Keep one version for both students.
      mkdir -p "$ROOT"
      [[ -f "$INIT" ]] || ln "$BEST" "$INIT"
      SOURCE=(--init-detector "$INIT")
    fi
    printf 'Training %s using %s\n' "$MODE" "${SOURCE[1]:-ImageNet backbone weights}"
    python -u train_attention_models.py train --memory "$MODE" --run-dir "$DIR" ${SOURCE[@]+"${SOURCE[@]}"} --device cuda
  done
fi

if [[ "$ACTION" != train ]]; then
  RESULTS="$(python path.py --get RESULTS_DIR)"
  mkdir -p "$RESULTS"
  OUT="$(mktemp -d "$RESULTS/attention_eval_XXXXXX")"
  for MODE in none p4; do
    BEST_STUDENT="$ROOT/$MODE/detection_best.pt"
    [[ -f "$BEST_STUDENT" ]] || { echo "Missing best checkpoint: $BEST_STUDENT" >&2; exit 1; }
    printf 'Evaluating %s using %s\n' "$MODE" "$BEST_STUDENT"
    python -u train_attention_models.py eval --memory "$MODE" --checkpoint "$BEST_STUDENT" --device cuda \
      --output-dir "$OUT/$MODE" --visuals
  done
  printf 'Evaluation results: %s\n' "$OUT"
fi
