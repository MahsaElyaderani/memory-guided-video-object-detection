#!/usr/bin/env bash
# Finish the organized full-data runs, then evaluate best checkpoints at interval 10.
set -eo pipefail
ACTION="${1:-all}"
if [[ $# -gt 1 || ! "$ACTION" =~ ^(all|train|eval)$ ]]; then
  echo "Usage: bash launchers/run_full_experiments.sh [all|train|eval]" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
RUNS="$(python utils/path.py --get RUNS_DIR)"
RESULTS="$(python utils/path.py --get RESULTS_DIR)"
ROOT="$RUNS/experiments"
MANIFEST="$ROOT/manifests/val100x20_seed7.json"
[[ -f "$MANIFEST" ]] || { echo "Missing validation manifest: $MANIFEST" >&2; exit 1; }

if [[ "$ACTION" != eval ]]; then
  [[ -f "$ROOT/train.sh" && -f "$ROOT/manifests/train_100pct.json" ]] || {
    echo "Copy the organized train.sh and full-training manifest into $ROOT first." >&2; exit 1;
  }
  # The helper preserves the source 25% checkpoint and resumes the new full-data run.
  # Do not retrain the completed joint EfficientNet-LSTM model.
  for EXP in joint_120d small_050 big_120d small_tinynet_d big_efficientnet_b0; do
    bash "$ROOT/train.sh" "$EXP"
  done
  # Both attention variants use full data; the previous 25% no_memory run is untouched.
  TRAIN_SUBSET=1 RUN_NAME=train100_seed0 EARLY_STOPPING=1 \
    bash launchers/run_attention_experiments.sh train
fi

if [[ "$ACTION" != train ]]; then
  mkdir -p "$RESULTS"
  OUT="$(mktemp -d "$RESULTS/full_validation_interval10_XXXXXX")"
  printf '\nFull-validation results: %s\n' "$OUT"
  while read -r EXP MODE RELATIVE_DIR; do
    CHECKPOINT="$ROOT/$RELATIVE_DIR/detection_best.pt"
    [[ -f "$CHECKPOINT" ]] || { echo "Missing best checkpoint: $CHECKPOINT" >&2; exit 1; }
    python -u train_lstm_models.py eval --checkpoint "$CHECKPOINT" \
      --mode "$MODE" --interval 10 --score-threshold 0 --max-detections 100 \
      --warmup-frames 10 --backbone-benchmark-steps 200 --no-losses --no-visuals \
      --device cuda --output-dir "$OUT/$EXP"
  done <<'MODELS'
small_050 big lstm/mobilenetv2/small/train100_seed0
big_120d big lstm/mobilenetv2/big/train100_seed0
joint_120d fixed lstm/mobilenetv2/joint/train25_to100_seed0
small_tinynet_d big lstm/efficientnet/small/train100_seed0
big_efficientnet_b0 big lstm/efficientnet/big/train100_seed0
joint_efficientnet fixed lstm/efficientnet/joint/train25_to50_to100_seed0
MODELS
  for MODEL in no_memory attention_memory; do
    CHECKPOINT="$ROOT/attention/tinynet_d/$MODEL/train100_seed0/detection_best.pt"
    [[ -f "$CHECKPOINT" ]] || { echo "Missing best checkpoint: $CHECKPOINT" >&2; exit 1; }
    python -u train_attention_models.py eval --memory "$MODEL" --checkpoint "$CHECKPOINT" \
      --manifest "$MANIFEST" --full-validation --device cuda --output-dir "$OUT/$MODEL"
  done
  printf '\nCompleted evaluations: %s\n' "$OUT"
fi
