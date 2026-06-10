#!/usr/bin/env bash
set -euo pipefail

# Default options
CONFIG="src/configs/qwen3.yaml"
MODELS="qwen3_1p7b"
RANKS="4,8,16,32,64"
METHODS="lora,qlora"
LEARNING_RATE=""

TRAIN_ONLY=false
EVAL_ONLY=false

print_usage() {
  echo "Usage:"
  echo "  bash src/run_exps.sh --models qwen3_8b --ranks 16 --methods lora,qlora"
  echo ""
  echo "Options:"
  echo "  --config         Config file path. Default: src/configs/qwen3.yaml"
  echo "  --models         Comma-separated model keys, e.g. qwen3_1p7b,qwen3_4b,qwen3_8b"
  echo "  --ranks          Comma-separated ranks, e.g. 4,8,16,32,64"
  echo "  --methods        Comma-separated methods, e.g. lora,qlora"
  echo "  --learning_rate  Override learning rate, e.g. 1e-4"
  echo "  --train-only     Only run training"
  echo "  --eval-only      Only run evaluation"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --ranks)
      RANKS="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --train-only)
      TRAIN_ONLY=true
      shift
      ;;
    --eval-only)
      EVAL_ONLY=true
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

if [[ "${TRAIN_ONLY}" == true && "${EVAL_ONLY}" == true ]]; then
  echo "Error: --train-only and --eval-only cannot be used together."
  exit 1
fi

# Convert comma-separated strings to arrays
IFS=',' read -ra MODEL_ARRAY <<< "${MODELS}"
IFS=',' read -ra RANK_ARRAY <<< "${RANKS}"
IFS=',' read -ra METHOD_ARRAY <<< "${METHODS}"

echo "================================================"
echo "Config:        ${CONFIG}"
echo "Models:        ${MODELS}"
echo "Methods:       ${METHODS}"
echo "Ranks:         ${RANKS}"
echo "Learning rate: ${LEARNING_RATE:-default}"
echo "Mode:          train_only=${TRAIN_ONLY}, eval_only=${EVAL_ONLY}"
echo "================================================"

for MODEL in "${MODEL_ARRAY[@]}"; do
  for METHOD in "${METHOD_ARRAY[@]}"; do
    for RANK in "${RANK_ARRAY[@]}"; do
      echo "================================================"
      echo "Model:  ${MODEL}"
      echo "Method: ${METHOD}"
      echo "Rank:   ${RANK}"
      echo "LR:     ${LEARNING_RATE:-default}"
      echo "================================================"

      if [[ "${EVAL_ONLY}" == false ]]; then
        echo "[Train] ${MODEL} ${METHOD} r=${RANK}"

        TRAIN_CMD=(
          python -m src.train
          --config "${CONFIG}"
          --model_key "${MODEL}"
          --method "${METHOD}"
          --rank "${RANK}"
        )

        if [[ -n "${LEARNING_RATE}" ]]; then
          TRAIN_CMD+=(--learning_rate "${LEARNING_RATE}")
        fi

        "${TRAIN_CMD[@]}"
      fi

      if [[ "${TRAIN_ONLY}" == false ]]; then
        echo "[Evaluate] ${MODEL} ${METHOD} r=${RANK}"

        python -m src.evaluate \
          --config "${CONFIG}" \
          --model_key "${MODEL}" \
          --method "${METHOD}" \
          --rank "${RANK}"
      fi
    done
  done
done

echo "Done."