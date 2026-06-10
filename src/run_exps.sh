#!/usr/bin/env bash
set -e

# Default options
BASE_MODELS="qwen3-1.7b"
RANKS="4,8,16,32,64"
METHODS="lora,qlora"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) BASE_MODELS="$2"; shift 2 ;;
    --ranks) RANKS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash run_exps.sh --models qwen3-1.7b --ranks 8,16 --methods lora,qlora"
      exit 1
      ;;
  esac
done

# Convert comma-separated strings to arrays
IFS=',' read -ra BASE_MODEL_ARRAY <<< "$BASE_MODELS"
IFS=',' read -ra RANK_ARRAY <<< "$RANKS"
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"

for BASE_MODEL in "${BASE_MODEL_ARRAY[@]}"; do
  for RANK in "${RANK_ARRAY[@]}"; do
    echo "================================================"
    echo "Base model: ${BASE_MODEL}, Rank: ${RANK}, Methods: ${METHODS}"
    echo "================================================"

    for METHOD in "${METHOD_ARRAY[@]}"; do
      CONFIG="src/configs/${METHOD}_${BASE_MODEL}.yaml"

      echo "Running method=${METHOD}, base_model=${BASE_MODEL}, rank=${RANK}, config=${CONFIG}"

      python -m src.train --config "${CONFIG}" --rank "${RANK}"
      python -m src.evaluate --config "${CONFIG}" --rank "${RANK}"
    done
  done
done