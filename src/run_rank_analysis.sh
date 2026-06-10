#!/usr/bin/env bash
set -euo pipefail

MODELS=("qwen3_1p7b")
METHODS=("lora" "qlora")
RANKS=(4 8 16 32 64)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --methods)
      shift
      METHODS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        METHODS+=("$1")
        shift
      done
      ;;
    --ranks)
      shift
      RANKS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        RANKS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

for MODEL in "${MODELS[@]}"; do
  OUTPUT_ROOT="src/outputs/${MODEL}"

  echo "============================================================"
  echo "Model: ${MODEL}"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Methods: ${METHODS[*]}"
  echo "Ranks: ${RANKS[*]}"
  echo "============================================================"

  for METHOD in "${METHODS[@]}"; do
    for RANK in "${RANKS[@]}"; do
      ADAPTER_DIR="${OUTPUT_ROOT}/${METHOD}_r${RANK}/adapter"
      OUTPUT_JSON="${OUTPUT_ROOT}/${METHOD}_r${RANK}/effective_rank.json"

      echo "------------------------------------------------------------"
      echo "Analyzing ${MODEL} ${METHOD} r=${RANK}"
      echo "Adapter: ${ADAPTER_DIR}"
      echo "------------------------------------------------------------"

      if [[ ! -d "${ADAPTER_DIR}" ]]; then
        echo "[Warning] Adapter directory not found: ${ADAPTER_DIR}"
        echo "Skipping..."
        continue
      fi

      python -m src.compute_effective_rank \
        --adapter "${ADAPTER_DIR}" \
        --output "${OUTPUT_JSON}"
    done
  done

  echo "------------------------------------------------------------"
  echo "Summarizing effective rank for ${MODEL}"
  echo "------------------------------------------------------------"

  python -m src.summarize_rank \
    --output_root "${OUTPUT_ROOT}" \
    --methods "${METHODS[@]}" \
    --ranks "${RANKS[@]}"
done

echo "Done."