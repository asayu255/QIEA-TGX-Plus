#!/usr/bin/env bash
set -euo pipefail

# Usage: ./cpckpt.sh [dataset]
# Copies the best model produced by learn_edge.py to the canonical path used
# by the explainer runners.
dataset="${1:-reddit}"
model="tgat"

case "${dataset}" in
  wikipedia|reddit|movielens|enron) ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    echo "Choose one of: wikipedia, reddit, movielens, enron" >&2
    exit 2
    ;;
esac

source_path="./saved_models/tgat_${dataset}_best.pth"
target_dir="./../../checkpoints"
target_path="${target_dir}/${model}_${dataset}_best.pth"

if [[ ! -f "${source_path}" ]]; then
    echo "Checkpoint not found: ${source_path}" >&2
    echo "Run ./train.sh ${dataset} first." >&2
    exit 1
fi

mkdir -p "${target_dir}"
cp "${source_path}" "${target_path}"
echo "${source_path} -> ${target_path} copied"
