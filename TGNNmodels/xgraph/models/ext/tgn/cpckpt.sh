#!/usr/bin/env bash
set -euo pipefail

# Usage: ./cpckpt.sh [dataset] [seed]
dataset="${1:-reddit}"
seed="${2:-123}"
model="tgn"

case "${dataset}" in
  wikipedia|reddit|movielens|enron) ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    echo "Choose one of: wikipedia, reddit, movielens, enron" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source_path="./saved_models/tgn_${dataset}_${seed}_best.pth"
target_dir="./../../checkpoints"
target_path="${target_dir}/${model}_${dataset}_best.pth"

if [[ ! -f "${source_path}" ]]; then
    echo "Checkpoint not found: ${source_path}" >&2
    echo "Run ./train.sh ${dataset} <gpu> <runs> ${seed} first." >&2
    exit 1
fi

mkdir -p "${target_dir}"
cp "${source_path}" "${target_path}"
echo "${source_path} -> ${target_path} copied"
