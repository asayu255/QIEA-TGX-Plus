#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train.sh [dataset] [gpu] [runs]
# dataset: wikipedia, reddit, movielens, enron
# Defaults preserve the previous public setup (reddit, GPU 1, three runs).
dataset="${1:-reddit}"
gpu="${2:-1}"
runs="${3:-3}"

case "${dataset}" in
  wikipedia|reddit|movielens|enron) ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    echo "Choose one of: wikipedia, reddit, movielens, enron" >&2
    exit 2
    ;;
esac

mkdir -p log saved_models saved_checkpoints

for ((i=0; i<runs; i++)); do
    echo "${i}-th run: dataset=${dataset}, gpu=${gpu}"
    python learn_edge.py \
        -d "${dataset}" \
        --bs 512 \
        --n_degree 10 \
        --n_epoch 10 \
        --agg_method attn \
        --attn_mode prod \
        --gpu "${gpu}" \
        --n_head 2 \
        --prefix "${dataset}"
done
