#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train.sh [dataset] [gpu] [runs] [seed]
# dataset: wikipedia, reddit, movielens, enron
# The model/training hyperparameters match the real-dataset TGN setup used by
# T-GNNExplainer: two layers/heads, 10 neighbors, 10 epochs, memory enabled.
dataset="${1:-reddit}"
gpu="${2:-1}"
runs="${3:-1}"
seed="${4:-123}"

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

./setup.sh
mkdir -p log results saved_models saved_checkpoints

for ((i=0; i<runs; i++)); do
    echo "${i}-th run: dataset=${dataset}, gpu=${gpu}, seed=${seed}"
    python train_self_supervised.py \
        -d "${dataset}" \
        --prefix tgn-attn \
        --bs 200 \
        --n_layer 2 \
        --n_head 2 \
        --n_degree 10 \
        --n_epoch 10 \
        --lr 0.0001 \
        --drop_out 0.1 \
        --node_dim 172 \
        --time_dim 172 \
        --message_dim 172 \
        --memory_dim 172 \
        --patience 5 \
        --n_runs 1 \
        --use_memory \
        --embedding_module graph_attention \
        --message_function identity \
        --aggregator last \
        --memory_updater gru \
        --gpu "${gpu}" \
        --seed "${seed}"
done
