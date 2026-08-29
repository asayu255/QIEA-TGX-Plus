#!/usr/bin/env bash
set -euo pipefail

# Generate target events and their two-hop neighborhoods for all four datasets.
# Usage: bash create_targets.sh [gpu]
#
# Targets and neighborhoods depend on the newly trained TGAT/TGN checkpoints,
# so this script is part of the from-scratch reproduction workflow.
gpu="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DATASET_ROOT}/.." && pwd)"
cd "${SCRIPT_DIR}"

datasets=("wikipedia" "reddit" "movielens" "enron")
models=("tgat" "tgn")

mkdir -p "${DATASET_ROOT}/test_data" "${DATASET_ROOT}/hops" "${DATASET_ROOT}/xgraph"

# create_targets.py and one_2hop.py import the compatible TGN class even for
# TGAT runs. Install the pinned source on first use if necessary.
TGN_DIR="${REPO_ROOT}/TGNNmodels/xgraph/models/ext/tgn"
if [[ ! -f "${TGN_DIR}/model/tgn.py" ]]; then
    "${TGN_DIR}/setup.sh"
fi

# Some legacy runners still refer to dataset/xgraph/test_data. Keep one
# canonical target directory and provide a compatibility symlink on fresh
# checkouts instead of maintaining a second copy of the target files.
legacy_target_dir="${DATASET_ROOT}/xgraph/test_data"
if [[ ! -e "${legacy_target_dir}" && ! -L "${legacy_target_dir}" ]]; then
    ln -s ../test_data "${legacy_target_dir}"
fi

for dataset in "${datasets[@]}"; do
    case "${dataset}" in
        wikipedia) hop_name="wiki" ;;
        reddit)    hop_name="redi" ;;
        movielens) hop_name="movielens" ;;
        enron)     hop_name="enron" ;;
    esac

    for model in "${models[@]}"; do
        for factual in 0 1; do
            for existed in 0 1; do
                if [[ "${factual}" -eq 1 && "${existed}" -eq 1 ]]; then
                    target_kind="existed_fac"
                    hop_class="fac_existed"
                elif [[ "${factual}" -eq 0 && "${existed}" -eq 1 ]]; then
                    target_kind="existed_cf"
                    hop_class="cf_existed"
                elif [[ "${factual}" -eq 1 && "${existed}" -eq 0 ]]; then
                    target_kind="nonexisted_fac"
                    hop_class="fac_nonexisted"
                else
                    target_kind="nonexisted_cf"
                    hop_class="cf_nonexisted"
                fi

                echo "Generating targets: dataset=${dataset}, model=${model}, factual=${factual}, existed=${existed}, gpu=${gpu}"
                python create_targets.py \
                    datasets="${dataset}" \
                    device_id="${gpu}" \
                    explainers=pg_explainer_tg \
                    models="${model}" \
                    factual="${factual}" \
                    existed="${existed}"

                generated="${SCRIPT_DIR}/new_test_${target_kind}_${model}_${dataset}.csv"
                canonical="${SCRIPT_DIR}/test_${target_kind}_${model}_${dataset}.csv"
                if [[ ! -f "${generated}" ]]; then
                    echo "Generated target file not found: ${generated}" >&2
                    exit 1
                fi
                mv -f "${generated}" "${canonical}"

                # If a user's old checkout already has a real legacy directory
                # rather than the compatibility symlink, keep it usable too.
                if [[ -d "${legacy_target_dir}" && ! -L "${legacy_target_dir}" ]]; then
                    cp -f "${canonical}" "${legacy_target_dir}/$(basename "${canonical}")"
                fi

                hop_dir="${DATASET_ROOT}/hops/${hop_class}/${model}_${hop_name}"
                mkdir -p "${hop_dir}"

                python one_2hop.py \
                    datasets="${dataset}" \
                    device_id="${gpu}" \
                    explainers=pg_explainer_tg \
                    models="${model}" \
                    factual="${factual}" \
                    existed="${existed}"
            done
        done
    done
done
