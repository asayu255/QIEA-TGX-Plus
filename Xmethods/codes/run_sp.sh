#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_sp.sh [dataset] [model] [method] [factual] [existed] [sparsity] [gpu]
#
# dataset : wikipedia | reddit | movielens | enron
# model   : tgat | tgn
# method  : qiea | qiea-plus | ga | greedy | pg | random | tgnnex | all
# factual : 1 (factual) or 0 (counterfactual)
# existed : 1 (existing target) or 0 (non-existing target)
#
# Defaults match the paper's default setting: Wikipedia/TGAT, factual/existing,
# sparsity 0.2 (the same default as Xmethods/config/config.yaml).
dataset="${1:-wikipedia}"
model="${2:-tgat}"
method="${3:-tgnnex}"
factual="${4:-1}"
existed="${5:-1}"
val="${6:-0.2}"
gpu="${7:-1}"

case "${dataset}" in
  wikipedia|reddit|movielens|enron) ;;
  *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac
case "${model}" in
  tgat|tgn) ;;
  *) echo "Unsupported model: ${model}" >&2; exit 2 ;;
esac
case "${method}" in
  qiea|qiea-plus|ga|greedy|pg|random|tgnnex|all) ;;
  *) echo "Unsupported method: ${method}" >&2; exit 2 ;;
esac
case "${factual}" in 0|1) ;; *) echo "factual must be 0 or 1" >&2; exit 2 ;; esac
case "${existed}" in 0|1) ;; *) echo "existed must be 0 or 1" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/dataset"
OUTPUT_ROOT="${REPO_ROOT}/outputs"
cd "${SCRIPT_DIR}"

# The Python runners import the compatible TGN module even when TGAT is the
# selected model. Install the pinned source on first use if it is not present.
TGN_DIR="${REPO_ROOT}/TGNNmodels/xgraph/models/ext/tgn"
if [[ ! -f "${TGN_DIR}/model/tgn.py" ]]; then
    "${TGN_DIR}/setup.sh"
fi

# The canonical target-event location is dataset/test_data, which every runner
# now reads directly. Older checkouts may still hold targets under
# dataset/xgraph/test_data, so keep that path resolving to the same files.
mkdir -p "${DATASET_ROOT}/test_data" "${DATASET_ROOT}/xgraph"
legacy_target_dir="${DATASET_ROOT}/xgraph/test_data"
if [[ ! -e "${legacy_target_dir}" && ! -L "${legacy_target_dir}" ]]; then
    ln -s ../test_data "${legacy_target_dir}"
fi

if [[ "${factual}" -eq 1 ]]; then
    factual_flag="fact"
else
    factual_flag="cf"
fi
if [[ "${existed}" -eq 1 ]]; then
    existed_flag="existed"
else
    existed_flag="nonexisted"
fi
class_dir="${factual_flag}_${existed_flag}"
data_dir="${model}_${dataset}"

# All current runners write with np.savetxt and assume these leaf directories
# already exist. Create them here so a fresh checkout can run directly.
output_methods=("QIEA-TGX" "QIEA-TGX+" "GA-TGX" "GreeDy" "PGExplainer" "random" "T-GNNExplainer")
for output_method in "${output_methods[@]}"; do
    mkdir -p \
        "${OUTPUT_ROOT}/${output_method}/fidelity_time/${class_dir}/${data_dir}" \
        "${OUTPUT_ROOT}/${output_method}/loop/${class_dir}/${data_dir}"
done

common=(
    "datasets=${dataset}"
    "device_id=${gpu}"
    "models=${model}"
    "sparse_ratio=${val}"
    "factual=${factual}"
    "existed=${existed}"
)

run_qiea() {
    local plus_enabled="$1"
    python QIEA-TGX_run.py \
        "${common[@]}" \
        explainers=pg_explainer_tg \
        "qiea_tgx_plus.enabled=${plus_enabled}"
}

run_one() {
    case "$1" in
        qiea)      run_qiea false ;;
        qiea-plus) run_qiea true ;;
        ga)        python GA-TGX_run.py "${common[@]}" explainers=pg_explainer_tg ;;
        greedy)    python GreeDy_run.py "${common[@]}" explainers=pg_explainer_tg ;;
        pg)        python PGEx_run.py "${common[@]}" explainers=pg_explainer_tg ;;
        random)    python random_run.py "${common[@]}" explainers=pg_explainer_tg ;;
        tgnnex)    python TGNNEx_run.py "${common[@]}" explainers=subgraphx_tg ;;
    esac
}

if [[ "${method}" == "all" ]]; then
    for selected in qiea qiea-plus ga greedy pg random tgnnex; do
        echo "Running ${selected}: dataset=${dataset}, model=${model}, factual=${factual}, existed=${existed}, sparsity=${val}"
        run_one "${selected}"
    done
else
    echo "Running ${method}: dataset=${dataset}, model=${model}, factual=${factual}, existed=${existed}, sparsity=${val}"
    run_one "${method}"
fi
