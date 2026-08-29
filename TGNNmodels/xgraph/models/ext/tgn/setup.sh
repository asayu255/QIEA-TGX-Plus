#!/usr/bin/env bash
set -euo pipefail

# Install the T-GNNExplainer-compatible TGN source used by this project.
# The source is pinned so the setup is reproducible and does not depend on the
# moving head of the upstream repository.
SOURCE_REPO="https://github.com/cisaic/tgnnexplainer.git"
SOURCE_COMMIT="f467ccfd6dd4b23566ae0c3ff7ada6dbf3b555d8"
SOURCE_SUBDIR="TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/ext/tgn"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="${SCRIPT_DIR}/.source_commit"

if [[ -f "${SCRIPT_DIR}/model/tgn.py" && -f "${MARKER}" ]] && \
   [[ "$(cat "${MARKER}")" == "${SOURCE_COMMIT}" ]]; then
    echo "Compatible TGN source is already installed (${SOURCE_COMMIT})."
    exit 0
fi

command -v git >/dev/null 2>&1 || {
    echo "git is required to install the TGN source." >&2
    exit 1
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

git -C "${tmp_dir}" init -q
git -C "${tmp_dir}" remote add origin "${SOURCE_REPO}"
git -C "${tmp_dir}" fetch -q --depth 1 origin "${SOURCE_COMMIT}"
git -C "${tmp_dir}" checkout -q FETCH_HEAD -- "${SOURCE_SUBDIR}"
source_dir="${tmp_dir}/${SOURCE_SUBDIR}"

rm -rf \
    "${SCRIPT_DIR}/model" \
    "${SCRIPT_DIR}/modules" \
    "${SCRIPT_DIR}/utils" \
    "${SCRIPT_DIR}/evaluation"

cp -R "${source_dir}/model" "${SCRIPT_DIR}/model"
cp -R "${source_dir}/modules" "${SCRIPT_DIR}/modules"
cp -R "${source_dir}/utils" "${SCRIPT_DIR}/utils"
cp -R "${source_dir}/evaluation" "${SCRIPT_DIR}/evaluation"
cp "${source_dir}/train_self_supervised.py" "${SCRIPT_DIR}/train_self_supervised.py"
cp "${source_dir}/LICENSE" "${SCRIPT_DIR}/LICENSE"

# The original T-GNNExplainer source lives under the package name
# `tgnnexplainer`. This repository keeps the model under `TGNNmodels`, so
# rewrite package-qualified imports while leaving the algorithm unchanged.
while IFS= read -r -d '' py_file; do
    sed -i \
        -e 's/from tgnnexplainer\.xgraph\.models\.ext\.tgn/from TGNNmodels.xgraph.models.ext.tgn/g' \
        -e 's/from tgnnexplainer\.xgraph\.models\.ext\.tgat/from TGNNmodels.xgraph.models.ext.tgat/g' \
        -e 's/from tgnnexplainer import ROOT_DIR/from TGNNmodels import ROOT_DIR/g' \
        "${py_file}"
done < <(find "${SCRIPT_DIR}/model" "${SCRIPT_DIR}/modules" "${SCRIPT_DIR}/utils" "${SCRIPT_DIR}/evaluation" -type f -name '*.py' -print0)

# Training script uses local imports and is intentionally run from this
# directory, matching the original TGN training workflow.
if grep -R -nE '^(from|import) tgnnexplainer' \
    "${SCRIPT_DIR}/model" "${SCRIPT_DIR}/modules" "${SCRIPT_DIR}/utils" "${SCRIPT_DIR}/evaluation"; then
    echo "Unconverted tgnnexplainer import remains in installed TGN source." >&2
    exit 1
fi

printf '%s\n' "${SOURCE_COMMIT}" > "${MARKER}"
echo "Installed compatible TGN source from ${SOURCE_COMMIT}."
