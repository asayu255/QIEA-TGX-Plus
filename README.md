# QIEA-TGX+

QIEA-TGX+ is a multi-agent extension of QIEA-TGX for post-hoc explanation of temporal graph neural networks using a quantum-inspired evolutionary algorithm. The original QIEA-TGX implementation is also included as the base method.

Supported datasets and models:

- Datasets: Wikipedia, Reddit, MovieLens, Enron
- Models: TGAT, TGN
- Methods: QIEA-TGX+, QIEA-TGX, GA-TGX, GreeDy, PGExplainer, Random, T-GNNExplainer

# How to run

Run the following commands from the repository root unless otherwise noted.

```bash
export REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p dataset/data
```

## 1. Prepare datasets

### Wikipedia / Reddit

Download the JODIE files and place them at:

```text
dataset/data/wikipedia.csv
dataset/data/reddit.csv
```

- Wikipedia: http://snap.stanford.edu/jodie/wikipedia.csv
- Reddit: http://snap.stanford.edu/jodie/reddit.csv

### MovieLens

Download MovieLens 32M:

- https://grouplens.org/datasets/movielens/32m/
- https://files.grouplens.org/datasets/movielens/ml-32m.zip

```bash
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip -P dataset/data
unzip dataset/data/ml-32m.zip -d dataset/data
```

The default expected file is:

```text
dataset/data/ml-32m/ratings.csv
```

### Enron

The default workflow uses the BenchTemp Enron release:

- https://zenodo.org/records/8267825

Place these files together:

```text
dataset/data/benchtemp/
├── ml_enron.csv
├── ml_enron.npy
└── ml_enron_node.npy
```

To use a raw CMU Enron maildir instead, pass `--maildir /path/to/maildir` when preprocessing.

## 2. Preprocess datasets

```bash
cd "${REPO_ROOT}/TGNNmodels/xgraph/models/ext/tgat"

python process.py -d wikipedia
python process.py -d reddit
python process.py -d movielens
python process.py -d enron
```

Optional custom paths:

```bash
python process.py -d movielens --ratings /path/to/ratings.csv
python process.py -d enron --benchtemp /path/to/benchtemp
python process.py -d enron --maildir /path/to/maildir
```

Processed files are written under:

```text
TGNNmodels/xgraph/models/ext/tgat/processed/
```

## 3. Generate explain indices

```bash
cd "${REPO_ROOT}/dataset"

python tg_dataset.py -d wikipedia -c index
python tg_dataset.py -d reddit -c index
python tg_dataset.py -d movielens -c index
python tg_dataset.py -d enron -c index
```

## 4. Train TGAT / TGN

### TGAT

```bash
cd "${REPO_ROOT}/TGNNmodels/xgraph/models/ext/tgat"
bash train.sh <dataset> <gpu> <runs>
bash cpckpt.sh <dataset>
```

Example:

```bash
bash train.sh wikipedia 0 1
bash cpckpt.sh wikipedia
```

### TGN

```bash
cd "${REPO_ROOT}/TGNNmodels/xgraph/models/ext/tgn"
bash train.sh <dataset> <gpu> <runs> <seed>
bash cpckpt.sh <dataset> <seed>
```

Example:

```bash
bash train.sh wikipedia 0 1 123
bash cpckpt.sh wikipedia 123
```

Canonical checkpoints are copied to:

```text
TGNNmodels/xgraph/models/checkpoints/
```

## 5. Generate target events and neighborhoods

After training the TGNN checkpoints:

```bash
cd "${REPO_ROOT}/dataset/test_data"
bash create_targets.sh 0
```

This generates targets and neighborhoods for all four datasets, TGAT/TGN, factual/counterfactual, and existing/non-existing settings.

## 6. Run explainers

```bash
cd "${REPO_ROOT}/Xmethods/codes"
bash run_sp.sh <dataset> <model> <method> <factual> <existed> <sparsity> <gpu>
```

Arguments:

```text
dataset : wikipedia | reddit | movielens | enron
model   : tgat | tgn
method  : qiea-plus | qiea | ga | greedy | pg | random | tgnnex | all
factual : 1 = factual, 0 = counterfactual
existed : 1 = existing, 0 = non-existing
sparsity: default 0.2
gpu     : GPU ID
```

Examples:

```bash
# QIEA-TGX+
bash run_sp.sh wikipedia tgat qiea-plus 1 1 0.2 0

# Original QIEA-TGX
bash run_sp.sh wikipedia tgat qiea 1 1 0.2 0

# All methods
bash run_sp.sh wikipedia tgat all 1 1 0.2 0
```

Results are written under `outputs/`.

# Reference

```bibtex
@misc{ohara2026explainer,
  title        = {QIEA-TGX+: Post-hoc Explainer for Temporal Graph Neural Networks with Quantum-Inspired Evolution},
  author       = {Ohara, Rikuya and Mitani, Masahiro and Sasaki, Yuya},
  year         = {2026}
}
```
