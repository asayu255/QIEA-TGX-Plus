import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from torch import positive

from TGNNmodels import ROOT_DIR


def check_wiki_reddit_dataformat(df):
    assert df.iloc[:, 0].min() == 0
    assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique() # 0, 1, 2, ...
    assert df.iloc[:, 1].min() == 0
    assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique() # 0, 1, 2, ...
    
    for col in ['u', 'i', 'ts', 'label']:
        assert col in df.columns.to_list()


def verify_dataframe_homogeneous(df):
    """Validate a homogeneous temporal graph such as Enron.

    Unlike Wikipedia/Reddit/MovieLens, sender and receiver IDs share the same
    node-ID space, so destination IDs must not be forced to start after the
    source-ID range.
    """
    for col in ['u', 'i', 'ts', 'label', 'e_idx', 'idx']:
        assert col in df.columns.to_list()

    assert len(df) > 0
    assert df['u'].min() >= 1
    assert df['i'].min() >= 1

    # process.py factorizes sender/receiver addresses jointly, so the shared
    # node IDs should form one contiguous 1-based range.
    node_ids = np.unique(
        np.concatenate((df['u'].to_numpy(), df['i'].to_numpy()))
    )
    assert node_ids.min() == 1
    assert node_ids.max() == len(node_ids)

    assert df['e_idx'].min() == 1
    assert df['e_idx'].max() == len(df)
    assert df['idx'].min() == 1
    assert df['idx'].max() == len(df)


def is_homogeneous_dataframe(df):
    """True when sources and destinations share one node-ID space.

    Bipartite datasets reindex destinations to start after the last source,
    so any overlap means the graph is homogeneous. Callers that do not know
    the dataset name rely on this.
    """
    sources = pd.unique(df.iloc[:, 0])
    destinations = pd.unique(df.iloc[:, 1])
    return np.intersect1d(sources, destinations).size > 0


def verify_dataframe_unify(df, dataset_name=None):
    """Validate the processed temporal-graph format.

    Wikipedia, Reddit and MovieLens use the original bipartite TGAT layout.
    Enron is homogeneous and therefore uses a shared source/destination node
    ID space.
    """
    if dataset_name == 'enron' or is_homogeneous_dataframe(df):
        verify_dataframe_homogeneous(df)
        return

    for col in ['u', 'i', 'ts', 'label', 'e_idx', 'idx']:
        assert col in df.columns.to_list()
    
    assert df.iloc[:, 0].min() == 1
    assert df.iloc[:, 1].min() == df.iloc[:, 0].max() + 1
    assert df.iloc[:, 1].max() == df.iloc[:, 0].max() + df.iloc[:, 1].nunique()
    assert df['e_idx'].min() == 1
    assert df['e_idx'].max() == len(df)
    assert df['idx'].min() == 1
    assert df['idx'].max() == len(df)

    
def load_events_data(path, dataset_name=None):
    df = pd.read_csv(path)
    verify_dataframe_unify(df, dataset_name=dataset_name)
    return df


def load_tg_dataset(dataset_name):
    data_dir = ROOT_DIR/'xgraph'/'models'/'ext'/'tgat'/'processed'
    df = pd.read_csv(data_dir/f'ml_{dataset_name}.csv')
    edge_feats = np.load(data_dir/f'ml_{dataset_name}.npy')
    node_feats = np.load(data_dir/f'ml_{dataset_name}_node.npy')

    verify_dataframe_unify(df, dataset_name=dataset_name)

    max_node_id = int(max(df.u.max(), df.i.max()))
    assert max_node_id + 1 == len(node_feats)
    assert df.e_idx.max() + 1 == len(edge_feats)

    # print
    if dataset_name == 'enron':
        n_nodes = len(np.unique(np.concatenate((df.u.values, df.i.values))))
        print(
            f"#Dataset: {dataset_name}, #Nodes: {n_nodes}, "
            f"#Interactions: {len(df)}, #Timestamps: {df.ts.nunique()}"
        )
    else:
        n_users = df.iloc[:, 0].max()
        n_items = df.iloc[:, 1].max() - df.iloc[:, 0].max()
        print(f"#Dataset: {dataset_name}, #Users: {n_users}, #Items: {n_items}, #Interactions: {len(df)}, #Timestamps: {df.ts.nunique()}")
    print(f'#node feats shape: {node_feats.shape}, #edge feats shape: {edge_feats.shape}')
    
    return df, edge_feats, node_feats


def load_explain_idx(explain_idx_filepath, start=0, end=None):
    df = pd.read_csv(explain_idx_filepath)
    event_idxs = df['event_idx'].to_list()
    if end is not None:
        event_idxs = event_idxs[start:end]
    else: event_idxs = event_idxs[start:]
    
    print(f'{len(event_idxs)} events to explain')

    return event_idxs



def generate_explain_index(file, explainer_idx_dir, dataset_name, explain_idx_name=None):
    df = pd.read_csv(file)
    verify_dataframe_unify(df, dataset_name=dataset_name)
    
    size = 500 # 100, 200, 300, 400, 500

    if dataset_name in ['simulate_v1', 'simulate_v2']:
        indices = df.label == 1
        explain_idxs = np.random.choice(df[indices].e_idx.values, size=size, replace=False)
    elif dataset_name in ['wikipedia', 'reddit', 'movielens', 'enron']:
        np.random.seed(1024)
        e_num = len(df)
        start_ratio = 0.7
        end_ratio = 0.99
        low = int(e_num*start_ratio)
        high = int(e_num*end_ratio)
        explain_idxs = np.random.randint(low=low, high=high, size=size)

    ############## save
    explain_idxs = sorted(explain_idxs)
    explain_idxs_dict = {
        'event_idx': explain_idxs, 
    }
    explain_idxs_df = pd.DataFrame(explain_idxs_dict)
    explainer_idx_dir = Path(explainer_idx_dir)
    explainer_idx_dir.mkdir(parents=True, exist_ok=True)
    if explain_idx_name is None:
        out_file = explainer_idx_dir/f'{dataset_name}.csv'
    else:
        out_file = explainer_idx_dir/f'{explain_idx_name}.csv'
    
    explain_idxs_df.to_csv(out_file, index=False)
    print(f'explain index file {str(out_file)} saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='wikipedia')
    parser.add_argument('-c', type=str, choices=['format', 'index'])
    args = parser.parse_args()

    data_dir = ROOT_DIR/'xgraph'/'models'/'ext'/'tgat'/'processed'
    explainer_idx_dir = ROOT_DIR/'..'/'dataset'/'explain_index'
    file = data_dir/f'ml_{args.d}.csv'

    if args.c == 'format':
        pass
    elif args.c == 'index':
        generate_explain_index(file, explainer_idx_dir, args.d)
    else:
        raise NotImplementedError



