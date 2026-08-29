from platform import node
from matplotlib import use
import argparse
from datetime import timezone
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd


REAL_DATASETS = {'wikipedia', 'reddit', 'movielens', 'enron'}
ENRON_FEATURE_DIM = 172
# The Enron corpus was collected in 2002 and covers roughly 1998-2002.
# Dates outside this window are clock errors, not interactions.
# TGAT builds its attention over concat(node, edge, time) with time as wide as
# the node features, so 2*node_dim + edge_dim must divide the head count.
# MovieLens ships a single rating per event, which would leave that width odd.
MOVIELENS_NODE_FEATURE_DIM = 172
MOVIELENS_EDGE_FEATURE_DIM = 2
ENRON_MIN_YEAR = 1998
ENRON_MAX_YEAR = 2002


def simulate_dataset_train_flag(df):
    labels = df['label'].to_numpy()
    mask = (labels == 1) | (labels == 0)
    return mask


def rename_columns_wiki_reddit(file):
    df = pd.read_csv(file, skiprows=1, header=None)
    feat_nums = df.shape[1] - 4
    new_columns = ['u', 'i', 'ts', 'label']

    for i in range(feat_nums):
        new_columns.append(f'f{i}')

    rename_dict = {i: new_columns[i] for i in range(len(new_columns))}
    df.rename(columns=rename_dict, inplace=True)
    df.to_csv(file, index=False)
    print(f'rename the columns of {file}.')


def reindex(df):
    """Reindex a bipartite graph (Wikipedia/Reddit/MovieLens)."""
    df = df.copy()
    df['i'] += df['u'].max() + 1
    df['u'] += 1
    df['i'] += 1
    df['e_idx'] = df.index.values + 1
    df['idx'] = df.e_idx
    return df


def reindex_non_bipartite(df):
    """Map source and destination addresses into one shared node-ID space.

    Enron is a homogeneous communication graph: the same email address can
    appear as both a sender and a recipient. This follows the non-bipartite
    reindexing used by BenchTemp, while also adding the e_idx column expected
    by the QIEA-TGX data loader.
    """
    df = df.copy().reset_index(drop=True)
    interaction_num = len(df)

    all_nodes = np.concatenate(
        (df['u'].to_numpy(), df['i'].to_numpy()),
        axis=0,
    )
    all_index, _ = pd.factorize(all_nodes, sort=False)

    df['u'] = all_index[:interaction_num] + 1
    df['i'] = all_index[interaction_num:] + 1
    df['idx'] = np.arange(1, interaction_num + 1)
    df['e_idx'] = df['idx']
    return df


def widen_features(features, width):
    """Zero-pad a feature matrix to `width` columns."""
    if features.shape[1] > width:
        raise ValueError(
            f'Cannot narrow features from {features.shape[1]} to {width} columns.'
        )
    if features.shape[1] == width:
        return features.astype(np.float32)
    widened = np.zeros((features.shape[0], width), dtype=np.float32)
    widened[:, :features.shape[1]] = features
    return widened


def find_movielens_ratings(data_dir, explicit_ratings=None):
    """Locate an official MovieLens ratings.csv file.

    MovieLens 32M/25M/latest distributions all use the columns
    userId,movieId,rating,timestamp. The explicit --ratings argument takes
    precedence; common extracted directory layouts are checked otherwise.
    """
    if explicit_ratings is not None:
        ratings_path = Path(explicit_ratings).expanduser().resolve()
        if not ratings_path.is_file():
            raise FileNotFoundError(f'MovieLens ratings file not found: {ratings_path}')
        return ratings_path

    candidates = [
        data_dir/'movielens.csv',
        data_dir/'ratings.csv',
        data_dir/'movielens'/'ratings.csv',
        data_dir/'ml-32m'/'ratings.csv',
        data_dir/'ml-25m'/'ratings.csv',
        data_dir/'ml-latest'/'ratings.csv',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidate_text = '\n'.join(f'  - {path}' for path in candidates)
    raise FileNotFoundError(
        'MovieLens ratings.csv was not found. Put it in one of:\n'
        f'{candidate_text}\n'
        'or pass --ratings /path/to/ratings.csv.'
    )


def parse_movielens_ratings(ratings_path):
    """Convert official MovieLens ratings into the repository's raw TG format.

    Users and movies are independently factorized to contiguous zero-based ID
    spaces, the rating is retained as the single edge feature f0, and the data
    is sorted chronologically before the standard bipartite reindexing step.
    """
    raw_df = pd.read_csv(ratings_path)
    official_columns = {'userId', 'movieId', 'rating', 'timestamp'}

    if official_columns.issubset(raw_df.columns):
        if len(raw_df) == 0:
            raise ValueError(f'MovieLens ratings file is empty: {ratings_path}')

        source = pd.to_numeric(raw_df['userId'], errors='raise')
        destination = pd.to_numeric(raw_df['movieId'], errors='raise')
        rating = pd.to_numeric(raw_df['rating'], errors='raise')
        timestamp = pd.to_numeric(raw_df['timestamp'], errors='raise')

        if source.isna().any() or destination.isna().any() or rating.isna().any() or timestamp.isna().any():
            raise ValueError('MovieLens ratings.csv contains missing values in required columns.')

        user_ids, users = pd.factorize(source, sort=False)
        movie_ids, movies = pd.factorize(destination, sort=False)

        df = pd.DataFrame({
            'u': user_ids.astype(np.int64),
            'i': movie_ids.astype(np.int64),
            'ts': timestamp.to_numpy(dtype=np.float64),
            'label': np.zeros(len(raw_df), dtype=np.float32),
            'f0': rating.to_numpy(dtype=np.float32),
        })

        # Official MovieLens ratings files are not globally time-sorted.
        df = df.sort_values('ts', kind='stable').reset_index(drop=True)
        print(
            f'MovieLens ratings parsed: interactions={len(df)}, '
            f'users={len(users)}, movies={len(movies)}'
        )
        return df

    # Preserve compatibility with an already prepared u/i/ts/label/f* file.
    required_columns = {'u', 'i', 'ts', 'label'}
    feature_columns = [c for c in raw_df.columns if c.startswith('f')]
    if required_columns.issubset(raw_df.columns) and feature_columns:
        df = raw_df.copy()
        for column in ['u', 'i', 'ts', 'label'] + feature_columns:
            df[column] = pd.to_numeric(df[column], errors='raise')

        # Normalize arbitrary original user/movie IDs only when needed.
        u_values = df['u'].to_numpy()
        i_values = df['i'].to_numpy()
        if not (
            df['u'].min() == 0
            and df['u'].max() + 1 == df['u'].nunique()
            and df['i'].min() == 0
            and df['i'].max() + 1 == df['i'].nunique()
        ):
            df['u'] = pd.factorize(u_values, sort=False)[0]
            df['i'] = pd.factorize(i_values, sort=False)[0]

        df = df.sort_values('ts', kind='stable').reset_index(drop=True)
        return df

    raise ValueError(
        f'Unsupported MovieLens input format in {ratings_path}. Expected official '
        'columns userId,movieId,rating,timestamp or prepared u,i,ts,label,f* columns.'
    )


def run_movielens(data_dir, out_dir, ratings=None):
    """Build ml_movielens.* directly from official MovieLens ratings.csv."""
    from dataset.tg_dataset import verify_dataframe_unify, check_wiki_reddit_dataformat

    ratings_path = find_movielens_ratings(data_dir, ratings)
    print(f'MovieLens ratings: {ratings_path}')

    df = parse_movielens_ratings(ratings_path)
    check_wiki_reddit_dataformat(df)

    new_df = reindex(df)
    verify_dataframe_unify(new_df, dataset_name='movielens')

    feature_columns = [c for c in df.columns if c.startswith('f')]
    edge_feat = np.zeros(
        (len(new_df) + 1, len(feature_columns)),
        dtype=np.float32,
    )
    edge_feat[1:, :] = df[feature_columns].to_numpy(dtype=np.float32)
    edge_feat = widen_features(edge_feat, MOVIELENS_EDGE_FEATURE_DIM)

    num_nodes = int(new_df['i'].max())
    node_feat = np.zeros(
        (num_nodes + 1, MOVIELENS_NODE_FEATURE_DIM),
        dtype=np.float32,
    )

    out_df = out_dir/'ml_movielens.csv'
    out_edge_feat = out_dir/'ml_movielens.npy'
    out_node_feat = out_dir/'ml_movielens_node.npy'

    print('dataset: movielens')
    _save_processed_dataset(
        new_df,
        edge_feat,
        node_feat,
        out_df,
        out_edge_feat,
        out_node_feat,
    )


def _normalize_email_address(address):
    """Return a conservative canonical form for an email address."""
    if address is None:
        return None

    address = address.strip().lower()
    if not address or '@' not in address:
        return None

    # Headers occasionally contain trailing punctuation around an address.
    address = address.strip("<> ,;'\"")
    if not address or '@' not in address:
        return None
    return address


def _header_addresses(message, header_names):
    """Extract valid email addresses from a collection of MIME headers."""
    raw_headers = []
    for header_name in header_names:
        raw_headers.extend(message.get_all(header_name, []))

    addresses = []
    seen = set()
    for _, address in getaddresses(raw_headers):
        normalized = _normalize_email_address(address)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        addresses.append(normalized)
    return addresses


def _message_timestamp(message):
    """Convert the RFC email Date header to a Unix timestamp.

    Malformed or missing Date headers are skipped rather than guessed, as are
    dates outside the corpus period (mail clients emit epoch-zero and
    far-future timestamps that would otherwise distort the chronological
    train/validation/test split).
    """
    date_header = message.get('Date')
    if date_header is None:
        return None

    try:
        dt = parsedate_to_datetime(str(date_header))
    except (TypeError, ValueError, OverflowError):
        return None

    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if not ENRON_MIN_YEAR <= dt.year <= ENRON_MAX_YEAR:
        return None

    try:
        return float(dt.timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def find_enron_maildir(data_dir, explicit_maildir=None):
    """Locate the raw Enron maildir.

    The explicit --maildir argument always wins. The fallback locations make
    common downloaded layouts work without requiring the corpus directory to
    be renamed.
    """
    if explicit_maildir is not None:
        maildir = Path(explicit_maildir).expanduser().resolve()
        if not maildir.is_dir():
            raise FileNotFoundError(f'Enron maildir not found: {maildir}')
        return maildir

    candidates = [
        data_dir/'maildir',
        data_dir/'enron_maildir',
        data_dir/'enron'/'maildir',
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    candidate_text = '\n'.join(f'  - {path}' for path in candidates)
    raise FileNotFoundError(
        'Enron raw maildir was not found. Put it in one of:\n'
        f'{candidate_text}\n'
        'or pass --maildir /path/to/maildir.'
    )


def parse_enron_maildir(maildir):
    """Parse raw Enron emails into sender-recipient temporal interactions.

    One email addressed to multiple recipients becomes multiple interactions.
    To, Cc, and Bcc are included when present. Messages without a valid Date,
    From, or recipient address are skipped. The result is intentionally a
    straightforward reconstruction rather than a byte-for-byte reproduction
    of any particular published Enron preprocessing.
    """
    parser = BytesParser(policy=policy.compat32)
    interactions = []

    parsed_files = 0
    skipped_files = 0

    for file_path in maildir.rglob('*'):
        if not file_path.is_file():
            continue

        parsed_files += 1
        try:
            with file_path.open('rb') as f:
                message = parser.parse(f, headersonly=True)
        except (OSError, ValueError):
            skipped_files += 1
            continue

        timestamp = _message_timestamp(message)
        senders = _header_addresses(message, ['From'])
        recipients = _header_addresses(message, ['To', 'Cc', 'Bcc'])

        if timestamp is None or not senders or not recipients:
            skipped_files += 1
            continue

        # RFC messages normally have one sender. If a malformed message lists
        # multiple From addresses, use the first valid address deterministically.
        sender = senders[0]
        for recipient in recipients:
            interactions.append((sender, recipient, timestamp))

        if parsed_files % 50000 == 0:
            print(
                f'parsed {parsed_files} mail files, '
                f'created {len(interactions)} interactions'
            )

    if len(interactions) == 0:
        raise ValueError(
            f'No valid Enron interactions were extracted from {maildir}.'
        )

    df = pd.DataFrame(interactions, columns=['u', 'i', 'ts'])
    df['label'] = 0.0

    # Temporal GNN datasets are consumed in chronological order.
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)

    unique_addresses = pd.unique(
        pd.concat([df['u'], df['i']], ignore_index=True)
    ).size
    print(
        f'Enron maildir parsed: files={parsed_files}, '
        f'skipped={skipped_files}, interactions={len(df)}, '
        f'unique_addresses={unique_addresses}'
    )
    return df


def verify_enron_dataframe(df):
    """Validate the homogeneous Enron format used by downstream models."""
    for col in ['u', 'i', 'ts', 'label', 'e_idx', 'idx']:
        assert col in df.columns.to_list()

    assert len(df) > 0
    assert df['u'].min() >= 1
    assert df['i'].min() >= 1
    assert df['e_idx'].min() == 1
    assert df['e_idx'].max() == len(df)
    assert df['idx'].min() == 1
    assert df['idx'].max() == len(df)


def _save_processed_dataset(new_df, edge_feat, node_feat, out_df, out_edge_feat, out_node_feat):
    print('edge feature shape: ', edge_feat.shape)
    print('node feature shape: ', node_feat.shape)

    new_df[['u', 'i', 'ts', 'label', 'idx', 'e_idx']].to_csv(out_df, index=False)
    np.save(out_edge_feat, edge_feat)
    np.save(out_node_feat, node_feat)

    print(f'{out_df} saved')
    print(f'{out_edge_feat} saved')
    print(f'{out_node_feat} saved')


def find_enron_benchtemp(data_dir, explicit_dir=None):
    """Locate the BenchTemp Enron release (ml_enron.csv/.npy/_node.npy).

    BenchTemp (ICDE 2024) publishes Enron already processed, in the same
    three-file layout this repository consumes, so no reconstruction from raw
    mail is needed. https://zenodo.org/records/8267825
    """
    candidates = [Path(explicit_dir)] if explicit_dir is not None else [
        data_dir/'benchtemp',
        data_dir/'benchtemp'/'enron',
        data_dir/'enron',
        data_dir,
    ]
    required = ('ml_enron.csv', 'ml_enron.npy', 'ml_enron_node.npy')
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if all((candidate/name).is_file() for name in required):
            return candidate.resolve()

    candidate_text = '\n'.join(f'  - {Path(c)}' for c in candidates)
    raise FileNotFoundError(
        'The BenchTemp Enron release was not found. Download ml_enron.csv, '
        'ml_enron.npy and ml_enron_node.npy from '
        'https://zenodo.org/records/8267825 into one of:\n'
        f'{candidate_text}\n'
        'or pass --benchtemp /path/to/dir (or --maildir to rebuild from raw mail).'
    )


def load_enron_benchtemp(source_dir):
    """Read the BenchTemp Enron files into this repository's conventions.

    Two shape differences are reconciled here: BenchTemp writes a pandas index
    column and names the event column ``idx`` only, and its edge features are
    32-dimensional while TGAT assumes node, time and edge features share one
    width. The published features are all zeros, so widening them to the node
    feature dimension loses nothing.
    """
    source_dir = Path(source_dir)
    df = pd.read_csv(source_dir/'ml_enron.csv')
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])
    df['e_idx'] = df['idx']

    edge_feat = np.load(source_dir/'ml_enron.npy')
    node_feat = np.load(source_dir/'ml_enron_node.npy')

    feat_dim = node_feat.shape[1]
    if edge_feat.shape[1] > feat_dim:
        raise ValueError(
            f'Enron edge features are wider ({edge_feat.shape[1]}) than node '
            f'features ({feat_dim}); TGAT requires a single feature width.'
        )
    edge_feat = widen_features(edge_feat, feat_dim)

    return df, edge_feat, node_feat.astype(np.float32)


def run_enron(data_dir, out_dir, maildir=None, benchtemp=None):
    """Build ml_enron.* from the BenchTemp release, or from raw mail.

    The BenchTemp release is the default because it is the dataset the
    reported Enron statistics come from. Passing --maildir instead rebuilds
    the graph from the raw CMU mail, which yields a much larger graph of every
    address seen in a header.
    """
    if maildir is None:
        source_dir = find_enron_benchtemp(data_dir, benchtemp)
        print(f'Enron (BenchTemp): {source_dir}')
        new_df, edge_feat, node_feat = load_enron_benchtemp(source_dir)
        verify_enron_dataframe(new_df)
        _save_processed_dataset(
            new_df, edge_feat, node_feat,
            out_dir/'ml_enron.csv', out_dir/'ml_enron.npy',
            out_dir/'ml_enron_node.npy',
        )
        return

    raw_maildir = find_enron_maildir(data_dir, maildir)
    print(f'Enron maildir: {raw_maildir}')

    df = parse_enron_maildir(raw_maildir)
    new_df = reindex_non_bipartite(df)
    verify_enron_dataframe(new_df)

    # BenchTemp-style Enron inputs use featureless communication events.
    # Keep the conventional TGN/TGAT feature width of 172 for compatibility
    # with the model configuration used by this repository.
    edge_feat = np.zeros(
        (len(new_df) + 1, ENRON_FEATURE_DIM),
        dtype=np.float32,
    )
    num_nodes = int(max(new_df['u'].max(), new_df['i'].max()))
    node_feat = np.zeros(
        (num_nodes + 1, ENRON_FEATURE_DIM),
        dtype=np.float32,
    )

    out_df = out_dir/'ml_enron.csv'
    out_edge_feat = out_dir/'ml_enron.npy'
    out_node_feat = out_dir/'ml_enron_node.npy'

    print('dataset: enron')
    _save_processed_dataset(
        new_df,
        edge_feat,
        node_feat,
        out_df,
        out_edge_feat,
        out_node_feat,
    )


def run(data_name, out_dir=None, maildir=None, ratings=None, benchtemp=None):
    from TGNNmodels import ROOT_DIR
    from dataset.tg_dataset import verify_dataframe_unify, check_wiki_reddit_dataformat

    data_dir = ROOT_DIR/'..'/'dataset'/'data'
    print(data_dir)

    if out_dir is None:
        out_dir = Path('./processed/')
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset-specific raw-data pipelines.
    if data_name == 'enron':
        run_enron(data_dir, out_dir, maildir=maildir, benchtemp=benchtemp)
        return
    if data_name == 'movielens':
        run_movielens(data_dir, out_dir, ratings=ratings)
        return

    data_path = data_dir/f'{data_name}.csv'
    OUT_DF = out_dir/f'ml_{data_name}.csv'
    OUT_EDGE_FEAT = out_dir/f'ml_{data_name}.npy'
    OUT_NODE_FEAT = out_dir/f'ml_{data_name}_node.npy'

    df = pd.read_csv(data_path)

    # Wikipedia/Reddit raw JODIE files may still have the original header.
    if 'comma_separated_list_of_features' in df.columns.tolist():
        rename_columns_wiki_reddit(data_path)
        df = pd.read_csv(data_path)

    check_wiki_reddit_dataformat(df)

    df = reindex(df)
    verify_dataframe_unify(df, dataset_name=data_name)
    new_df = df

    if data_name in ['simulate_v2', 'simulate_v1']:
        raise NotImplementedError
    elif data_name in REAL_DATASETS:
        select_columns = [c for c in new_df.columns if c.startswith('f')]
        if len(select_columns) == 0:
            raise ValueError(
                f'{data_name}.csv has no feature columns. Expected columns named f0, f1, ...'
            )

        edge_feat = np.zeros((len(df) + 1, len(select_columns)), dtype=np.float32)
        edge_feat[1:, :] = new_df[select_columns].to_numpy(dtype=np.float32)

        edge_feat_dim = edge_feat.shape[1]
        num_nodes = int(new_df.i.max())
        node_feat = np.zeros((num_nodes + 1, edge_feat_dim), dtype=np.float32)
    else:
        raise NotImplementedError(f'Unsupported dataset: {data_name}')

    assert len(node_feat) == new_df.i.max() + 1
    assert len(edge_feat) == len(new_df) + 1

    print('dataset: ', data_name)
    _save_processed_dataset(
        new_df,
        edge_feat,
        node_feat,
        OUT_DF,
        OUT_EDGE_FEAT,
        OUT_NODE_FEAT,
    )


def process_garden_5():
    from TGNNmodels import ROOT_DIR
    data_dir = ROOT_DIR/'xgraph'/'dataset'/'data'
    data_path = data_dir/'garden_5.csv'
    df = pd.read_csv(data_path)
    if 'label' not in df.columns.to_list():
        df['label'] = np.ones((len(df),))
        df.to_csv(data_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='simulate')
    parser.add_argument(
        '--maildir',
        type=str,
        default=None,
        help=(
            'Path to the raw Enron maildir. If omitted, process.py checks '
            'dataset/data/maildir, dataset/data/enron_maildir, and '
            'dataset/data/enron/maildir.'
        ),
    )
    parser.add_argument(
        '--benchtemp',
        type=str,
        default=None,
        help=(
            'Directory holding the BenchTemp Enron release (ml_enron.csv, '
            'ml_enron.npy, ml_enron_node.npy). Default for -d enron.'
        ),
    )
    parser.add_argument(
        '--ratings',
        type=str,
        default=None,
        help=(
            'Path to official MovieLens ratings.csv. If omitted, process.py '
            'checks common locations under dataset/data, including ml-32m.'
        ),
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Optional processed-data output directory.',
    )
    parser.add_argument(
        '-rename_w_r',
        action='store_true',
        help='rename columns of wikipedia and reddit',
    )
    args = parser.parse_args()
    dataset = args.data

    run(
        dataset,
        out_dir=args.out_dir,
        maildir=args.maildir,
        ratings=args.ratings,
        benchtemp=args.benchtemp,
    )
