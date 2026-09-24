#!/usr/bin/env python3
"""Build the fixed kNN graph, optimizer graph and spectral initialization of each dataset, then bind them.

Every variant and optimizer seed of a dataset starts from the same inputs:

  results/01_fixed_inputs/<dataset>/neighbors_15/
      knn_indices.npy  knn_distances.npy  fuzzy_graph_csr.npz  optimizer_graph_csr.npz
      init_embedding.npy  metadata.json

They are built once with ``IBUMAP.prepare_fixed_inputs`` (umap-learn kNN and
fuzzy graph, the optimizer graph pruned for ``n_epochs`` = 200, spectral
initialization; seed 42, ``n_jobs=1``; parameters in ``experiment.yaml``,
section ``fixed_inputs``). Existing inputs are reused when their recorded
parameters and hashes match. The per-suite binding
``results/<suite>/01_inputs/<dataset>/binding.json`` records the hashes that
all later stages verify.
"""
import time

from _common import (Context, parser, np, sparse, read_json, write_json, sha, digest, validate_inputs,
                     save_array, save_sparse, repo_relative, now_utc)
from common.dataset_io import load_processed_dataset
from common.fixed_inputs import build_fixed_inputs, fixed_input_paths

FILES = ('knn_indices', 'knn_distances', 'fuzzy_graph', 'optimizer_graph', 'init_embedding')


def canonical_csr(matrix):
    csr = matrix.tocsr().astype(np.float32, copy=False)
    csr.sum_duplicates()
    csr.sort_indices()
    return csr


def fixed_input_record(ctx, d, features_path, feature_shape):
    params = dict(ctx.e['fixed_inputs'])
    return {'dataset': d['name'], 'feature_shape': [int(v) for v in feature_shape],
            'feature_sha256': sha(features_path), 'params': params,
            'params_hash': digest(params)}


def prepare(ctx, d, dataset, features):
    paths = fixed_input_paths(ctx.fixed_root, d['name'], 15)
    expected = fixed_input_record(ctx, d, dataset.features_path, features.shape)
    if paths['metadata'].exists() and not ctx.args.overwrite:
        existing = read_json(paths['metadata'])
        if existing.get('status') == 'ok' and all(existing.get(k) == v for k, v in expected.items()):
            if all(paths[k].exists() and sha(paths[k]) == existing['hashes'][paths[k].name] for k in FILES):
                ctx.log.info('SKIP matching fixed inputs %s', d['name'])
                return paths
        raise RuntimeError(f"Fixed inputs of {d['name']} differ from the configuration; pass --overwrite")
    params = dict(ctx.e['fixed_inputs'])
    started = time.perf_counter()
    # A writable in-memory float32 copy, as pynndescent's compiled kernels reject read-only arrays.
    x = np.array(features, dtype=np.float32, order='C', copy=True)
    if x.ndim > 2:
        x = x.reshape(len(x), -1)
    bundle = build_fixed_inputs(x, params, device=str(params.get('device', 'cpu')))
    if bundle.optimizer_graph is None or bundle.init_embedding is None:
        raise RuntimeError('IBUMAP did not return the optimizer graph and the initialization')
    fuzzy = canonical_csr(bundle.fuzzy_graph)
    optimizer = canonical_csr(bundle.optimizer_graph)
    init = np.asarray(bundle.init_embedding, dtype=np.float32, order='C')
    validate_inputs(optimizer, init, d['n'])
    save_array(paths['knn_indices'], np.asarray(bundle.knn_indices, dtype=np.int64))
    save_array(paths['knn_distances'], np.asarray(bundle.knn_distances, dtype=np.float32))
    save_sparse(paths['fuzzy_graph'], fuzzy)
    save_sparse(paths['optimizer_graph'], optimizer)
    save_array(paths['init_embedding'], init)
    write_json(paths['metadata'], {
        **expected, 'status': 'ok', 'created_at': now_utc(),
        'features_path': repo_relative(dataset.features_path),
        'hashes': {paths[k].name: sha(paths[k]) for k in FILES},
        'fuzzy_graph_nnz': int(fuzzy.nnz), 'optimizer_graph_nnz': int(optimizer.nnz),
        'preparation': {'timings': bundle.timings, 'diagnostics': bundle.diagnostics,
                        'effective_config': bundle.effective_config,
                        'wall_seconds': time.perf_counter() - started},
        'environment': ctx.env})
    ctx.log.info('BUILT %s optimizer_graph_nnz=%d seconds=%.2f', d['name'], optimizer.nnz,
                 time.perf_counter() - started)
    return paths


def main():
    args = parser(__doc__.splitlines()[0]).parse_args()
    ctx = Context(args, '01_inputs')
    for d in ctx.datasets():
        ctx.log.info('PREPARE %s N=%d run=%s', d['name'], d['n'], args.run)
        if not args.run:
            continue
        try:
            dataset = load_processed_dataset(ctx.processed / d['name'])
            features = dataset.load_features(mmap=True)
            if features.ndim > 2:
                features = features.reshape(len(features), -1)
            if features.shape[0] != d['n']:
                raise ValueError(f"Processed feature rows {features.shape[0]} differ from the manifest N={d['n']}")
            prepare(ctx, d, dataset, features)
            files = ctx.source_files(d['name'])
            meta = read_json(files['metadata.json'])
            if meta['params'].get('n_epochs') != ctx.e['n_epochs']:
                raise ValueError('The optimizer graph is pruned for n_epochs; use the experiment n_epochs')
            for key, expected in ctx.e['common_params'].items():
                if key != 'umap_epsilon' and meta['params'].get(key) != expected:
                    raise ValueError(f"Fixed-input parameter differs: {key}: {meta['params'].get(key)} != {expected}")
            graph = sparse.load_npz(files['optimizer_graph_csr.npz']).tocsr()
            y = np.load(files['init_embedding.npy'])
            validate_inputs(graph, y, d['n'])
            hashes = {k: sha(path) for k, path in files.items()}
            feature_hash = meta['feature_sha256']
            record = {'dataset': d, 'hashes': hashes, 'feature_hash': feature_hash,
                      'feature_shape': list(features.shape), 'graph_nnz': int(graph.nnz),
                      'fixed_inputs': repo_relative(files['metadata.json'].parent)}
            record['input_key'] = digest({'hashes': {k: v for k, v in hashes.items() if k != 'metadata.json'},
                                          'features': feature_hash})
            output = ctx.binding(d['name'])
            if output.exists() and not args.overwrite:
                if read_json(output)['input_key'] != record['input_key']:
                    raise RuntimeError('Input binding changed; pass --overwrite')
                ctx.log.info('SKIP matching binding %s', d['name'])
                continue
            write_json(output, record)
            ctx.log.info('BOUND %s graph_nnz=%d', d['name'], graph.nnz)
        except Exception as exc:
            ctx.error(ctx.binding(d['name']).parent, exc)
    ctx.finish()


if __name__ == '__main__':
    main()
