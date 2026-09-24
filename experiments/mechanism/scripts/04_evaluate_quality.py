#!/usr/bin/env python3
"""Score the final embeddings with the paper's five quality metrics (configs/evaluation.yaml).

Trustworthiness, continuity and neighborhood preservation (k = 15) use all rows;
random triplet accuracy and distance Spearman correlation use 10^6 sampled
triplets / pairs (seed 42). The asynchronous control is not scored. Embeddings
that are bitwise identical (the seed-independent dense variants) reuse one score.
"""
import shutil
from _common import (Context, parser, np, sha, digest, complete, commit_output,
                     write_json, run_key, repo_relative)
from common.dataset_io import load_processed_dataset
from common.evaluation_cache import (request_from_config, cache_store_from_config,
                                     source_cache_key, embedding_cache_key, summarize_evaluation_scores,
                                     evaluation_device_status, scalar_scores)
from common.gpu_runtime import force_cleanup


def main():
    p = parser(__doc__)
    p.add_argument('--device', choices=['cpu', 'gpu', 'auto'])
    args = p.parse_args()
    ctx = Context(args, '04_quality')
    config = dict(ctx.cfg['evaluation'])
    if args.device:
        config['device'] = args.device
    request = request_from_config(config)
    resolved_device = evaluation_device_status(request.device)
    cache_config = {**config, 'cache': {**config.get('cache', {}),
                    'root': str(ctx.results / '04_quality/metric_cache')}}
    store = cache_store_from_config(cache_config, base_dir=ctx.config_dir,
                                   default_root=ctx.results / '04_quality/metric_cache')
    for d in ctx.datasets():
        if not args.run:
            ctx.log.info('PLAN quality %s seeds=%s algorithms=%s', d['name'], ctx.seeds(d),
                         [a['name'] for a in ctx.algorithms(d) if a.get('evaluate', True)])
            continue
        try:
            graph, init, binding = ctx.load(d)
            dataset = load_processed_dataset(ctx.processed / d['name'])
            if sha(dataset.features_path) != binding['feature_hash']:
                raise ValueError('Features changed after binding')
            x = np.asarray(dataset.load_features(mmap=True), dtype=request.dtype)
            if x.ndim > 2:
                x = x.reshape(len(x), -1)
            source_key = source_cache_key(dataset_id=d['name'], n_rows=len(x), n_features=x.shape[1],
                                          request=request, sample_indices=np.arange(len(x)),
                                          extra={'feature_hash': binding['feature_hash']})
        except Exception as exc:
            ctx.error(ctx.results / '04_quality' / d['name'], exc)
            continue
        reused = {}
        for seed in ctx.seeds(d):
            for a in ctx.algorithms(d):
                if not a.get('evaluate', True):
                    continue
                directory = ctx.results / '04_quality' / d['name'] / f"{a['name']}__seed_{seed}"
                try:
                    run_dir = ctx.run_dir(d['name'], a['name'], seed)
                    if not complete(run_dir, run_key(ctx, binding, a, seed)):
                        raise ValueError(f'Missing successful optimizer output: {run_dir}')
                    emb_hash = sha(run_dir / 'embedding.npy')
                    key = digest({'run_key': run_key(ctx, binding, a, seed),
                                  'embedding': emb_hash, 'evaluation': config, 'features': binding['feature_hash']})
                    if complete(directory, key, args.overwrite):
                        reused[emb_hash] = directory
                        continue
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / 'metadata.json').unlink(missing_ok=True)
                    if emb_hash in reused:
                        shutil.copyfile(reused[emb_hash] / 'scores.json', directory / 'scores.json')
                    else:
                        y = np.asarray(np.load(run_dir / 'embedding.npy'), dtype=request.dtype)
                        embed_key = embedding_cache_key(embedding_id=f"{d['name']}__{emb_hash}", embedding=y,
                                                        request=request, extra={'sha256': emb_hash})
                        result = store.evaluate_embedding(x, y, source_key=source_key, embedding_key=embed_key,
                                                          request=request, source_metadata={'dataset': d['name']},
                                                          embedding_metadata={'run': repo_relative(run_dir)})
                        scores = summarize_evaluation_scores(result.scores)
                        scalars = scalar_scores(scores)
                        if any(v is None for v in scalars.values()) or set(scalars) != set(config['metrics']):
                            raise ValueError(f'Missing/nonfinite quality metric: {scalars}')
                        write_json(directory / 'scores.json', {'scores': scores, 'scalars': scalars})
                    commit_output(directory, key, {'dataset': d, 'algorithm': a, 'seed': seed,
                                  'embedding_sha256': emb_hash, 'run_key': run_key(ctx, binding, a, seed),
                                  'evaluation': config, 'n_evaluated': len(x), 'environment': ctx.env})
                    reused[emb_hash] = directory
                    ctx.log.info('QUALITY %s %s seed=%d', d['name'], a['name'], seed)
                except Exception as exc:
                    ctx.error(directory, exc)
                finally:
                    force_cleanup(include_gpu=resolved_device.get('resolved') == 'gpu')
        deleted = store.cleanup_after_dataset(source_key)
        ctx.log.info('CACHE %s source_deleted=%s', d['name'], deleted)
    ctx.finish()


if __name__ == '__main__':
    main()
