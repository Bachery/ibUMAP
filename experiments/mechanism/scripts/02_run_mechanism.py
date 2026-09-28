#!/usr/bin/env python3
"""Run the optimizer variants of configs/algorithms.yaml on the fixed inputs of one size suite.

Each (dataset, variant, seed) writes results/<suite>/02_runs/<dataset>/<variant>__seed_<seed>/
with the final embedding, snapshots after 0, 10, 50, 100 and 200 updates and, for
the controlled synchronous variants, per-epoch force/geometry diagnostics. The
dense variants (sync_expected, sync_direct, sync_direct_capped, sync_fft,
sync_fft_guarded) do not depend on the optimizer seed under a fixed
initialization: they are computed once and copied for the other seeds. The
asynchronous control must reproduce umap-learn exactly; a mismatch fails the task.
"""
import shutil
import time
from _common import (Context, parser, np, digest, complete, commit_output, save_array,
                     sha, run_key, embedding_diagnostics, repo_relative)

DENSE_MODES = {'direct', 'event_expected', 'fft'}


def main():
    p = parser(__doc__)
    p.add_argument('--pilot', action='store_true', help='Short separate run; does not replace final outputs.')
    p.add_argument('--pilot-steps', type=int, default=5)
    args = p.parse_args()
    ctx = Context(args, '02_optimize')
    from _engine import run_matched, run_external, direct_field, kernel_params
    if args.pilot_steps < 1 or args.pilot_steps > ctx.e['n_epochs']:
        raise ValueError('pilot-steps must be in [1, n_epochs]')
    for d in ctx.datasets():
        algorithms = ctx.algorithms(d)
        for seed in ctx.seeds(d):
            for a in algorithms:
                ctx.log.info('PLAN dataset=%s algorithm=%s seed=%d pilot=%s run=%s',
                             d['name'], a['name'], seed, args.pilot, args.run)
        if not args.run:
            continue
        try:
            graph, init, binding = ctx.load(d)
        except Exception as exc:
            ctx.error(ctx.results / '02_runs' / d['name'], exc)
            continue
        cached_dense = {}
        for seed in ctx.seeds(d):
            for a in algorithms:
                directory = ctx.run_dir(d['name'], a['name'], seed)
                if args.pilot:
                    directory = ctx.results / '02_pilot' / d['name'] / directory.name
                try:
                    key = run_key(ctx, binding, a, seed)
                    if args.pilot:
                        key = digest({'full_key': key, 'pilot_steps': args.pilot_steps})
                    if complete(directory, key, args.overwrite):
                        ctx.log.info('SKIP %s', directory)
                        if a['mode'] in DENSE_MODES:
                            cached_dense[a['name']] = directory
                        continue
                    if directory.exists():
                        shutil.rmtree(directory)
                    directory.mkdir(parents=True)
                    params = ctx.params(seed)
                    cfg = dict(ctx.e)
                    if args.pilot:
                        # Pilot timing only; it uses a short schedule and lives separately.
                        params['n_epochs'] = args.pilot_steps
                        cfg['snapshot_steps'] = [0, args.pilot_steps]
                        cfg['log_stride'] = 1
                    mode = a['mode']
                    if mode in DENSE_MODES and len(init) > ctx.e['max_full_direct_n']:
                        raise ValueError(f"Full-trajectory limit {ctx.e['max_full_direct_n']} exceeded; "
                                         'select the matching size suite')
                    if mode in DENSE_MODES and a['name'] in cached_dense:
                        source = cached_dense[a['name']]
                        for path in source.rglob('*'):
                            if path.is_file() and path.name not in ['metadata.json', 'failure.json']:
                                target = directory / path.relative_to(source)
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(path, target)
                        commit_output(directory, key, {'dataset': d, 'algorithm': a, 'seed': seed,
                                      'input_key': binding['input_key'], 'run_hash': ctx.run_hash,
                                      'reused_deterministic_run': repo_relative(source),
                                      'independent_seed_replicate': False, 'environment': ctx.env})
                        ctx.log.info('REUSE deterministic %s (fixed initialization)', directory.name)
                        continue
                    # Compile the exact kernel before timing, with the same dtype.
                    if mode in ('direct', 'event_expected'):
                        direct_field(init[:8], np.arange(8), kernel_params(params), component_clip=mode == 'event_expected')
                    started = time.perf_counter()
                    if mode in ('reference', 'async_control', 'production'):
                        y, extras = run_external(graph, init, params, a, cfg, directory)
                    else:
                        y, extras = run_matched(graph, init, params, a, cfg, directory, ctx.log)
                    if not np.isfinite(y).all() or y.shape != init.shape:
                        raise ValueError('Invalid final embedding')
                    save_array(directory / 'embedding.npy', y)
                    control = None
                    if mode == 'async_control':
                        reference_dir = ctx.run_dir(d['name'], 'umap_learn', seed)
                        if args.pilot:
                            reference_dir = ctx.results / '02_pilot' / d['name'] / reference_dir.name
                        reference_variant = next(v for v in ctx.cfg['algorithms']['algorithms'] if v['mode'] == 'reference')
                        reference_key = run_key(ctx, binding, reference_variant, seed)
                        if args.pilot:
                            reference_key = digest({'full_key': reference_key, 'pilot_steps': args.pilot_steps})
                        if not complete(reference_dir, reference_key):
                            raise ValueError('Run umap_learn first for the current seed')
                        ref = np.load(reference_dir / 'embedding.npy')
                        control = {'exact': bool(np.array_equal(ref, y)),
                                   'max_abs_difference': float(np.max(np.abs(ref.astype(float) - y))),
                                   'reference_hash': sha(reference_dir / 'embedding.npy')}
                        if not control['exact']:
                            from _common import write_json
                            write_json(directory / 'control_mismatch.json', control)
                            raise ValueError(f'Async control mismatch: {control}')
                    elapsed = time.perf_counter() - started
                    commit_output(directory, key, {'dataset': d, 'algorithm': a, 'seed': seed,
                                  'input_key': binding['input_key'], 'run_hash': ctx.run_hash,
                                  'params': params, 'extras': extras, 'control': control,
                                  'environment': ctx.env, 'source_fingerprints': ctx.sources,
                                  'wall_seconds_provenance_only': elapsed,
                                  'embedding_diagnostics': embedding_diagnostics(y, expected_rows=len(y)),
                                  'pilot': args.pilot, 'independent_seed_replicate': mode not in DENSE_MODES})
                    if mode in DENSE_MODES:
                        cached_dense[a['name']] = directory
                    ctx.log.info('DONE %s seconds=%.2f', directory, elapsed)
                except Exception as exc:
                    ctx.error(directory, exc)
    ctx.finish()


if __name__ == '__main__':
    main()
