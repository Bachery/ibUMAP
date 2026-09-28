#!/usr/bin/env python3
"""Optional diagnostic (not in the paper): exact q-by-N repulsive fields on frozen snapshots versus FFT grids.

For the seed-42 trajectories of the variants in ``field.source_algorithms`` and
the snapshot steps in ``field.snapshot_steps``, 256 random and 3 x 32 stress
targets receive the exact float64 sum over all N sources (raw kernel and cap 4),
compared with CPU ibFFT at four (p, box-scale) grids.
"""
import shutil
from _common import (Context, parser, np, digest, complete, commit_output,
                     save_array, write_json, write_csv, sha, run_key)


def main():
    p = parser(__doc__)
    p.add_argument('--step', type=int, action='append')
    args = p.parse_args()
    ctx = Context(args, '03_fields')
    from _engine import direct_field, fft_field, kernel_params, select_targets, error_rows
    field_cfg = ctx.e['field']
    steps = args.step or field_cfg['snapshot_steps']
    if set(steps) - set(ctx.e['snapshot_steps']):
        raise ValueError('Requested field steps are not in optimizer snapshot_steps')
    for d in ctx.datasets():
        names = field_cfg['source_algorithms']
        variants = [a for a in ctx.algorithms(d) if a['name'] in names]
        for seed in (args.seed or [field_cfg['seed']]):
            for a in variants:
                for step in steps:
                    ctx.log.info('PLAN field %s %s seed=%d step=%d run=%s', d['name'], a['name'], seed, step, args.run)
                    if not args.run:
                        continue
                    directory = ctx.results / '03_fields' / d['name'] / f"{a['name']}__seed_{seed}" / f'step_{step:04d}'
                    try:
                        graph, init, binding = ctx.load(d)
                        run_dir = ctx.run_dir(d['name'], a['name'], seed)
                        if not complete(run_dir, run_key(ctx, binding, a, seed)):
                            raise ValueError(f'Missing successful snapshot source: {run_dir}')
                        snapshot = run_dir / f'snapshots/step_{step:04d}.npy'
                        key = digest({'run': run_key(ctx, binding, a, seed), 'snapshot': sha(snapshot), 'field': field_cfg})
                        if complete(directory, key, args.overwrite):
                            ctx.log.info('SKIP %s', directory)
                            continue
                        if directory.exists():
                            shutil.rmtree(directory)
                        directory.mkdir(parents=True)
                        y = np.load(snapshot)
                        if y.shape != init.shape or not np.isfinite(y).all():
                            raise ValueError('Invalid snapshot')
                        targets, groups = select_targets(y, graph, field_cfg['random_targets'],
                                                         field_cfg['stress_targets_per_group'], field_cfg['seed'])
                        save_array(directory / 'target_indices.npy', targets)
                        write_json(directory / 'target_groups.json', groups)
                        degree = np.asarray(graph.sum(axis=1)).ravel()
                        save_array(directory / 'target_degree.npy', degree[targets])
                        kernel = kernel_params(ctx.params(seed))
                        rows, grids = [], []
                        for cap in field_cfg['kernel_caps']:
                            label = 'raw' if cap is None else f'cap_{cap:g}'
                            def progress(done, total):
                                ctx.log.info('DIRECT %s %s %s step=%d targets=%d/%d sources=%d',
                                             d['name'], a['name'], label, step, done, total, len(y))
                            exact, cap_counts = direct_field(y, targets, kernel, cap,
                                                            batch=ctx.e['direct_target_batch'], progress=progress)
                            save_array(directory / f'exact_{label}.npy', exact)
                            save_array(directory / f'kernel_capped_source_counts_{label}.npy', cap_counts)
                            for grid in field_cfg['grids']:
                                fft_cfg = {**ctx.e['fft'], **grid}
                                diagnostics = {}
                                # ALWAYS deposit all N sources, then inspect selected targets.
                                approximate = fft_field(y, kernel, fft_cfg, cap, diagnostics=diagnostics)[targets].astype(np.float64)
                                if not np.isfinite(approximate).all():
                                    raise FloatingPointError('Nonfinite FFT field')
                                save_array(directory / f"fft_{label}_{grid['name']}.npy", approximate)
                                base = {'dataset': d['name'], 'size_suite': d['size_suite'], 'family': d['family'],
                                        'algorithm': a['name'], 'seed': seed, 'step': step,
                                        'n_sources': len(y), 'kernel': label, 'grid': grid['name']}
                                for row in error_rows(exact, approximate, targets, groups, field_cfg['near_zero_relative_floor']):
                                    rows.append({**base, **row})
                                grids.append({**base, 'requested': fft_cfg, 'resolved': diagnostics})
                                ctx.log.info('FFT %s %s %s complete', d['name'], label, grid['name'])
                        write_csv(directory / 'errors.csv', rows)
                        write_json(directory / 'grid_diagnostics.json', grids)
                        commit_output(directory, key, {'dataset': d, 'algorithm': a, 'seed': seed,
                                      'step': step, 'snapshot_sha256': sha(snapshot), 'field_config': field_cfg,
                                      'source_count': len(y), 'target_count': len(targets),
                                      'reference': 'float64 full-source sum R_i before degree/alpha/norm clipping',
                                      'environment': ctx.env})
                    except Exception as exc:
                        ctx.error(directory, exc)
    ctx.finish()


if __name__ == '__main__':
    main()
