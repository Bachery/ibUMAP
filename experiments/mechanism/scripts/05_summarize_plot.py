#!/usr/bin/env python3
"""Validate completeness of one size suite, summarize paired effects and plot dynamics / field errors.

Writes results/<suite>/05_summary/: completeness.csv, manifest.json, quality_long.csv,
paired_quality.csv, dataset_effects.csv, suite_summary.csv, dynamics.csv,
field_errors.csv and diagnostic plots. ``08_export_paper_data.py`` combines the
three suites into the paper-data layout.
"""
import csv
from collections import defaultdict
from _common import (Context, parser, read_json, write_json, write_csv, np,
                     complete, run_key, digest, sha)


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p = parser(__doc__)
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--require-fields', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    ctx = Context(args, '05_summary')
    output = ctx.results / '05_summary'
    # Summarization is cheap and read-only with respect to all source results.
    records, quality, dynamics, errors = [], [], [], []
    selected = ctx.datasets()
    for d in selected:
        try:
            graph, init, binding = ctx.load(d)
        except Exception as exc:
            records.append({'dataset': d['name'], 'stage': 'input', 'status': str(exc)})
            continue
        for seed in ctx.seeds(d):
            for a in ctx.algorithms(d):
                directory = ctx.run_dir(d['name'], a['name'], seed)
                row = {'dataset': d['name'], 'algorithm': a['name'], 'seed': seed, 'stage': 'optimization'}
                try:
                    key = run_key(ctx, binding, a, seed)
                    if not complete(directory, key):
                        raise ValueError('missing')
                    row['status'] = 'ok'
                    records.append(dict(row))
                    dyn = directory / 'dynamics.csv'
                    if dyn.exists():
                        dynamics.extend([{**row, **r} for r in csv_rows(dyn)])
                    if not a.get('evaluate', True):
                        continue
                    row['stage'] = 'quality'
                    ev = ctx.results / '04_quality' / d['name'] / directory.name
                    metadata = read_json(ev / 'metadata.json')
                    expected = digest({'run_key': key, 'embedding': sha(directory / 'embedding.npy'),
                                       'evaluation': metadata['evaluation'], 'features': binding['feature_hash']})
                    if not complete(ev, expected):
                        raise ValueError('missing')
                    # Device override is allowed; metric definitions must still match.
                    for k, v in ctx.cfg['evaluation'].items():
                        if k != 'device' and metadata['evaluation'].get(k) != v:
                            raise ValueError(f'Stale evaluation config: {k}')
                    for metric, value in read_json(ev / 'scores.json')['scalars'].items():
                        quality.append({**row, 'size_suite': d['size_suite'], 'family': d['family'],
                                        'metric': metric, 'value': value})
                    records.append(dict(row))
                except Exception as exc:
                    records.append({**row, 'status': str(exc)})
        field_cfg = ctx.e['field']
        names = field_cfg['source_algorithms']
        for seed in (args.seed or [field_cfg['seed']]):
            for a in [a for a in ctx.algorithms(d) if a['name'] in names]:
                for step in field_cfg['snapshot_steps']:
                    directory = ctx.results / '03_fields' / d['name'] / f"{a['name']}__seed_{seed}" / f'step_{step:04d}'
                    row = {'dataset': d['name'], 'algorithm': a['name'], 'seed': seed, 'stage': 'field', 'step': step}
                    try:
                        snapshot = ctx.run_dir(d['name'], a['name'], seed) / f'snapshots/step_{step:04d}.npy'
                        key = digest({'run': run_key(ctx, binding, a, seed), 'snapshot': sha(snapshot), 'field': field_cfg})
                        if not complete(directory, key):
                            raise ValueError('missing')
                        errors.extend(csv_rows(directory / 'errors.csv'))
                        records.append({**row, 'status': 'ok'})
                    except Exception as exc:
                        records.append({**row, 'status': str(exc), 'optional': not args.require_fields})
    write_csv(output / 'completeness.csv', records)
    bad = [r for r in records if r['status'] != 'ok' and not r.get('optional')]
    write_json(output / 'manifest.json', {'status': 'incomplete' if bad else 'complete',
               'datasets': selected, 'missing_or_stale': bad, 'environment': ctx.env,
               'field_results_required': args.require_fields,
               'seed_scope': 'optimizer draws only; graph and initialization fixed',
               'aggregation': 'average paired seeds within each dataset, then summarize datasets; no pooled-seed confidence claims',
               'excluded_datasets': ctx.excluded()})
    if bad and not args.allow_incomplete:
        raise SystemExit(f'{len(bad)} missing/stale task(s); see {output / "completeness.csv"}')
    write_csv(output / 'quality_long.csv', quality)
    write_csv(output / 'dynamics.csv', dynamics)
    write_csv(output / 'field_errors.csv', errors)
    values = {(r['dataset'], r['seed'], r['algorithm'], r['metric']): r['value'] for r in quality}
    pairs = []
    for d in selected:
        for seed in ctx.seeds(d):
            for name, ref, cand in ctx.e['contrasts']:
                for metric in ctx.cfg['evaluation']['metrics']:
                    rkey, ckey = (d['name'], seed, ref, metric), (d['name'], seed, cand, metric)
                    if rkey in values and ckey in values:
                        sign = 1  # all five metrics are higher-is-better
                        pairs.append({'dataset': d['name'], 'size_suite': d['size_suite'], 'family': d['family'],
                                      'seed': seed, 'contrast': name, 'reference': ref, 'candidate': cand,
                                      'metric': metric, 'signed_improvement': sign * (values[ckey] - values[rkey])})
    write_csv(output / 'paired_quality.csv', pairs)
    by_dataset = defaultdict(list)
    for r in pairs:
        by_dataset[(r['dataset'], r['size_suite'], r['contrast'], r['metric'])].append(r['signed_improvement'])
    means = [{'dataset': k[0], 'size_suite': k[1], 'contrast': k[2], 'metric': k[3],
              'seed_pair_count': len(v), 'mean_signed_improvement': float(np.mean(v)),
              'min_signed_improvement': float(min(v)), 'max_signed_improvement': float(max(v))}
             for k, v in by_dataset.items()]
    write_csv(output / 'dataset_effects.csv', means)
    grouped = defaultdict(list)
    for r in means:
        grouped[(r['size_suite'], r['contrast'], r['metric'])].append(r['mean_signed_improvement'])
    write_csv(output / 'suite_summary.csv', [{'size_suite': k[0], 'contrast': k[1], 'metric': k[2],
              'n_datasets': len(v), 'median_dataset_effect': float(np.median(v)),
              'min_dataset_effect': float(min(v)), 'max_dataset_effect': float(max(v))} for k, v in grouped.items()])
    if not args.no_plots:
        plot(output, selected, means, dynamics, errors)
    ctx.log.info('SUMMARY %s (%s)', output, 'incomplete' if bad else 'complete')


def plot(output, datasets, effects, dynamics, errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for d in datasets:
        name = d['name']
        rows = [r for r in dynamics if r['dataset'] == name and r['seed'] == 42]
        if rows:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for ax, field in zip(axes.flat, ['attraction_mean', 'repulsion_before_clip_mean',
                                            'robust_bbox_area', 'repulsion_clipped_fraction']):
                for algo in sorted({r['algorithm'] for r in rows}):
                    points = [r for r in rows if r['algorithm'] == algo]
                    ax.plot([int(r['step_before']) for r in points], [float(r[field]) for r in points], label=algo)
                ax.set(xlabel='Epoch (before update)', ylabel=field)
                if field != 'repulsion_clipped_fraction':
                    ax.set_yscale('symlog', linthresh=1e-6)
                ax.grid(alpha=.2)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(name + ' — fixed initialization, seed 42')
            fig.tight_layout()
            fig.savefig(output / f'{name}__dynamics.png', dpi=160)
            plt.close(fig)
        rows = [r for r in errors if r['dataset'] == name and r['target_group'] == 'random']
        if rows:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            grids = list(dict.fromkeys(r['grid'] for r in rows))
            for cap, ax in zip(['raw', 'cap_4'], axes):
                series = sorted({(r['algorithm'], r['step']) for r in rows if r['kernel'] == cap})
                for algo, step in series:
                    by_grid = {r['grid']: r for r in rows if r['kernel'] == cap and r['algorithm'] == algo and r['step'] == step}
                    ax.plot(grids, [float(by_grid[g]['relative_vector_error_p95']) for g in grids], marker='o', label=f'{algo} t={step}')
                ax.set(title=cap, ylabel='95th percentile relative vector error', yscale='log')
                ax.tick_params(axis='x', rotation=30)
                ax.grid(alpha=.2)
                ax.legend(fontsize=6)
            fig.suptitle(name + ' — uniform random target panel')
            fig.tight_layout()
            fig.savefig(output / f'{name}__field_errors.png', dpi=160)
            plt.close(fig)
    if effects:
        metrics = list(dict.fromkeys(r['metric'] for r in effects))
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            rows = [r for r in effects if r['metric'] == metric]
            labels = list(dict.fromkeys(r['contrast'] for r in rows))
            for d in datasets:
                rr = {r['contrast']: r['mean_signed_improvement'] for r in rows if r['dataset'] == d['name']}
                if rr:
                    ax.scatter([labels.index(k) for k in rr], list(rr.values()), label=d['name'])
            ax.axhline(0, color='black', linewidth=.8)
            ax.set_xticks(range(len(labels)), labels, rotation=20, ha='right')
            ax.set_ylabel(metric + '\npositive = better')
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / 'paired_quality.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
