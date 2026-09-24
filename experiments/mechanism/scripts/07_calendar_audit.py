#!/usr/bin/env python3
"""Replay umap-learn's sampling calendar on every retained dataset (appendix table tab:calendar-audit).

Read-only with respect to the optimizer results. For each retained dataset of the
three size suites, loads the optimizer graph from results/01_fixed_inputs/
(hash-checked against results/<suite>/01_inputs/<name>/binding.json) and replays
umap-learn 0.5.12's positive/negative sampling calendar in pure NumPy (no RNG, no
embedding): ``make_epochs_per_sample``, post-activation increments and
``n_neg_samples = int((t - next_neg) / neg_interval)``.

Reported per dataset:
  graph checks  : symmetry, v_max, v_min / v_max (pruning at v_max / T)
  schedule      : first active epoch, draws at an edge's first activation
  event weights : time-averaged c_i(t) / (m d_i); per-epoch total draws
                  relative to the epoch mean (t >= 1); share of epochs with
                  c_i(t) = 0
  normalizations: E_i / mean_t c_i(t) with E_i = m d_i / 2 (times 1/n on both);
                  repulsion-to-attraction ratio of D-G (two-endpoint attraction,
                  E_i) and H (head-wise attraction, E_i), each relative to the
                  reference/C ratio (mean c_i over two-endpoint attraction events).

    python scripts/07_calendar_audit.py [--only NAME ...] [--overwrite]   # per-dataset JSON, cached
    python scripts/07_calendar_audit.py --summarize                       # summary.json, per_dataset.csv

Output: results/07_calendar_audit/{per_dataset/<name>.json, summary.json, per_dataset.csv}.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

EXP = Path(__file__).resolve().parents[1]
CONFIG = FIXED = RESULTS = HERE = None
T = M = None


def configure(config_dir):
    global CONFIG, FIXED, RESULTS, HERE, T, M
    config_dir = Path(config_dir).resolve()
    CONFIG = {name: yaml.safe_load((config_dir / f'{name}.yaml').read_text()) for name in ('experiment', 'datasets')}
    FIXED = (config_dir / CONFIG['experiment']['fixed_inputs_root']).resolve()
    RESULTS = (config_dir / CONFIG['experiment']['results_root']).resolve()
    HERE = RESULTS / '07_calendar_audit'
    T = int(CONFIG['experiment']['n_epochs'])
    M = int(CONFIG['experiment']['common_params']['negative_sample_rate'])


def datasets():
    return [(d['name'], int(d['n']), d['family']) for d in CONFIG['datasets']['datasets']
            if int(d['n']) <= 100_000 and not d.get('excluded')]


def suite(n):
    return 'le10k' if n <= 10_000 else ('10k_50k' if n <= 50_000 else '50k_100k')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def q(x):
    return {k: float(np.percentile(x, p)) for k, p in (('p5', 5), ('median', 50), ('p95', 95))} | \
           {'min': float(np.min(x)), 'max': float(np.max(x))}


def audit(name, n_expected, family):
    path = FIXED / name / 'neighbors_15' / 'optimizer_graph_csr.npz'
    binding = json.loads((RESULTS / suite(n_expected) / '01_inputs' / name / 'binding.json').read_text())
    digest = sha(path)
    if digest != binding['hashes']['optimizer_graph_csr.npz']:
        raise RuntimeError(f'{name}: graph hash differs from binding')
    g = sp.load_npz(path).tocsr()
    n = g.shape[0]
    assert n == n_expected, name
    coo = g.tocoo()
    w = coo.data.astype(np.float64)
    vmax = float(w.max())
    # umap-learn 0.5.12 make_epochs_per_sample (all weights > 0 after pruning)
    eps = float(T) / (T * (w / w.max()))
    neg = eps / M
    nxt, nneg = eps.copy(), neg.copy()
    c_tot = np.zeros(n); a_head = np.zeros(n); a_tail = np.zeros(n)
    zero_epochs = np.zeros(n); totals = np.zeros(T)
    first_draws = np.full(len(w), -1, dtype=np.int64)
    first_active = None
    for t in range(T):
        act = np.flatnonzero(nxt <= t)
        if act.size and first_active is None:
            first_active = t
        draws = np.trunc((t - nneg[act]) / neg[act]).astype(np.int64)
        draws = np.maximum(draws, 0)
        fresh = first_draws[act] < 0
        first_draws[act[fresh]] = draws[fresh]
        c = np.bincount(coo.row[act], weights=draws, minlength=n)
        c_tot += c
        a_head += np.bincount(coo.row[act], minlength=n)
        a_tail += np.bincount(coo.col[act], minlength=n)
        zero_epochs += (c == 0)
        totals[t] = c.sum()
        nxt[act] += eps[act]
        nneg[act] += draws * neg[act]
    d = np.asarray(g.sum(axis=1), dtype=np.float64).ravel()
    ok = d > 0
    cbar = c_tot / T
    r = cbar[ok] / (M * d[ok])
    e_over_c = (0.5 * M * d[ok]) / cbar[ok]
    rho_ref = cbar[ok] / ((a_head + a_tail)[ok] / T)
    rho_dg = (0.5 * M * d[ok]) / ((a_head + a_tail)[ok] / T)
    rho_h = (0.5 * M * d[ok]) / (a_head[ok] / T)
    active_totals = totals[1:]
    rel = active_totals / active_totals.mean()
    fd = first_draws[first_draws >= 0]
    return {
        'dataset': name, 'family': family, 'n': n, 'nnz': int(g.nnz), 'graph_sha256': digest,
        'symmetric_max_abs_diff': float(abs(g - g.T).max()) if g.nnz else 0.0,
        'v_max': vmax, 'v_min_over_v_max': float(w.min() / vmax),
        'isolated_vertices': int((~ok).sum()),
        'first_active_epoch': first_active,
        'epoch0_scheduled_draws': float(totals[0]),
        'first_activation_draws_share_m_minus_1': float(np.mean(fd == M - 1)),
        'first_activation_draws_share_m': float(np.mean(fd == M)),
        'first_activation_draws_min': int(fd.min()), 'first_activation_draws_max': int(fd.max()),
        'head_tail_activation_max_abs_diff': float(np.abs(a_head - a_tail).max()),
        'cbar_over_m_d': q(r),
        'epoch_total_over_mean_t_ge_1': {'min': float(rel.min()), 'max': float(rel.max()), 'cv': float(rel.std() / rel.mean()),
                                         'argmin_epoch': int(np.argmin(rel)) + 1, 'argmax_epoch': int(np.argmax(rel)) + 1,
                                         'epoch1': float(rel[0]),
                                         'min_t_ge_2': float(rel[1:].min()), 'max_t_ge_2': float(rel[1:].max()),
                                         'min_t_ge_10': float(rel[9:].min()), 'max_t_ge_10': float(rel[9:].max())},
        'share_epochs_with_zero_draws': q(zero_epochs[ok] / T),
        'E_over_cbar': q(e_over_c),
        'rho_DG_over_ref': q(rho_dg / rho_ref),
        'rho_H_over_ref': q(rho_h / rho_ref),
    }


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted((HERE / 'per_dataset').glob('*.json'))]
    names = {r['dataset'] for r in rows}
    missing = [d for d, _, _ in datasets() if d not in names]
    if missing:
        raise SystemExit(f'missing: {missing}')
    med = lambda key, stat='median': np.array([r[key][stat] for r in rows])
    s = {
        'n_datasets': len(rows), 'n_range': [min(r['n'] for r in rows), max(r['n'] for r in rows)],
        'all_symmetric': all(r['symmetric_max_abs_diff'] == 0 for r in rows),
        'v_max_values': sorted({r['v_max'] for r in rows}),
        'min_v_min_over_v_max': min(r['v_min_over_v_max'] for r in rows),
        'isolated_vertices_total': sum(r['isolated_vertices'] for r in rows),
        'first_active_epochs': sorted({r['first_active_epoch'] for r in rows}),
        'epoch0_draws_total': sum(r['epoch0_scheduled_draws'] for r in rows),
        'head_tail_activation_max_abs_diff': max(r['head_tail_activation_max_abs_diff'] for r in rows),
        'first_activation_share_m_minus_1': q(np.array([r['first_activation_draws_share_m_minus_1'] for r in rows])),
        'first_activation_draws_range': [min(r['first_activation_draws_min'] for r in rows), max(r['first_activation_draws_max'] for r in rows)],
    }
    for key in ('cbar_over_m_d', 'share_epochs_with_zero_draws', 'E_over_cbar', 'rho_DG_over_ref', 'rho_H_over_ref'):
        s[key] = {'dataset_medians': q(med(key)), 'pooled_p5_min': float(med(key, 'p5').min()), 'pooled_p95_max': float(med(key, 'p95').max())}
    s['epoch_total_over_mean'] = {
        'min_over_datasets': q(np.array([r['epoch_total_over_mean_t_ge_1']['min'] for r in rows])),
        'max_over_datasets': q(np.array([r['epoch_total_over_mean_t_ge_1']['max'] for r in rows])),
        'cv': q(np.array([r['epoch_total_over_mean_t_ge_1']['cv'] for r in rows])),
        **{k: q(np.array([r['epoch_total_over_mean_t_ge_1'][k] for r in rows])) for k in ('epoch1', 'min_t_ge_2', 'max_t_ge_2', 'min_t_ge_10', 'max_t_ge_10')},
        'argmin_epochs': sorted({r['epoch_total_over_mean_t_ge_1']['argmin_epoch'] for r in rows}),
        'argmax_epochs': sorted({r['epoch_total_over_mean_t_ge_1']['argmax_epoch'] for r in rows})}
    (HERE / 'summary.json').write_text(json.dumps(s, indent=1))
    keys = ['dataset', 'family', 'n', 'nnz', 'v_max', 'v_min_over_v_max', 'first_active_epoch']
    with open(HERE / 'per_dataset.csv', 'w') as f:
        f.write(','.join(keys + ['cbar_over_m_d_median', 'E_over_cbar_median', 'rho_DG_over_ref_median', 'rho_H_over_ref_median', 'epoch_total_rel_min', 'epoch_total_rel_max', 'zero_epoch_share_median']) + '\n')
        for r in rows:
            vals = [str(r[k]) for k in keys] + [f"{r[k]['median']:.6f}" for k in ('cbar_over_m_d', 'E_over_cbar', 'rho_DG_over_ref', 'rho_H_over_ref')]
            vals += [f"{r['epoch_total_over_mean_t_ge_1']['min']:.6f}", f"{r['epoch_total_over_mean_t_ge_1']['max']:.6f}", f"{r['share_epochs_with_zero_draws']['median']:.6f}"]
            f.write(','.join(vals) + '\n')
    print(json.dumps(s, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--config-dir', type=Path, default=EXP / 'configs')
    ap.add_argument('--only', action='append')
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--max-n', type=int)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    configure(a.config_dir)
    if a.summarize:
        summarize(); sys.exit()
    out = HERE / 'per_dataset'
    out.mkdir(parents=True, exist_ok=True)
    for name, n, fam in datasets():
        if (a.only and name not in a.only) or (a.max_n and n > a.max_n) or ((out / f'{name}.json').exists() and not a.overwrite):
            continue
        (out / f'{name}.json').write_text(json.dumps(audit(name, n, fam), indent=1))
        print(name, n, flush=True)
