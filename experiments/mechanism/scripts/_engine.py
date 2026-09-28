"""Controlled synchronous ladder, exact source sums, and production snapshots.

No library code is modified. All matched variants call the SAME production
synchronous single-epoch routine for attraction. Dense variants replace only
its repulsion buffer. The production ibUMAP anchor remains separate.
"""
from __future__ import annotations

from contextlib import contextmanager
import time

from _common import np, save_array, write_csv
import numba
from scipy.spatial import cKDTree
from umap.umap_ import make_epochs_per_sample, INT32_MIN, INT32_MAX
from ibumap.optimizers.umap_optimizer import (
    optimize_layout_euclidean_synchronous_single_epoch, _make_rng_state_per_sample,
)
from ibumap.optimizers.ibumap_optimizer import _clip_force_norm_inplace
from ibumap.kernels.cpu.ibfft import ibFFT_repulsive_sampling
from common.algorithm_adapters import run_ibumap_optimize_from_prepared_graph
from common.fixed_inputs import find_ab_params

sync_epoch = numba.njit(optimize_layout_euclidean_synchronous_single_epoch, fastmath=True)


def run_umap_learn_optimizer(graph, init_embedding, params):
    """umap-learn's own ``optimize_layout_euclidean`` on the fixed graph and initialization."""
    from sklearn.utils import check_random_state
    from umap.layouts import optimize_layout_euclidean

    n_epochs = int(params.get('n_epochs', 200))
    matrix = graph.tocoo(copy=True)
    epochs_per_sample = make_epochs_per_sample(matrix.data, n_epochs)
    random_state = check_random_state(params.get('random_state'))
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    a, b = find_ab_params(float(params.get('spread', 1.0)), float(params.get('min_dist', 0.1)))
    embedding = np.asarray(init_embedding, dtype=np.float32, order='C').copy()
    result = optimize_layout_euclidean(
        embedding, embedding, np.asarray(matrix.row), np.asarray(matrix.col), n_epochs, int(graph.shape[0]),
        epochs_per_sample, a, b, rng_state, gamma=float(params.get('repulsion_strength', 1.0)),
        initial_alpha=float(params.get('learning_rate', 1.0)),
        negative_sample_rate=float(params.get('negative_sample_rate', 5)), parallel=False,
        verbose=bool(params.get('verbose', False)), densmap=False, densmap_kwds={}, tqdm_kwds=None,
        move_other=True)
    return np.asarray(result, dtype=np.float32, order='C'), {
        'effective_params': {'entrypoint': 'umap.layouts.optimize_layout_euclidean',
                             'random_state': params.get('random_state'), 'parallel': False, 'n_epochs': n_epochs,
                             'a': a, 'b': b, 'graph_preprocessing_bypassed': True,
                             'initialization_bypassed': True},
        'time_costs': {}}


def run_ibumap_prepared(graph, init_embedding, params, *, algorithm):
    """IBUMAP's CPU optimizer (``algorithm='umap'`` or ``'ibumap'``) on the prepared optimizer graph."""
    embedding, extras = run_ibumap_optimize_from_prepared_graph(
        None, graph.copy(), np.asarray(init_embedding, dtype=np.float32, order='C').copy(), params,
        algorithm=algorithm, device='cpu')
    return np.asarray(embedding, dtype=np.float32, order='C'), {
        'effective_params': dict(extras.get('used_params') or {}),
        'effective_config': extras.get('effective_config'),
        'time_costs': dict(extras.get('time_costs') or {})}


@numba.njit(parallel=True, cache=True)
def _direct(y, targets, a, b, gamma, epsilon, cap, component_clip):
    """Float64 accumulation; O(qN) arithmetic, O(q+N) memory, no NxN arrays."""
    out = np.zeros((len(targets), 2), dtype=np.float64)
    clipped_counts = np.zeros(len(targets), dtype=np.int64)
    for z in numba.prange(len(targets)):
        i = targets[z]
        x0, x1 = np.float64(y[i, 0]), np.float64(y[i, 1])
        r0, r1 = 0.0, 0.0
        ncap = 0
        for j in range(len(y)):
            if i == j:
                continue
            dx, dy = x0 - np.float64(y[j, 0]), x1 - np.float64(y[j, 1])
            s = dx * dx + dy * dy
            k = 2.0 * gamma * b / ((epsilon + s) * (1.0 + a * s ** b))
            if k > cap:
                k = cap
                ncap += 1
            fx, fy = k * dx, k * dy
            if component_clip:
                fx = min(4.0, max(-4.0, fx))
                fy = min(4.0, max(-4.0, fy))
            r0 += fx
            r1 += fy
        out[z, 0], out[z, 1] = r0, r1
        clipped_counts[z] = ncap
    return out, clipped_counts


def direct_field(y, targets, kernel, cap=None, component_clip=False, batch=128, progress=None):
    y = np.asarray(y)
    targets = np.asarray(targets, dtype=np.int64)
    if batch < 1 or targets.ndim != 1 or np.any(targets < 0) or np.any(targets >= len(y)):
        raise ValueError('Invalid direct-field targets or batch size')
    out = np.empty((len(targets), 2), dtype=np.float64)
    counts = np.empty(len(targets), dtype=np.int64)
    for start in range(0, len(targets), batch):
        stop = min(start + batch, len(targets))
        out[start:stop], counts[start:stop] = _direct(
            y, targets[start:stop], *kernel, np.inf if cap is None else float(cap), component_clip)
        if progress:
            progress(stop, len(targets))
    return out, counts


def kernel_params(params):
    a, b = find_ab_params(params['spread'], params['min_dist'])
    return a, b, float(params['repulsion_strength']), float(params['umap_epsilon'])


def fft_field(y, kernel, fft, cap=None, workspace=None, diagnostics=None):
    a, b, gamma, eps = kernel
    estimate = fft_memory_estimate(y, fft)
    if estimate['estimated_gib'] > fft.get('max_estimated_gib', 8.):
        raise MemoryError(f"FFT conservative workspace estimate {estimate['estimated_gib']:.2f} GiB "
                          f"exceeds configured {fft.get('max_estimated_gib', 8.)} GiB: {estimate}")
    if diagnostics is not None:
        diagnostics.update(estimate)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        return ibFFT_repulsive_sampling(
            np.ascontiguousarray(y), fft['n_interpolation_points'], fft['intervals_per_integer'],
            fft['min_num_intervals'], 1.0, 1.0, None, fft['n_boxes_per_dim'],
            kernel_method='UMAP_kernel', umap_a=a, umap_b=b, umap_gamma=gamma,
            umap_epsilon=eps, ibfft_kernel_clip=np.inf if cap is None else float(cap),
            deterministic=True, p2m_mode='auto', _workspace=workspace,
            p2m_diagnostics=diagnostics, fft_kernel_cache_policy='disabled',
        )


def fft_memory_estimate(y, fft):
    from ibumap.kernels.cpu.ibfft import _ALLOWED_N_BOXES_PER_DIM
    n = len(y)
    span = float(np.max(y)) - float(np.min(y))
    estimate = int(min(np.sqrt(16 * n), max(np.sqrt(4 * n / np.log(n)),
                   fft['min_num_intervals'], span / fft['intervals_per_integer'])))
    estimate = int(float(fft['n_boxes_per_dim']) * estimate)
    if estimate >= _ALLOWED_N_BOXES_PER_DIM[-1]:
        boxes = int(_ALLOWED_N_BOXES_PER_DIM[-1])
    else:
        boxes = int(_ALLOWED_N_BOXES_PER_DIM[_ALLOWED_N_BOXES_PER_DIM > estimate][0])
    p = fft['n_interpolation_points']
    side = 2 * p * boxes
    # Deliberately conservative allowance for real/complex channels, FFT plans,
    # interpolation and deposition buffers. Not a claimed peak-memory benchmark.
    gib = (128 * side * side + 256 * n * p * p) / 2**30
    return {'estimated_boxes_per_dim': boxes, 'estimated_fft_side': side,
            'estimated_gib': gib}


def geometry(y):
    points = np.asarray(y, dtype=np.float64)
    centered = points - np.median(points, axis=0)
    radial = np.linalg.norm(centered, axis=1)
    q = np.percentile(points, [1, 99], axis=0)
    neighbors = cKDTree(points).query(points, k=2)[0][:, 1]
    return {'radius_p99': float(np.percentile(radial, 99)),
            'radius_max': float(radial.max()),
            'radius_max_over_p99': float(radial.max() / max(np.percentile(radial, 99), 1e-12)),
            'robust_bbox_area': float(np.prod(q[1] - q[0])),
            'nearest_neighbor_median': float(np.median(neighbors))}


def _norm_summary(prefix, x):
    v = np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)
    return {prefix + '_mean': float(np.mean(v)), prefix + '_p99': float(np.percentile(v, 99)),
            prefix + '_max': float(np.max(v))}


def run_matched(graph, init, params, variant, cfg, directory, log=None):
    directory.mkdir(parents=True, exist_ok=True)
    y = np.array(init, dtype=np.float32, order='C', copy=True)
    n = len(y)
    tmax = int(params['n_epochs'])
    snapshots = set(cfg['snapshot_steps'])
    if 0 in snapshots:
        save_array(directory / 'snapshots/step_0000.npy', y)
    coo = graph.tocoo(copy=True)
    intervals = make_epochs_per_sample(coo.data, tmax)
    neg_intervals = intervals / params['negative_sample_rate']
    next_pos, next_neg = intervals.copy(), neg_intervals.copy()
    rng = np.random.RandomState(params['random_state'])
    rng_state = rng.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    rng_per_head = _make_rng_state_per_sample(y, rng_state)
    known = np.zeros(n, dtype=np.bool_)
    known_pos = np.empty((0, 2), dtype=np.float32)
    known_index = np.full(n, -1, dtype=np.int64)
    attr, rep = np.zeros_like(y), np.zeros_like(y)
    kernel = kernel_params(params)
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64)
    weight = degree * (0.5 * params['negative_sample_rate'] / n)
    damping = np.ones(n, dtype=np.float32)
    if variant.get('degree_damping'):
        damping = np.minimum(1.0, (np.percentile(degree, 99) / np.maximum(degree, 1e-12)) ** 0.5).astype(np.float32)
    high_degree = degree >= np.percentile(degree, 99)
    rows = []
    workspace = {}
    alpha0 = float(params['learning_rate'])
    alpha = alpha0
    targets = np.arange(n, dtype=np.int64)
    start = time.perf_counter()
    for epoch in range(tmax):
        step_start = time.perf_counter()
        attr.fill(0)
        rep.fill(0)
        # Count scheduled draws BEFORE the production routine advances calendars.
        active = next_pos <= epoch
        draws = np.maximum(0, ((epoch - next_neg[active]) / neg_intervals[active]).astype(np.int64))
        event_counts = np.bincount(coo.row[active], weights=draws, minlength=n)
        sync_epoch(y, y, attr, rep, coo.row, coo.col, n, 2, rng_per_head,
                   *kernel[:3], alpha, epoch, intervals, neg_intervals,
                   next_neg, next_pos, True, kernel[3], known, known_pos, known_index)
        mode = variant['mode']
        if mode in ('direct', 'event_expected'):
            field, _ = direct_field(y, targets, kernel, variant.get('kernel_cap'),
                                    component_clip=mode == 'event_expected',
                                    batch=cfg['direct_target_batch'])
            weights = event_counts / n if mode == 'event_expected' else weight
            rep[:] = field * (alpha * weights[:, None])
        elif mode == 'fft':
            field = fft_field(y, kernel, cfg['fft'], variant.get('kernel_cap'), workspace)
            rep[:] = np.asarray(field, dtype=np.float64) * (alpha * weight[:, None])
        elif mode != 'sampled':
            raise ValueError(f'Not a matched optimizer: {mode}')
        record = epoch % cfg['diagnostic_stride'] == 0 or epoch == tmax - 1
        row = {'step_before': epoch, 'alpha': alpha, 'scheduled_negative_draws': int(event_counts.sum())}
        if record:
            row.update(_norm_summary('attraction_before_damping', attr))
            row.update(_norm_summary('repulsion_before_clip', rep))
            pre_geom = geometry(y)
        attr *= damping[:, None]
        clipped = np.zeros(n, dtype=bool)
        if variant.get('norm_clip') is not None:
            threshold = variant['norm_clip'] * alpha
            clipped = np.linalg.norm(rep.astype(np.float64), axis=1) > threshold
            _clip_force_norm_inplace(rep, threshold, False)
        if record:
            row.update(_norm_summary('attraction', attr))
            row.update(_norm_summary('repulsion', rep))
            row['repulsion_clipped_fraction'] = float(clipped.mean())
            row['repulsion_clipped_high_degree_fraction'] = float(clipped[high_degree].mean())
            row['degree_damped_fraction'] = float(np.mean(damping < 1))
        attr += rep
        y += attr
        if not np.isfinite(y).all():
            save_array(directory / 'nonfinite_embedding.npy', y)
            write_csv(directory / 'dynamics.csv', rows)
            raise FloatingPointError(f'Nonfinite embedding after step {epoch + 1}')
        if record:
            row.update(_norm_summary('update', attr))
            row.update(geometry(y))
            row['robust_area_change'] = row['robust_bbox_area'] - pre_geom['robust_bbox_area']
            row['step_seconds'] = time.perf_counter() - step_start
            rows.append(row)
        if epoch + 1 in snapshots:
            save_array(directory / f'snapshots/step_{epoch + 1:04d}.npy', y)
        # Exact production UMAP schedule: alpha is updated AFTER the epoch.
        alpha = alpha0 * (1.0 - float(epoch) / tmax)
        if log and ((epoch + 1) % cfg['log_stride'] == 0 or epoch == tmax - 1):
            log.info('%s step=%d/%d elapsed=%.1fs', directory.name, epoch + 1, tmax, time.perf_counter() - start)
    write_csv(directory / 'dynamics.csv', rows)
    return y, {'kernel': kernel, 'attraction': 'reference_two_endpoint_synchronous',
               'alpha_schedule': 'umap_learn_post_epoch', 'wall_seconds': time.perf_counter() - start,
               'dense_weights': 'scheduled_draws/n' if mode == 'event_expected' else 'degree*m/(2*n)',
               'component_clip': mode in ('sampled', 'event_expected')}


@contextmanager
def production_snapshot_hook(directory, steps):
    """Observe pre-update ibFFT input without changing the computed field.

    Capture call t as the state after t completed updates. The final state is
    saved by the caller. Scope is one single-threaded Python process/run.
    """
    import ibumap.optimizers.ibumap_optimizer as module
    original = module.ibFFT_repulsive_sampling
    state = {'calls': 0}
    def wrapped(y, *args, **kwargs):
        step = state['calls']
        if step in steps:
            save_array(directory / f'snapshots/step_{step:04d}.npy', y)
        state['calls'] += 1
        return original(y, *args, **kwargs)
    module.ibFFT_repulsive_sampling = wrapped
    try:
        yield state
    finally:
        module.ibFFT_repulsive_sampling = original


def run_external(graph, init, params, variant, cfg, directory):
    mode = variant['mode']
    p = dict(params)
    p.pop('umap_epsilon', None)
    if mode == 'reference':
        return run_umap_learn_optimizer(graph, init, p)
    p.update(deterministic=True)
    if mode == 'async_control':
        p.update(rng_lifecycle='shared_rng', umap_sgd_update_mode='asynchronous',
                 repulsion_clip_norm=None, total_update_clip_norm=None)
        return run_ibumap_prepared(graph, init, p, algorithm='umap')
    if mode != 'production':
        raise ValueError(mode)
    p.update(attraction_mode='sampling', repulsion_mode='true_loss',
             attraction_degree_damping=True, repulsion_clip_norm=4.0,
             repulsion_clip_with_alpha=True, repulsion_clip_epoch_range=None,
             total_update_clip_norm=None)
    with production_snapshot_hook(directory, set(cfg['snapshot_steps'])) as state:
        result, extras = run_ibumap_prepared(graph, init, p, algorithm='ibumap')
    if state['calls'] != params['n_epochs']:
        raise RuntimeError(f"Snapshot hook expected {params['n_epochs']} FFT calls, observed {state['calls']}")
    if params['n_epochs'] in cfg['snapshot_steps']:
        save_array(directory / f"snapshots/step_{params['n_epochs']:04d}.npy", result)
    extras['snapshot_calls'] = state['calls']
    return result, extras


def select_targets(y, graph, random_count, stress_count, seed):
    n = len(y)
    rng = np.random.default_rng(seed)
    random = np.sort(rng.choice(n, min(n, random_count), replace=False))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    radial = np.linalg.norm(y.astype(np.float64) - np.median(y, axis=0), axis=1)
    k = min(8, n)
    radii = cKDTree(y).query(y, k=k)[0][:, -1]
    q = min(stress_count, n)
    groups = {'random': random, 'high_degree': np.argsort(-degree, kind='stable')[:q],
              'dense_region': np.argsort(radii, kind='stable')[:q],
              'radial_tail': np.argsort(-radial, kind='stable')[:q]}
    targets = np.unique(np.concatenate(list(groups.values())))
    return targets, groups


def error_rows(exact, approximate, targets, groups, floor_ratio):
    norms = np.linalg.norm(exact, axis=1)
    predicted_norms = np.linalg.norm(approximate, axis=1)
    floor = max(float(np.median(norms)) * floor_ratio, np.finfo(float).eps)
    abs_error = np.linalg.norm(approximate - exact, axis=1)
    relative = abs_error / np.maximum(norms, floor)
    norm_error = np.abs(predicted_norms - norms) / np.maximum(norms, floor)
    valid = (norms > floor) & (predicted_norms > floor)
    cosine = np.sum(exact * approximate, axis=1) / np.maximum(norms * predicted_norms, floor * floor)
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    rows = []
    for name, ids in groups.items():
        ix = np.searchsorted(targets, np.sort(ids))
        angular = angles[ix][valid[ix]]
        rows.append({'target_group': name, 'n_targets': len(ix),
                     'absolute_error_median': float(np.median(abs_error[ix])),
                     'relative_vector_error_median': float(np.median(relative[ix])),
                     'relative_vector_error_p95': float(np.percentile(relative[ix], 95)),
                     'relative_vector_error_max': float(np.max(relative[ix])),
                     'relative_norm_error_median': float(np.median(norm_error[ix])),
                     'angular_error_median_deg': float(np.median(angular)) if len(angular) else None,
                     'angular_error_p95_deg': float(np.percentile(angular, 95)) if len(angular) else None,
                     'angular_valid_count': int(valid[ix].sum()),
                     'near_zero_reference_count': int((norms[ix] <= floor).sum()),
                     'relative_error_denominator_floor': floor})
    return rows
