"""Small independent numerical oracles and compatibility gates."""
from pathlib import Path
import tempfile
from _common import np, sparse


def fixture():
    n = 16
    rng = np.random.RandomState(17)
    weights = rng.uniform(.2, 1., (n, n)).astype(np.float32)
    weights = (weights + weights.T) * np.float32(.5)
    np.fill_diagonal(weights, 0)
    weights[0, 1] = weights[1, 0] = 1
    graph = sparse.csr_matrix(weights)
    y = rng.normal(size=(n, 2)).astype(np.float32)
    params = dict(n_components=2, n_neighbors=5, spread=1., min_dist=.1,
                  n_epochs=12, learning_rate=1., repulsion_strength=1.,
                  negative_sample_rate=5, random_state=42, metric='euclidean',
                  umap_epsilon=.001, verbose=False)
    cfg = {'snapshot_steps': [0, 6, 12], 'diagnostic_stride': 3, 'log_stride': 6,
           'direct_target_batch': 7, 'fft': {'n_interpolation_points': 1,
            'intervals_per_integer': 1., 'min_num_intervals': 10, 'n_boxes_per_dim': 1.}}
    return graph, y, params, cfg


def numerical_checks():
    from _engine import direct_field, fft_field, kernel_params, run_matched, run_external, run_ibumap_prepared
    graph, y, params, cfg = fixture()
    kernel = kernel_params(params)
    # Independent dense NumPy oracle; test unequal target ordering and duplicates.
    y = y.copy()
    y[3] = y[2]
    targets = np.array([7, 2, 0, 3])
    d = y[targets].astype(float)[:, None, :] - y.astype(float)[None, :, :]
    s = np.sum(d * d, axis=2)
    a, b, gamma, eps = kernel
    k = 2 * gamma * b / ((eps + s) * (1 + a * s ** b))
    for cap, clip in [(None, False), (4., False), (None, True)]:
        kk = k if cap is None else np.minimum(k, cap)
        force = kk[:, :, None] * d
        if clip:
            force = np.clip(force, -4., 4.)
        expected = force.sum(axis=1)
        got, counts = direct_field(y, targets, kernel, cap, clip, batch=2)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    all_force, _ = direct_field(y, np.arange(len(y)), kernel)
    np.testing.assert_allclose(all_force.sum(axis=0), 0, atol=1e-10)
    shifted, _ = direct_field(y.astype(float) + [4., -8.], targets, kernel)
    original, _ = direct_field(y, targets, kernel)
    np.testing.assert_allclose(shifted, original, atol=1e-10)
    checks = {'direct_numpy_oracle': True, 'self_duplicate_translation_symmetry': True}
    graph, y, params, cfg = fixture()
    with tempfile.TemporaryDirectory(prefix='field_checks_') as tmp:
        root = Path(tmp)
        for seed in [42, 137]:
            pp = {**params, 'random_state': seed}
            ref, _ = run_external(graph, y, pp, {'mode': 'reference'}, cfg, root / 'ref')
            async_y, _ = run_external(graph, y, pp, {'mode': 'async_control'}, cfg, root / 'async')
            np.testing.assert_array_equal(ref, async_y)
            for norm in [None, 4.]:
                variant = {'mode': 'sampled', 'norm_clip': norm}
                actual, _ = run_matched(graph, y, pp, variant, cfg, root / f'sync_{seed}_{norm}')
                p = {k: v for k, v in pp.items() if k != 'umap_epsilon'}
                p.update(deterministic=True, rng_lifecycle='shared_rng', umap_sgd_update_mode='synchronous',
                         repulsion_clip_norm=norm, repulsion_clip_with_alpha=True, total_update_clip_norm=None)
                expected, _ = run_ibumap_prepared(graph, y, p, algorithm='umap')
                np.testing.assert_array_equal(actual, expected)
        checks['async_and_sync_bitwise_alignment_two_seeds'] = True
        # Verify the production observer leaves both final output and initial snapshot unchanged.
        hook_dir = root / 'production'
        hook_dir.mkdir()
        observed, extras = run_external(graph, y, params, {'mode': 'production'}, cfg, hook_dir)
        p = {k: v for k, v in params.items() if k != 'umap_epsilon'}
        p.update(deterministic=True, attraction_mode='sampling', repulsion_mode='true_loss',
                 attraction_degree_damping=True, repulsion_clip_norm=4., repulsion_clip_with_alpha=True,
                 repulsion_clip_epoch_range=None, total_update_clip_norm=None)
        unobserved, _ = run_ibumap_prepared(graph, y, p, algorithm='ibumap')
        np.testing.assert_array_equal(observed, unobserved)
        np.testing.assert_array_equal(np.load(hook_dir / 'snapshots/step_0000.npy'), y)
        np.testing.assert_array_equal(np.load(hook_dir / 'snapshots/step_0012.npy'), observed)
        checks['production_snapshot_observer_bitwise_neutral'] = True
        for mode in ['direct', 'event_expected', 'fft']:
            v = {'mode': mode, 'kernel_cap': 4.}
            first, _ = run_matched(graph, y, params, v, cfg, root / (mode + 'a'))
            second, _ = run_matched(graph, y, {**params, 'random_state': 137}, v, cfg, root / (mode + 'b'))
            np.testing.assert_array_equal(first, second)
        checks['fixed_init_dense_seed_invariance'] = True
    # Sanity-check real FFT against the correct capped reference, not uncapped.
    exact, _ = direct_field(y, np.arange(len(y)), kernel, 4.)
    fine = {**cfg['fft'], 'n_interpolation_points': 3, 'n_boxes_per_dim': 2.}
    approx = fft_field(y, kernel, fine, 4.)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    if not np.isfinite(rel) or rel > .1:
        raise AssertionError(f'FFT capped-field smoke error unexpectedly large: {rel}')
    checks['fft_capped_relative_l2_error'] = float(rel)
    return checks
