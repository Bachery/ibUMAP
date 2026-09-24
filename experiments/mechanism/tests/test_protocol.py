import sys
from pathlib import Path

# Other experiments also have a scripts/_common.py; import this experiment's copies.
for _name in ('_common', '_engine', '_checks'):
    sys.modules.pop(_name, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import pytest  # noqa: E402
from _common import np, save_array, complete, commit_output  # noqa: E402
from _engine import direct_field, error_rows, select_targets, run_matched, fft_field, kernel_params  # noqa: E402
from _checks import fixture, numerical_checks  # noqa: E402


def test_numerical_compatibility():
    assert numerical_checks()['async_and_sync_bitwise_alignment_two_seeds']


def test_uniform_negative_expectation_exhaustive():
    # Enumerate every possible negative target for a tiny one-head sampling event.
    y = np.array([[0., 0.], [.0001, 0.], [1., 2.]], dtype=np.float32)
    kernel = (1.4, .8, 1., .001)
    field, _ = direct_field(y, np.array([0]), kernel, component_clip=True)
    events = []
    for j in range(3):
        delta = y[0].astype(float) - y[j]
        s = delta @ delta
        k = 2 * kernel[2] * kernel[1] / ((kernel[3] + s) * (1 + kernel[0] * s ** kernel[1]))
        events.append(np.clip(k * delta, -4, 4))
    np.testing.assert_allclose(field[0] / 3, np.mean(events, axis=0), atol=1e-12)


def test_targets_use_all_sources():
    graph, y, _, _ = fixture()
    targets, groups = select_targets(y, graph, 4, 2, 42)
    assert len(groups['random']) == 4
    field, _ = direct_field(y, targets, (1., 1., 1., .001))
    subset, _ = direct_field(y[targets], np.arange(len(targets)), (1., 1., 1., .001))
    assert not np.allclose(field, subset)


def test_zero_field_does_not_invent_angles():
    rows = error_rows(np.zeros((2, 2)), np.ones((2, 2)), np.arange(2), {'random': np.arange(2)}, 1e-8)
    assert rows[0]['angular_valid_count'] == 0
    assert rows[0]['angular_error_median_deg'] is None
    assert rows[0]['near_zero_reference_count'] == 2


def test_output_hash_and_staleness(tmp_path):
    save_array(tmp_path / 'embedding.npy', np.ones((3, 2)))
    commit_output(tmp_path, 'key1', {})
    assert complete(tmp_path, 'key1')
    with pytest.raises(RuntimeError, match='Stale'):
        complete(tmp_path, 'key2')
    save_array(tmp_path / 'embedding.npy', np.zeros((3, 2)))
    with pytest.raises(RuntimeError, match='Corrupt'):
        complete(tmp_path, 'key1')


def test_input_state_not_mutated(tmp_path):
    graph, y, params, cfg = fixture()
    original = y.copy()
    data = graph.data.copy()
    run_matched(graph, y, params, {'mode': 'direct'}, cfg, tmp_path)
    np.testing.assert_array_equal(y, original)
    np.testing.assert_array_equal(graph.data, data)


def test_fft_memory_guard():
    _, y, params, cfg = fixture()
    with pytest.raises(MemoryError, match='workspace estimate'):
        fft_field(y, kernel_params(params), {**cfg['fft'], 'max_estimated_gib': 0.0}, cap=4.)
