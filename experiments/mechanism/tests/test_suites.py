"""Size-suite selection, exclusions, aliases and per-suite output paths."""
import sys
from pathlib import Path

import pytest
import yaml

# Other experiments also have a scripts/_common.py; import this experiment's copy.
sys.modules.pop('_common', None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from _common import Context, ROOT, SIZE_SUITES, parser  # noqa: E402


def selection(suite, names=None, entries=None, include_excluded=False):
    ctx = Context.__new__(Context)
    argv = ['--suite', suite] + (['--include-excluded'] if include_excluded else [])
    ctx.args = parser('test').parse_args(argv)
    ctx.args.dataset = names
    ctx.cfg = {name: yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text())
               for name in ['datasets', 'algorithms']}
    ctx.e = {'seeds': [42, 137, 2026]}
    if entries is not None:
        ctx.cfg['datasets']['datasets'] = entries
    return ctx


def test_manifest_partition_and_exclusion():
    groups = []
    for suite, count, candidates in [('le10k', 19, 19), ('10k_50k', 28, 28), ('50k_100k', 12, 13)]:
        ctx = selection(suite)
        datasets = ctx.datasets()
        assert len(datasets) == count
        assert len(selection(suite, include_excluded=True).datasets()) == candidates
        groups.append({d['name'] for d in datasets})
        for d in datasets:
            assert d['size_suite'] == suite
            assert len(ctx.algorithms(d)) == 10
            assert ctx.seeds(d) == [42, 137, 2026]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 59
    assert {'spambase', 'hiva'} <= groups[0]
    assert {'cifar10', 'scdeed_cart'} <= groups[2]
    assert [d['name'] for d in selection('50k_100k').excluded()] == ['c_elegans_embryogenesis_global_qc_annotated']


def test_boundaries():
    entries = [dict(name=str(n), n=n, family='synthetic') for n in [10000, 10001, 50000, 50001, 100000, 100001]]
    for suite, expected in [('le10k', {10000}), ('10k_50k', {10001, 50000}), ('50k_100k', {50001, 100000})]:
        assert {d['n'] for d in selection(suite, entries=entries).datasets()} == expected


@pytest.mark.parametrize('alias', ['≤10k', '<=10k', 'le10k'])
def test_aliases_and_filters(alias):
    ctx = selection(alias, ['hiva'])
    assert ctx.args.suite == 'le10k'
    assert [d['name'] for d in ctx.datasets()] == ['hiva']
    with pytest.raises(ValueError, match='No datasets selected'):
        selection(alias, ['cifar10']).datasets()
    with pytest.raises(ValueError, match='Unknown datasets'):
        selection(alias, ['not_a_dataset']).datasets()


def test_suite_is_required():
    with pytest.raises(SystemExit):
        parser('test').parse_args([])


def test_context_paths_and_direct_limits(tmp_path):
    config = tmp_path / 'configs'
    config.mkdir()
    for name in ['experiment', 'datasets', 'algorithms', 'evaluation']:
        data = yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text())
        if name == 'experiment':
            data['results_root'] = str(tmp_path / 'results')
            data['fixed_inputs_root'] = str(tmp_path / 'results' / '01_fixed_inputs')
        (config / f'{name}.yaml').write_text(yaml.safe_dump(data))
    for suite, (_, hi) in SIZE_SUITES.items():
        ctx = Context(parser('test').parse_args(['--config-dir', str(config), '--suite', suite]), 'test_suites')
        assert ctx.results == tmp_path / 'results' / suite
        assert ctx.fixed_root == tmp_path / 'results' / '01_fixed_inputs'
        assert ctx.source_files('iris')['init_embedding.npy'].parent == ctx.fixed_root / 'iris' / 'neighbors_15'
        assert ctx.e['max_full_direct_n'] == hi
        assert ctx.cfg['experiment']['max_full_direct_n'] == 6000
