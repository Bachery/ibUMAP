"""Shared context of the mechanism experiment: suites, fixed inputs, atomic outputs and stale-result gates.

Datasets are grouped into three size suites; every stage takes ``--suite`` and
writes to ``results/<suite>/``. The fixed inputs (kNN graph, optimizer graph and
spectral initialization, built once with seed 42) live in
``results/01_fixed_inputs/`` and are shared by all suites.

Every output directory carries a ``metadata.json`` with a key that hashes the
experiment configuration, the source code (``src/ibumap``, ``scripts/common``
and these scripts), the runtime environment and the input hashes. A completed
output is reused only when its key and all recorded output hashes still match.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SIZE_SUITES = {'le10k': (0, 10_000), '10k_50k': (10_000, 50_000),
               '50k_100k': (50_000, 100_000)}


def canonical_suite(value):
    return {'≤10k': 'le10k', '<=10k': 'le10k'}.get(value, value)


if str(REPO / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO / 'scripts'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))

from common.paths import ensure_ibumap_importable, repo_relative  # noqa: E402

ensure_ibumap_importable()

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import sparse  # noqa: E402
from common.gpu_runtime import machine_info  # noqa: E402
from common.run_metadata import git_commit, json_default, now_utc, package_versions  # noqa: E402


def sha(path, block_size=8 * 1024 * 1024):
    digest_ = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            digest_.update(block)
    return digest_.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _atomic(path, suffix, save):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.{os.getpid()}.{uuid.uuid4().hex}{suffix}')
    try:
        save(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def save_array(path, value):
    _atomic(path, '.npy', lambda tmp: np.save(tmp, value))


def save_sparse(path, value):
    _atomic(path, '.npz', lambda tmp: sparse.save_npz(tmp, value))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))

    def save(tmp):
        with tmp.open('w', newline='', encoding='utf-8') as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
    _atomic(path, '', save)


def write_json(path, value):
    _atomic(path, '', lambda tmp: tmp.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default).encode()).hexdigest()


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config-dir', type=Path, default=ROOT / 'configs')
    p.add_argument('--suite', type=canonical_suite, choices=list(SIZE_SUITES), required=True,
                   help='Size suite: le10k (N <= 10k; also accepts ≤10k or <=10k), 10k_50k or 50k_100k.')
    p.add_argument('--dataset', action='append')
    p.add_argument('--include-excluded', action='store_true',
                   help='Also select datasets marked `excluded` in configs/datasets.yaml.')
    p.add_argument('--seed', type=int, action='append')
    p.add_argument('--algorithm', action='append')
    p.add_argument('--run', action='store_true', help='Execute; otherwise print a plan.')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--threads', type=int)
    return p


class Context:
    def __init__(self, args, stage):
        self.args = args
        self.config_dir = args.config_dir.resolve()
        self.cfg = {n: yaml.safe_load((self.config_dir / f'{n}.yaml').read_text())
                    for n in ('experiment', 'datasets', 'algorithms', 'evaluation')}
        self.e = dict(self.cfg['experiment'])

        def resolve(key):
            return (self.config_dir / self.e[key]).resolve()
        self.processed = resolve('processed_root')
        self.fixed_root = resolve('fixed_inputs_root')
        self.results = resolve('results_root') / args.suite
        self.e['max_full_direct_n'] = SIZE_SUITES[args.suite][1]
        self.results.mkdir(parents=True, exist_ok=True)
        logdir = self.results.parent / 'logs'
        logdir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(stage + str(self.results))
        self.log.setLevel(logging.INFO)
        for h in list(self.log.handlers):
            self.log.removeHandler(h)
            h.close()
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        for h in (logging.StreamHandler(), logging.FileHandler(logdir / f'{stage}_{time.time_ns()}.log')):
            h.setFormatter(fmt)
            self.log.addHandler(h)
        self.threads = args.threads or self.e['threads']
        if self.threads < 1:
            raise ValueError('threads must be positive')
        import numba
        numba.set_num_threads(min(self.threads, numba.config.NUMBA_NUM_THREADS))
        self.threads = numba.get_num_threads()
        os.environ.setdefault('IBUMAP_FFT_THREADS', str(self.threads))
        machine = machine_info()
        self.env = {'platform': machine['platform'], 'python': machine['python_version'],
                    'cpu_model': machine['cpu_model'], 'git_commit': git_commit(REPO),
                    'numba_threads': self.threads,
                    'fft_threads': int(os.environ['IBUMAP_FFT_THREADS']),
                    'fftw_planner': os.environ.get('IBUMAP_FFTW_PLANNER_EFFORT', 'FFTW_ESTIMATE'),
                    'packages': package_versions(['ibumap', 'numpy', 'scipy', 'numba', 'umap-learn',
                                                  'pynndescent', 'pyFFTW', 'scikit-learn', 'pandas'])}
        # Runtime/source edits invalidate optimization, even without a new commit.
        files = list((REPO / 'src' / 'ibumap').rglob('*.py'))
        files += list((REPO / 'scripts' / 'common').glob('*.py'))
        files += list((ROOT / 'scripts').glob('*.py'))
        self.sources = {str(p.relative_to(REPO)): sha(p) for p in sorted(set(files))}
        self.code_hash = digest(self.sources)
        e = dict(self.e)
        for key in ['field', 'contrasts', 'results_root', 'fixed_inputs_root', 'processed_root', 'seeds']:
            e.pop(key, None)
        # The commit id is recorded but not hashed: the source fingerprint above
        # already covers every file that affects the computation.
        env = {k: v for k, v in self.env.items() if k != 'git_commit'}
        self.run_hash = digest({'experiment': e, 'code': self.code_hash, 'environment': env})
        self.failures = 0

    def datasets(self):
        entries = self.cfg['datasets']['datasets']
        if len({d['name'] for d in entries}) != len(entries):
            raise ValueError('Duplicate dataset names in manifest')
        if self.args.dataset:
            unknown = set(self.args.dataset) - {e['name'] for e in entries}
            if unknown:
                raise ValueError(f'Unknown datasets: {sorted(unknown)}')
        lo, hi = SIZE_SUITES[self.args.suite]
        selected = []
        for d in entries:
            if self.args.dataset and d['name'] not in self.args.dataset:
                continue
            if d.get('excluded') and not self.args.include_excluded:
                continue
            if lo < d['n'] <= hi:
                selected.append({'name': d['name'], 'n': d['n'], 'family': d['family'],
                                 'size_suite': self.args.suite})
        if not selected:
            raise ValueError('No datasets selected; check --suite and --dataset together')
        return selected

    def excluded(self):
        lo, hi = SIZE_SUITES[self.args.suite]
        return [d for d in self.cfg['datasets']['datasets'] if d.get('excluded') and lo < d['n'] <= hi]

    def algorithms(self, dataset=None):
        entries = self.cfg['algorithms']['algorithms']
        if self.args.algorithm and set(self.args.algorithm) - {a['name'] for a in entries}:
            raise ValueError('Unknown algorithm filter')
        return [a for a in entries if not self.args.algorithm or a['name'] in self.args.algorithm]

    def seeds(self, dataset=None):
        return self.args.seed or self.e['seeds']

    def params(self, seed):
        p = dict(self.e['common_params'])
        p.update(n_epochs=self.e['n_epochs'], random_state=seed, verbose=False)
        return p

    def source_files(self, name):
        root = self.fixed_root / name / 'neighbors_15'
        return {k: root / k for k in ['optimizer_graph_csr.npz', 'init_embedding.npy', 'metadata.json']}

    def binding(self, name):
        return self.results / '01_inputs' / name / 'binding.json'

    def load(self, d):
        binding = read_json(self.binding(d['name']))
        paths = self.source_files(d['name'])
        for k, p in paths.items():
            if sha(p) != binding['hashes'][k]:
                raise ValueError(f'Fixed input changed: {p}; rebuild or rebind with --overwrite')
        graph = sparse.load_npz(paths['optimizer_graph_csr.npz']).tocsr()
        y = np.load(paths['init_embedding.npy'])
        validate_inputs(graph, y, d['n'])
        return graph, y, binding

    def run_dir(self, name, algorithm, seed):
        return self.results / '02_runs' / name / f'{algorithm}__seed_{seed}'

    def error(self, directory, exc):
        self.failures += 1
        write_json(Path(directory) / 'failure.json', {'status': 'failed', 'time': now_utc(),
                   'error': str(exc), 'traceback': traceback.format_exc()})
        self.log.exception('FAILED %s', directory)

    def finish(self):
        if self.failures:
            raise SystemExit(f'{self.failures} task(s) failed; see failure.json and logs.')


def validate_inputs(graph, y, n):
    if graph.shape != (n, n) or y.shape != (n, 2):
        raise ValueError(f'Unexpected graph/init shapes: {graph.shape}, {y.shape}, expected N={n}')
    if graph.dtype != np.float32 or y.dtype != np.float32:
        raise ValueError('float32 graph/init required')
    if not graph.has_canonical_format or not graph.has_sorted_indices:
        raise ValueError('The optimizer graph must be canonical sorted CSR')
    if not np.isfinite(graph.data).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite fixed input')
    if np.any(graph.data <= 0) or np.any(graph.data > 1):
        raise ValueError('Graph weights must lie in (0,1]')


def complete(path, expected, overwrite=False):
    """Never silently reuse partial, corrupted, or stale successful output."""
    path = Path(path)
    meta = path / 'metadata.json'
    if not meta.exists() or overwrite:
        return False
    record = read_json(meta)
    if record.get('status') != 'ok' or record.get('key') != expected:
        raise RuntimeError(f'Stale output {path}; pass --overwrite')
    for rel, value in record.get('output_hashes', {}).items():
        if not (path / rel).is_file() or sha(path / rel) != value:
            raise RuntimeError(f'Corrupt/incomplete output {path / rel}; pass --overwrite')
    return True


def commit_output(directory, key, extra):
    directory = Path(directory)
    output_hashes = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
                     if p.is_file() and p.name not in ['metadata.json', 'failure.json']}
    write_json(directory / 'metadata.json', {'status': 'ok', 'key': key, 'time': now_utc(),
                                           'output_hashes': output_hashes, **extra})
    (directory / 'failure.json').unlink(missing_ok=True)


def run_key(ctx, binding, variant, seed):
    return digest({'run': ctx.run_hash, 'inputs': binding['input_key'], 'variant': variant, 'seed': seed})


def embedding_diagnostics(embedding, *, expected_rows):
    from common.embedding_diagnostics import bbox_summary
    points = np.asarray(embedding)
    result = {'shape': [int(v) for v in points.shape], 'dtype': str(points.dtype),
              'finite': bool(np.isfinite(points).all()),
              'row_count_matches': bool(points.ndim == 2 and points.shape[0] == expected_rows)}
    if result['finite'] and result['row_count_matches'] and points.shape[1] >= 2:
        result.update(bbox_summary(points[:, :2]))
        radial = np.linalg.norm(np.asarray(points[:, :2], dtype=np.float64), axis=1)
        result['radial_max'] = float(radial.max())
        result['radial_max_over_p99'] = float(radial.max() / max(np.percentile(radial, 99), 1e-12))
    return result


__all__ = ['ROOT', 'REPO', 'SIZE_SUITES', 'Context', 'parser', 'np', 'sparse', 'sha', 'read_json', 'write_json',
           'write_csv', 'save_array', 'save_sparse', 'digest', 'complete', 'commit_output', 'run_key',
           'validate_inputs', 'embedding_diagnostics', 'repo_relative', 'now_utc']
