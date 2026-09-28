#!/usr/bin/env python3
"""Run every stage on two 64-point synthetic datasets in a separate folder (minutes on a laptop CPU).

Builds processed features, then runs 01 -> 02 -> 03 -> 04 -> 05, reruns 02 and
03 to exercise the completed-output checks and the deterministic reuse, and
finishes with the calendar audit (07) and the paper-data export (08) of the
``le10k`` suite. Uses 12 epochs, two seeds and small metric samples.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from _common import ROOT, np, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--output', type=Path, help='Fresh directory; default: a temporary directory kept for inspection.')
    args = p.parse_args()
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='ibumap_mechanism_smoke_'))
    if args.output and root.exists():
        raise SystemExit('--output must be a fresh directory')
    config_dir = root / 'configs'
    config_dir.mkdir(parents=True)
    config = {name: yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text())
              for name in ['experiment', 'datasets', 'algorithms', 'evaluation']}
    e = config['experiment']
    e.update(processed_root=str(root / 'processed'), fixed_inputs_root=str(root / 'results/01_fixed_inputs'),
             results_root=str(root / 'results'), threads=2, n_epochs=12,
             snapshot_steps=[0, 6, 12], diagnostic_stride=3, log_stride=6, seeds=[42, 137])
    e['fixed_inputs']['n_epochs'] = 12
    e['field'].update(random_targets=8, stress_targets_per_group=2, snapshot_steps=[0, 6, 12])
    e['field']['grids'] = [e['field']['grids'][0], e['field']['grids'][2]]
    config['evaluation'].update(device='cpu', rta_n_triplets=2000, distance_spearman_n_pairs=2000)
    config['evaluation']['cache']['root'] = str(root / 'results/le10k/04_quality/metric_cache')
    config['datasets']['datasets'] = [{'name': 'smoke_a', 'n': 64, 'family': 'synthetic'},
                                      {'name': 'smoke_b', 'n': 64, 'family': 'synthetic'}]
    rng = np.random.RandomState(42)
    for d in config['datasets']['datasets']:
        centers = rng.normal(scale=6.0, size=(4, 6))
        x = (centers[np.arange(d['n']) % 4] + rng.normal(size=(d['n'], 6))).astype(np.float32)
        processed = root / 'processed' / d['name']
        processed.mkdir(parents=True)
        np.save(processed / 'features.npy', x)
        write_json(processed / 'metadata.json', {'dataset_id': d['name'], 'primary_array': 'features.npy',
                                                'feature_shape': list(x.shape), 'synthetic_smoke_only': True})
    for name, value in config.items():
        (config_dir / f'{name}.yaml').write_text(yaml.safe_dump(value, sort_keys=False))
    suite = ['--suite', 'le10k']
    commands = [
        ['01_prepare_fixed_inputs.py', *suite, '--run'],
        ['02_run_mechanism.py', *suite, '--run'],
        ['03_check_field_accuracy.py', *suite, '--run'],
        ['04_evaluate_quality.py', *suite, '--run'],
        ['05_summarize_plot.py', *suite, '--require-fields'],
        # Rerun: completed outputs are verified and skipped.
        ['01_prepare_fixed_inputs.py', *suite, '--run'],
        ['02_run_mechanism.py', *suite, '--run'],
        ['03_check_field_accuracy.py', *suite, '--run'],
        ['07_calendar_audit.py'],
        ['07_calendar_audit.py', '--summarize'],
        ['08_export_paper_data.py', *suite, '--output', str(root / 'paper_data')],
    ]
    print(f'Synthetic smoke output: {root}', flush=True)
    for command in commands:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / command[0]),
                        '--config-dir', str(config_dir), *command[1:]], check=True)
    write_json(root / 'smoke_result.json', {'status': 'passed', 'commands': commands, 'synthetic_only': True})
    print(f'PASS: {root / "smoke_result.json"}', flush=True)


if __name__ == '__main__':
    main()
