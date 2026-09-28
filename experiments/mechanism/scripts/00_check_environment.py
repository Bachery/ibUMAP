#!/usr/bin/env python3
"""Check dependencies, synthetic numerical controls and the availability of processed datasets."""
from _common import Context, parser, repo_relative, write_json


def main():
    p = parser(__doc__)
    p.add_argument('--strict-data', action='store_true')
    args = p.parse_args()
    ctx = Context(args, '00_environment')
    from _checks import numerical_checks
    issues = []
    version = ctx.env['packages'].get('umap-learn')
    if version != ctx.e['required_umap_version']:
        issues.append(f"Need umap-learn=={ctx.e['required_umap_version']}; found {version}")
    try:
        checks = numerical_checks()
    except Exception as exc:
        checks = {'error': str(exc)}
        issues.append(f'Numerical control failed: {exc}')
        ctx.log.exception('Numerical control failed')
    missing = []
    for d in ctx.datasets():
        meta = ctx.processed / d['name'] / 'metadata.json'
        if not meta.is_file():
            missing.append(repo_relative(meta))
        else:
            from _common import read_json
            feature = meta.parent / read_json(meta).get('primary_array', 'features.npy')
            if not feature.is_file():
                missing.append(repo_relative(feature))
    if args.strict_data and missing:
        issues.append(f'{len(missing)} required input files are missing')
    result = {'status': 'failed' if issues else 'ok', 'environment': ctx.env,
              'code_hash': ctx.code_hash, 'source_fingerprints': ctx.sources,
              'numerical_checks': checks, 'missing_inputs': missing, 'issues': issues}
    write_json(ctx.results / '00_environment.json', result)
    ctx.log.info('CHECK status=%s missing_inputs=%d issues=%s', result['status'], len(missing), issues)
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
