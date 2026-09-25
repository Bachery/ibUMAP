# Classic Benchmarks

Prepare small scikit-learn benchmark datasets in the shared processed dataset
format:

```text
features.npy
obs.csv.gz
target.npy      # classification datasets only
metadata.json
checksums.txt
```

Supported datasets:

- `iris`
- `wine`
- `breast_cancer`
- `digits`
- `diabetes` (regression; no `target.npy`)

The datasets are loaded from scikit-learn's built-in loaders, so no raw file is
downloaded or persisted. Their metadata uses `raw_role="not_persisted"` and
`raw_path=null`.

## Usage

```bash
python scripts/datasets/classic_benchmarks/prepare.py iris
python scripts/datasets/classic_benchmarks/prepare.py wine
python scripts/datasets/classic_benchmarks/prepare.py breast_cancer
python scripts/datasets/classic_benchmarks/prepare.py digits
python scripts/datasets/classic_benchmarks/prepare.py diabetes
python scripts/datasets/classic_benchmarks/prepare.py --all
python scripts/datasets/classic_benchmarks/prepare.py --all --force
```

By default, the script refuses to overwrite an existing processed directory. Use
`--force` only when intentionally rebuilding generated outputs.

Validate outputs with:

```bash
python scripts/datasets/validate_dataset.py --reference datasets/processed/iris
python scripts/datasets/validate_dataset.py --all
python scripts/datasets/build_catalog.py
```
