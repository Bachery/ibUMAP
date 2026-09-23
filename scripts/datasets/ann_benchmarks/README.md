# ANN Benchmark Dataset Scripts

This directory contains reproducible entry points for ANN Benchmark HDF5 datasets.

## Supported Datasets

- `gist_960_euclidean`

## Download

```bash
python scripts/datasets/ann_benchmarks/download.py gist_960_euclidean
python scripts/datasets/ann_benchmarks/download.py --all
```

The downloader writes to `datasets/raw/<dataset_id>/` by default and skips existing files unless `--force` is provided.

## Prepare

```bash
python scripts/datasets/ann_benchmarks/prepare.py gist_960_euclidean
python scripts/datasets/ann_benchmarks/prepare.py --all
```

Each prepare run writes a standardized processed dataset:

```text
datasets/processed/<dataset_id>/
  features.npy
  queries.npy
  neighbors.npy
  distances.npy
  obs.csv.gz
  metadata.json
  checksums.txt
```

Refresh metadata and checksums without rewriting large arrays:

```bash
python scripts/datasets/ann_benchmarks/prepare.py gist_960_euclidean --metadata-only
```

## Validate

```bash
python scripts/datasets/validate_dataset.py datasets/processed/gist_960_euclidean
python scripts/datasets/validate_dataset.py --all
python scripts/datasets/build_catalog.py
```
