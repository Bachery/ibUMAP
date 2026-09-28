# Image Benchmarks

Prepare image benchmark datasets without torch, torchvision, or tensorflow. This
pipeline downloads official benchmark files into a disposable raw cache and then
parses the raw IDX/CIFAR formats into the shared processed dataset layout.

```text
datasets/raw_cache/image_benchmarks/<dataset>/
  original downloaded files

datasets/processed/<dataset_id>/
  features.npy
  obs.csv.gz
  target.npy
  metadata.json
  checksums.txt
```

Supported datasets:

- `mnist784`
- `fashion_mnist`
- `cifar10`

## Download

```bash
python scripts/datasets/image_benchmarks/download.py mnist784
python scripts/datasets/image_benchmarks/download.py fashion_mnist
python scripts/datasets/image_benchmarks/download.py cifar10
python scripts/datasets/image_benchmarks/download.py --all
```

Existing raw cache files are skipped by default. Use `--force` to redownload.

## Prepare

```bash
python scripts/datasets/image_benchmarks/prepare.py mnist784
python scripts/datasets/image_benchmarks/prepare.py fashion_mnist
python scripts/datasets/image_benchmarks/prepare.py cifar10
python scripts/datasets/image_benchmarks/prepare.py --all
python scripts/datasets/image_benchmarks/prepare.py --all --force
```

`prepare.py` does not download files. If raw cache files are missing, run the
matching `download.py` command first.

## Output Rules

- Images are flattened in row-major order.
- Features are stored as `float32` normalized to `[0,1]`.
- Original image cubes and `uint8` arrays are not persisted in processed output.
- `obs.csv.gz` contains `sample_id`, `split`, `label`, and `label_name`.
- `target.npy` is an `int32` cache derived from the `label` column.
