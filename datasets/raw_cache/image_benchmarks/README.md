# Image Benchmark Raw Cache

This directory stores re-downloadable raw cache files for image benchmark
datasets prepared by `scripts/datasets/image_benchmarks/`.

The cache is not an authoritative raw archive. Files here can be recreated with:

```bash
python scripts/datasets/image_benchmarks/download.py --all
```

Processed datasets are written under `datasets/processed/`.
