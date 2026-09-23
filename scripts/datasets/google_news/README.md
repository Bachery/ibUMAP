# Google News Dataset Scripts

This directory contains the reproducible entry points for the Google News Word2Vec dataset.

## Download

```bash
python scripts/datasets/google_news/download.py
```

The downloader writes to `datasets/raw/google_news/GoogleNews-vectors-negative300.bin.gz` by default. It skips the download if the raw file already exists. Use `--force` to replace the existing raw archive.

## Prepare

```bash
python scripts/datasets/google_news/prepare.py
```

The prepare step reads the raw gzip archive with streaming I/O and writes:

```text
datasets/processed/google_news_300d/
  features.npy
  obs.csv.gz
  metadata.json
  checksums.txt
```

Use `--force` to overwrite the known processed outputs in that directory.

Refresh metadata and checksums without rewriting the large feature matrix:

```bash
python scripts/datasets/google_news/prepare.py --metadata-only
```

## Validate

```bash
python scripts/datasets/validate_dataset.py datasets/processed/google_news_300d
python scripts/datasets/build_catalog.py
```
