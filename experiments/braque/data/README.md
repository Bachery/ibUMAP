# BRAQUE case data

This directory is populated by the numbered scripts:

```text
data/
├── raw/                 # verified Mendeley source files and source_manifest.json
└── processed/<sample>/  # canonical raw marker matrix, cell metadata, and QA reports
```

Run from the experiment directory (`experiments/braque`):

```bash
python scripts/01_download_data.py
python scripts/01_download_data.py --run
python scripts/02_check_and_prepare_data.py
python scripts/02_check_and_prepare_data.py --run --sample L2
```

`markers_raw.npy` is not Lognormal-Shrinkage output.  Feature selection, LNS,
robust standardization, and frozen UMAP-input generation belong to later stages.
The public Mendeley CSVs contain a CSV-export index plus marker intensities, but
not the x/y/area columns assumed by the paper-specific legacy loader.  The
preparation script drops only a verified zero-based export index and does not
invent spatial coordinates from the final three marker columns.

Runtime dependencies for data preparation are Python 3.10+, NumPy, pandas, and
openpyxl.  The downloader itself uses only the Python standard library.
