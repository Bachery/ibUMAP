# Tabular ML

Seven small tabular classification datasets, prepared in the shared processed
format:

```text
datasets/processed/<dataset_id>/
  features.npy   float32
  target.npy     int32
  obs.csv.gz     sample_id, label, label_name
  metadata.json
  checksums.txt
```

| dataset_id | Shape | Classes | Source |
|---|---|---|---|
| `bank` | 2,059 × 63 | 2 | Espadoto et al. (2021) |
| `epileptic` | 5,750 × 178 | 5 | Espadoto et al. (2021) |
| `hiva` | 3,076 × 1,617 | 2 | Espadoto et al. (2021) |
| `secom` | 1,567 × 590 | 2 | Espadoto et al. (2021) |
| `seismic` | 646 × 24 | 2 | Espadoto et al. (2021) |
| `spambase` | 4,601 × 57 | 2 | Espadoto et al. (2021) |
| `foresttype` | 523 × 27 | 4 | UCI Forest type mapping (id 333) |

## Sources

**Espadoto et al. (2021)** — *Toward a Quantitative Survey of Dimension
Reduction Techniques*, IEEE TVCG 27(3). The authors publish the preprocessed
arrays (`X.npy`, `y.npy`) in
[`mespadoto/proj-quant-eval`](https://github.com/mespadoto/proj-quant-eval);
we download them at the pinned commit `9f14239`. Their preprocessing
(`code/01_data_collection/get_datasets.py`) starts from the UCI / challenge
originals, one-hot encodes categorical columns (bank, seismic), replaces NaN
by 0 (secom), takes a stratified subsample with `random_state=42` (bank 5 %,
epileptic 50 %, hiva 80 %, seismic 25 %), and MinMax-scales every feature to
[0, 1] as float32. We use their published arrays rather than re-running that
pipeline, because they are exactly the arrays used in the paper.

**Forest type mapping** — UCI Machine Learning Repository, dataset 333
(CC BY 4.0, doi:10.24432/C5QP56). `download.py` tries the official UCI zip
first and falls back to a GitHub mirror pinned at a fixed commit. The mirror
has the two file names swapped; files are stored under the UCI names:
`training.csv` (198 rows) and `testing.csv` (325 rows). The processed dataset
is `training.csv` followed by `testing.csv`, raw values (no scaling), with the
class letters `d, h, o, s` encoded as `0-3`.

All raw files are pinned by SHA-256 in `_sources.py`. Both `download.py` and
`prepare.py` refuse files whose hash does not match.

## Usage

```bash
python scripts/datasets/tabular_ml/download.py --all      # -> datasets/raw_cache/tabular_ml/<id>/
python scripts/datasets/tabular_ml/prepare.py --all       # -> datasets/processed/<id>/
python scripts/datasets/tabular_ml/prepare.py --all --force
python scripts/datasets/validate_dataset.py datasets/processed/bank
```

Only the Python standard library and NumPy are required.

## Reproducibility

`features.npy` and `target.npy` are byte-identical to the arrays used in the
paper experiments. `obs.csv.gz` has identical content; the gzip header is now
written with `mtime=0` and no file name, so its hash is stable across runs.
`metadata.json` records the new provenance fields (`source_url`,
`download_urls`, `raw_sha256`, `upstream_source`, `preprocessing`).
