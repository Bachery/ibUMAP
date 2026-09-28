# scDEED

Scripts for inspecting and preparing datasets associated with the scDEED paper:

```text
Statistical method scDEED for detecting dubious 2D single-cell embeddings and optimizing t-SNE and UMAP hyperparameters
Zenodo record: 7216361
Source URL: https://zenodo.org/records/7216361
```

The Zenodo archive may contain a mixture of R objects, H5AD/Loom files, CSV/TSV
files, NumPy arrays, and archives. This family therefore uses an inspect-first
workflow instead of assuming every file is directly convertible.

## Layout

```text
datasets/raw/scdeed/
  zenodo_metadata.json
  downloads/
    <Zenodo files>
  extracted/
    <optional extracted archives>
  inspect_manifest.json
  README.md

datasets/processed/scdeed_<dataset_id>/
  features.npy
  obs.csv.gz
  metadata.json
  checksums.txt
```

Optional reference outputs, when clearly identified, may be written under:

```text
reference_embeddings/
reference_metrics/
```

## Download

```bash
python scripts/datasets/scdeed/download.py --list-only
python scripts/datasets/scdeed/download.py --limit 1
python scripts/datasets/scdeed/download.py
python scripts/datasets/scdeed/download.py --force
```

`download.py` always stores Zenodo API metadata in
`datasets/raw/scdeed/zenodo_metadata.json`. File downloads stream to
`datasets/raw/scdeed/downloads/` and verify Zenodo checksums when provided.

## Inspect

```bash
python scripts/datasets/scdeed/inspect.py
python scripts/datasets/scdeed/inspect.py --extract
python scripts/datasets/scdeed/inspect.py --limit 20
python scripts/datasets/scdeed/inspect.py --count-large-text
```

`inspect.py` scans only `datasets/raw/scdeed/downloads/` and
`datasets/raw/scdeed/extracted/`. It does not read from `data/Samusik/` or any
legacy `data_raw/scDEED/` directory. It writes
`datasets/raw/scdeed/inspect_manifest.json` with file types, array shapes,
H5AD/Loom summaries, CSV previews, archive members, and candidate roles. Large
text matrices only have their headers inspected by default; use
`--count-large-text` when a full row count is needed. R files are only read when
`pyreadr` is already installed; the script does not install R dependencies.

## Prepare

```bash
python scripts/datasets/scdeed/prepare.py --list
python scripts/datasets/scdeed/prepare.py --dataset marrow
python scripts/datasets/scdeed/prepare.py --dataset velocity
python scripts/datasets/scdeed/prepare.py --dataset samusik
python scripts/datasets/scdeed/prepare.py --dataset cart
python scripts/datasets/scdeed/prepare.py --all
python scripts/datasets/scdeed/prepare.py --all --force
```

`prepare.py` consumes `inspect_manifest.json` and only converts recognized
feature/obs pairs from downloaded and extracted Zenodo raw files. It does not
force unknown files into processed datasets.

Currently recognized raw candidates:

- `marrow`: reads `Marrow_processed.h5ad`, uses `obsm/X_pca` as
  `features.npy`, and stores `obsm/X_tsne` as a reference embedding.
- `velocity`: reads `10X43_1.loom`, transposes the source gene-by-cell matrix
  to cells by genes, stores loom `col_attrs` as obs metadata, and stores
  Cell Ranger PCA/t-SNE coordinates as reference embeddings.
- `samusik`: reads `xp.RData:xp` as `features.npy`, `label.RData:cluster_label`
  as obs labels, and stores raw UMAP/loadings R objects as reference embeddings
  plus dubious-number R objects as reference metrics.
- `cart`: reads `normalised_counts.tsv`, transposes the source gene-by-cell
  matrix to cells by genes, joins `metadata_grey.tsv` and `meta-data.txt`, and
  stores t-SNE coordinates plus the scDEED dubious-number table.

The current Python-only workflow cannot safely prepare `alveolar`,
`across_techniques`, or `simulated_data` because their primary feature/obs
objects are Seurat/custom R objects that `pyreadr` reports as unrecognized or
unsupported. They should be handled with a separate R/Seurat extraction step
before conversion to the processed format.

`features.npy` is the UMAP input matrix stored as `float32`; `obs.csv.gz` is
sample-level metadata with at least `sample_id`, and any labels or scores that
can be safely matched row-by-row.
