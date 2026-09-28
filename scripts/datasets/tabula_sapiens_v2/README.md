# Tabula Sapiens v2

Scripts for preparing Tabula Sapiens v2 single-cell datasets from the CELLxGENE
collection:

```text
collection_id: e5f58829-1a66-40b5-a624-9046778e74f5
collection_api: https://api.cellxgene.cziscience.com/curation/v1/collections/e5f58829-1a66-40b5-a624-9046778e74f5
```

This family is intentionally implemented without `scanpy`. The scripts use
standard download tools plus `h5py`, `pandas`, and `numpy` to read H5AD content.

## Directory Layout

Raw files:

```text
datasets/raw/tabula_sapiens_v2/
  collection_metadata.json
  h5ad/
    TS2_All_Cells.h5ad
    TS2_Immune.h5ad
    ...
  reference_umap_figures/
    <dataset_id>_umap.png
```

Processed files:

```text
datasets/processed/tabula_sapiens_v2_all_cells/
  features.npy
  obs.csv.gz
  reference_embedding.npy
  metadata.json
  checksums.txt
```

Each H5AD subset is treated as an independent processed dataset and registered
with its own `dataset_id`.

## Naming

Raw names are converted to safe processed IDs by removing the `TS2_` prefix,
converting the remainder to snake case, and adding the shared prefix
`tabula_sapiens_v2_`.

Examples:

```text
TS2_All_Cells.h5ad  -> tabula_sapiens_v2_all_cells
TS2_Immune.h5ad     -> tabula_sapiens_v2_immune
TS2_Stromal.h5ad    -> tabula_sapiens_v2_stromal
TS2_Epithelium.h5ad -> tabula_sapiens_v2_epithelium
```

## Download

```bash
python scripts/datasets/tabula_sapiens_v2/download.py --list-only
python scripts/datasets/tabula_sapiens_v2/download.py --limit 1
python scripts/datasets/tabula_sapiens_v2/download.py
python scripts/datasets/tabula_sapiens_v2/download.py --force
```

`download.py` stores the collection API response as
`datasets/raw/tabula_sapiens_v2/collection_metadata.json` and streams H5AD files
into `datasets/raw/tabula_sapiens_v2/h5ad/`. Existing files are skipped unless
`--force` is used.

Do not hand-edit raw H5AD files. Re-download them instead.

## Prepare

```bash
python scripts/datasets/tabula_sapiens_v2/prepare.py --list
python scripts/datasets/tabula_sapiens_v2/prepare.py --dataset TS2_Immune
python scripts/datasets/tabula_sapiens_v2/prepare.py --dataset tabula_sapiens_v2_immune
python scripts/datasets/tabula_sapiens_v2/prepare.py --all
python scripts/datasets/tabula_sapiens_v2/prepare.py --all --force
```

`prepare.py` reads each H5AD with `h5py`:

- `features.npy` comes from `obsm/X_scvi` and is stored as `float32`.
- `obs.csv.gz` stores serializable H5AD `obs` metadata with `sample_id`.
- H5AD obs index values are saved as `original_obs_index` when available.
- `reference_embedding.npy` comes from `obsm/X_umap` when present.
- Missing `obsm/X_umap` is allowed and recorded in metadata.
- Missing `obsm/X_scvi` causes that H5AD file to be skipped.

## Plot Reference UMAP

```bash
python scripts/datasets/tabula_sapiens_v2/plot_reference_umap.py --dataset tabula_sapiens_v2_immune
python scripts/datasets/tabula_sapiens_v2/plot_reference_umap.py --dataset tabula_sapiens_v2_immune --color-by cell_type
python scripts/datasets/tabula_sapiens_v2/plot_reference_umap.py --all
```

Figures are written to `datasets/raw/tabula_sapiens_v2/reference_umap_figures/`,
not to processed dataset directories.

## Legacy Mapping

The older script wrote:

```text
X.npy
labels.csv
stored_umap_embedding.npy
```

The new layout writes:

```text
features.npy
obs.csv.gz
reference_embedding.npy
```
