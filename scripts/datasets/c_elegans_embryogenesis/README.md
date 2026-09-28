# C. elegans embryogenesis dataset preparation

This directory formalizes the Packer 2019 C. elegans embryogenesis single-cell
H5AD dataset under the shared `datasets/` layout.

## Source

- Source page: <https://data.caltech.edu/records/b1kj4-nh475>
- File: `packer2019.h5ad`
- Expected raw target: `datasets/raw/c_elegans_embryogenesis/packer2019.h5ad`
- CaltechDATA DOI: `10.22002/D1.1945`
- Expected md5 from the record page: `e2cda8a6cee91d1a326dea5b9e4d2539`

The raw H5AD is about 682 MB. Do not commit it to git.

## Obtaining the raw file

Use the direct CaltechDATA downloader:

```bash
python scripts/datasets/c_elegans_embryogenesis/download.py --list-only
python scripts/datasets/c_elegans_embryogenesis/download.py
python scripts/datasets/c_elegans_embryogenesis/download.py --force
```

If the direct API download ever changes, open the source page manually,
download `packer2019.h5ad`, and place it at:

```text
datasets/raw/c_elegans_embryogenesis/packer2019.h5ad
```

`download.py` writes `datasets/raw/c_elegans_embryogenesis/source_metadata.json`
after a successful download.

## Processed layout

Each full dataset or subset is written as an independent processed dataset under
`datasets/processed/<dataset_id>/` and can enter `datasets/catalog.json`.

Full datasets:

- `c_elegans_embryogenesis_qc_passed`: all cells where `passed_qc == True`.
- `c_elegans_embryogenesis_all_cells`: all cells, without QC filtering.

Trajectory-oriented subsets:

- `c_elegans_embryogenesis_global_qc_annotated`
- `c_elegans_embryogenesis_stage_early_annotated`
- `c_elegans_embryogenesis_stage_mid_annotated`
- `c_elegans_embryogenesis_stage_late_annotated`
- `c_elegans_embryogenesis_cell_type_top_4_mixture`
- `c_elegans_embryogenesis_plot_cell_type_top_8_mixture`
- `c_elegans_embryogenesis_pair_body_wall_muscle__hypodermis`
- `c_elegans_embryogenesis_pair_body_wall_muscle__ciliated_amphid_neuron`
- `c_elegans_embryogenesis_pair_body_wall_muscle__ciliated_non_amphid_neuron`
- `c_elegans_embryogenesis_pair_hypodermis__ciliated_amphid_neuron`

Each processed directory contains:

```text
features.npy
obs.csv.gz
pseudotime.npy
source_indices.npy
dataset_summary.json or subset_summary.json
metadata.json
checksums.txt
```

## Preprocessing

`prepare.py` reads the H5AD directly with `h5py`, `numpy`, `pandas`, and
`scipy`. It does not depend on `scanpy`. Sparse `X` is read from the H5AD,
subset by original cell index, normalized by `size_factor` when present, passed
through `log1p`, and reduced with sparse truncated SVD. `features.npy` is saved
as `float32`, with 32 components by default.

`pseudotime.npy` is saved from `embryo_time` as `float32`, and
`source_indices.npy` stores the original H5AD cell indices as `int64`.
`obs.csv.gz` keeps serializable cell metadata and adds:

- `sample_id`: row index in the processed dataset.
- `original_cell_index`: row or column index of the cell in the raw H5AD.

The script can detect both common H5AD layouts: cell metadata in `obs` with
cells along rows, or cell metadata in `var` with cells along columns.

## Commands

Low-risk inspection:

```bash
python scripts/datasets/c_elegans_embryogenesis/prepare.py --list-subsets
python scripts/datasets/c_elegans_embryogenesis/prepare.py --summary
```

Prepare selected outputs:

```bash
python scripts/datasets/c_elegans_embryogenesis/prepare.py --dataset qc_passed
python scripts/datasets/c_elegans_embryogenesis/prepare.py --dataset all_cells
python scripts/datasets/c_elegans_embryogenesis/prepare.py --dataset global_qc_annotated
python scripts/datasets/c_elegans_embryogenesis/prepare.py --all-default-subsets
python scripts/datasets/c_elegans_embryogenesis/prepare.py --all
python scripts/datasets/c_elegans_embryogenesis/prepare.py --dataset qc_passed --pca-components 32 --force
```

Validate and rebuild catalog after preparation:

```bash
python scripts/datasets/validate_dataset.py --all
python scripts/datasets/build_catalog.py
```
