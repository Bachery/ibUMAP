# Whole Mouse Brain MERFISH

Scripts for the four whole-mouse-brain MERFISH datasets in CELLxGENE collection
`0cca8620-8dee-45d0-aef5-23f032a5cf09`.

## Source layout

The downloaded H5AD files live under:

```text
datasets/raw/whole_mouse_brain_merfish/h5ad/
  WB_MERFISH_animal1_coronal.h5ad
  WB_MERFISH_animal2_coronal.h5ad
  WB_MERFISH_animal3_sagittal.h5ad
  WB_MERFISH_animal4_sagittal.h5ad
```

Each file contains a normalized log-expression CSR matrix in `X` (cells by
1120 genes), categorical cell metadata in `obs`, and source coordinates in
`obsm/X_umap`, `obsm/X_spatial_coords`, and `obsm/X_CCF`.

## Inspect

`inspect_h5ad.py` only reads H5AD structural metadata. It does not create processed
data unless an explicit `--output` path is supplied.

```bash
python scripts/datasets/whole_mouse_brain_merfish/inspect_h5ad.py --all
python scripts/datasets/whole_mouse_brain_merfish/inspect_h5ad.py --dataset WB_MERFISH_animal4_sagittal
```

## Prepare

The expression matrices are too large to safely densify. `prepare.py` retains
their sparse representation while fitting a truncated SVD, then writes the
row-batched `float32` projections as `features.npy`. The default is 50
components. It also writes `obs.csv.gz`, the source UMAP, spatial coordinates,
CCF coordinates, metadata, and checksums for each animal as an independent
processed dataset.

```bash
python scripts/datasets/whole_mouse_brain_merfish/prepare.py --list
python scripts/datasets/whole_mouse_brain_merfish/prepare.py --dataset WB_MERFISH_animal4_sagittal
python scripts/datasets/whole_mouse_brain_merfish/prepare.py --all
python scripts/datasets/whole_mouse_brain_merfish/prepare.py --all --svd-components 32 --force
```

The generated dataset IDs are:

- `whole_mouse_brain_merfish_animal1_coronal`
- `whole_mouse_brain_merfish_animal2_coronal`
- `whole_mouse_brain_merfish_animal3_sagittal`
- `whole_mouse_brain_merfish_animal4_sagittal`

## Source UMAP figures

The source UMAP can be plotted directly from H5AD without first preparing a
dataset. Figures are written to
`datasets/raw/whole_mouse_brain_merfish/reference_umap_figures/` and are capped
at one million points by default.

```bash
python scripts/datasets/whole_mouse_brain_merfish/plot_reference_umap.py --dataset WB_MERFISH_animal4_sagittal
python scripts/datasets/whole_mouse_brain_merfish/plot_reference_umap.py --all --color-by subclass_transfer
```
