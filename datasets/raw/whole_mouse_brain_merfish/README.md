# Whole Mouse Brain MERFISH Raw Data

This directory holds the four H5AD files from the CELLxGENE collection
[A molecularly defined and spatially resolved cell atlas of the whole mouse brain](https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09).

```text
collection_id: 0cca8620-8dee-45d0-aef5-23f032a5cf09
collection_api: https://api.cellxgene.cziscience.com/curation/v1/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09
```

## Contents

```text
collection_metadata.json
h5ad/
  WB_MERFISH_animal1_coronal.h5ad
  WB_MERFISH_animal2_coronal.h5ad
  WB_MERFISH_animal3_sagittal.h5ad
  WB_MERFISH_animal4_sagittal.h5ad
reference_umap_figures/  # created by plot_reference_umap.py when requested
```

The H5AD files contain normalized log-expression matrices in sparse CSR form,
with 1120 gene features. They also include cell metadata in `obs`, reference
UMAP coordinates in `obsm/X_umap`, spatial coordinates in
`obsm/X_spatial_coords`, and Allen CCF coordinates in `obsm/X_CCF`.

Download the four selected H5AD files and refresh `collection_metadata.json`:

```bash
python scripts/datasets/whole_mouse_brain_merfish/download.py
```

Inspect only the source structure:

```bash
python scripts/datasets/whole_mouse_brain_merfish/inspect_h5ad.py --all
```

Prepare the dimension-reduction-ready 50-dimensional SVD features:

```bash
python scripts/datasets/whole_mouse_brain_merfish/prepare.py --all
```

The H5AD files and derived array files are intentionally ignored by Git. The
processed dataset metadata and checksums are tracked under `datasets/processed/`.
