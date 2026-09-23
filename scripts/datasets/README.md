# Dataset Scripts

This directory owns the dataset lifecycle: download raw assets, prepare
standardized processed datasets, write metadata/checksums, validate data
contracts, rebuild the dataset catalog, and plot dataset-level summaries.

Keep long-running algorithm experiments, embedding quality evaluation, and
experiment-facing data loaders outside this directory.

Run commands from the repository root:

```bash
cd ibUMAP
```

## Directory Boundary

- `scripts/datasets/<family>/`: family-specific `download.py`, `prepare.py`,
  and local README files.
- `scripts/datasets/common/`: helpers used by dataset preparation and
  validation, such as metadata, checksums, and validation code.

## `validate_dataset.py`

Validates standardized dataset assets. The default scope is `processed`, so the
script can be used on benchmark machines that only have `datasets/processed/`
and do not keep large raw downloads.

Validate one processed dataset:

```bash
python scripts/datasets/validate_dataset.py datasets/processed/mnist784
```

This is equivalent to:

```bash
python scripts/datasets/validate_dataset.py --scope processed datasets/processed/mnist784
```

Validate all processed datasets:

```bash
python scripts/datasets/validate_dataset.py --all
```

Validate raw files or raw directories explicitly:

```bash
python scripts/datasets/validate_dataset.py --scope raw datasets/raw
python scripts/datasets/validate_dataset.py --scope raw datasets/raw/scdeed
```

Run full provenance validation when both processed and raw assets are present:

```bash
python scripts/datasets/validate_dataset.py --scope full datasets/processed/tabula_sapiens_v2_immune
```

Notes:

- `processed` checks processed metadata, arrays, obs/target files, extra arrays,
  reference embeddings, and processed `checksums.txt` when present.
- `processed` does not fail if `raw_path` or `raw_paths` are absent.
- `raw` checks raw path existence, known JSON metadata/manifest files, and raw
  `checksums.txt` when present.
- `full` runs processed validation and then requires declared raw provenance to
  exist.

## `build_catalog.py`

Builds `datasets/catalog.json` from `metadata.json` files under
`datasets/processed/`.

Default usage:

```bash
python scripts/datasets/build_catalog.py
```

Use a custom processed root or output path:

```bash
python scripts/datasets/build_catalog.py \
  --processed-root datasets/processed \
  --output datasets/catalog.json
```

## `plot_dataset_shapes.py`

Reads `datasets/catalog.json` and writes a PNG scatter plot of dataset row and
column counts.

Default usage:

```bash
python scripts/datasets/plot_dataset_shapes.py
```

Use a custom catalog, output path, or linear axes:

```bash
python scripts/datasets/plot_dataset_shapes.py \
  --catalog datasets/catalog.json \
  --output datasets/dataset_shapes_scatter.png

python scripts/datasets/plot_dataset_shapes.py --linear
```
