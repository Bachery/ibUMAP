# Raw C. elegans embryogenesis H5AD

Target file:

```text
datasets/raw/c_elegans_embryogenesis/packer2019.h5ad
```

Source:

- CaltechDATA record: <https://data.caltech.edu/records/b1kj4-nh475>
- File: `packer2019.h5ad`
- Size on record page: 682.3 MB
- Expected md5: `e2cda8a6cee91d1a326dea5b9e4d2539`
- Source name: Packer 2019 C. elegans embryogenesis

This directory should also contain:

```text
source_metadata.json
raw_metadata_summary.json
```

`source_metadata.json` is written by `download.py` after a successful download,
copy, or link. `raw_metadata_summary.json` is written by:

```bash
python scripts/datasets/c_elegans_embryogenesis/prepare.py --summary
```

The H5AD file itself is large and should stay out of git.
