# Datasets

| Directory | Tracked | Content |
| --- | --- | --- |
| `reference/<id>/` | yes | Record of the processed copy used in the paper: its `metadata.json` and a `checksums.txt` with the SHA-256 of every data file (for `*.gz`, of the decompressed content) |
| `processed/<id>/` | no | Prepared datasets: `features.npy`, `obs.csv.gz`, optional `target.npy` and extra arrays, `metadata.json`, `checksums.txt` |
| `raw/`, `raw_cache/` | READMEs only | Downloaded source files |
| `cache/` | no | Scratch space of the preparation scripts |
| `catalog.json` | yes | All 71 datasets with their shapes and families |

The download and preparation scripts are in `scripts/datasets/<family>/` (see
[`scripts/datasets/README.md`](../scripts/datasets/README.md)). They write to
`processed/<id>/`, which starts empty in a fresh checkout. To prepare a dataset
and compare it with the paper's copy:

```bash
python scripts/datasets/classic_benchmarks/prepare.py iris
python scripts/datasets/validate_dataset.py --reference datasets/processed/iris
```

`--reference` requires every data file listed in `reference/<id>/checksums.txt`
to be present with the recorded content hash, and the structural fields of
`metadata.json` (dataset id, family, shapes, dtypes, file names) to agree. Other
metadata fields, such as the generation time, are reported but not compared.
The experiments read `processed/<id>/`; `reference/` is never modified by the
scripts.
