# Tabular ML Raw Cache

Re-downloadable raw inputs for `scripts/datasets/tabular_ml/`:

- `<dataset_id>/X.npy`, `<dataset_id>/y.npy` for bank, epileptic, hiva,
  secom, seismic and spambase (Espadoto et al. 2021, pinned commit)
- `foresttype/training.csv`, `foresttype/testing.csv` (UCI dataset 333)

Recreate with:

```bash
python scripts/datasets/tabular_ml/download.py --all
```

Every file is verified against the SHA-256 values in
`scripts/datasets/tabular_ml/_sources.py`.
