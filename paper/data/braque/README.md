# BRAQUE derived results: source and license

These files contain ibUMAP experiment results derived from sample `L2` of:

Dall'Olio, Lorenzo; Bolognesi, Maddalena; Borghesi, Simone; Cattoretti, Giorgio;
Castellani, Gastone (2023). *BRAQUE: Bayesian Reduction for Amplified Quantization
in UMAP Embedding. Supplementary data.* Mendeley Data, V1.
https://doi.org/10.17632/j8xbwb93x9.1

The source dataset is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Verified on 2026-09-29 against the official
[Mendeley V1 metadata](https://data.mendeley.com/public-api/datasets/j8xbwb93x9?version=1),
whose `data_licence.short_name` is `CC BY 4.0`.

Changes: the workflow selects markers, applies reimplemented Lognormal
Shrinkage and robust standardization, computes UMAP embeddings and HDBSCAN
partitions, and exports coordinates, cluster labels, and repeatability and cost
summaries. Raw marker intensities and the antibody workbook are not included.
See the [experiment protocol](../../../experiments/braque/README.md).

Retain the source attribution, license link, and indication of these changes
when sharing these derived data. The repository's software license does not
replace the source dataset's CC BY 4.0 terms. No endorsement by the dataset
authors is implied.
