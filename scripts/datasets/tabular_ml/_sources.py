"""Pinned download sources for the tabular ML dataset family.

Six datasets (bank, epileptic, hiva, secom, seismic, spambase) are the
preprocessed arrays published with Espadoto et al., "Toward a Quantitative
Survey of Dimension Reduction Techniques" (IEEE TVCG 2021). They are fetched
from the authors' repository at a pinned commit, so the bytes are fixed.

Forest Type Mapping is the UCI dataset (id 333). The official UCI zip is tried
first; a public GitHub mirror at a pinned commit is the fallback. Text files are
stored with LF line endings, and the SHA-256 values pin those normalized bytes,
so both routes must yield identical files. Note that the mirror has the two file
names swapped relative to UCI: UCI training.csv (198 rows) is the mirror's
testing.csv and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass


ESPADOTO_REPO = "mespadoto/proj-quant-eval"
ESPADOTO_COMMIT = "9f14239f87bbacc6cac6ff52729284dd65956dd6"
ESPADOTO_RAW_BASE = f"https://raw.githubusercontent.com/{ESPADOTO_REPO}/{ESPADOTO_COMMIT}/docs/data"
ESPADOTO_PAGES_BASE = "https://mespadoto.github.io/proj-quant-eval/data"
ESPADOTO_LANDING_PAGE = "https://mespadoto.github.io/proj-quant-eval/post/datasets/"
ESPADOTO_PREPROCESSING_SCRIPT = (
    f"https://github.com/{ESPADOTO_REPO}/blob/{ESPADOTO_COMMIT}/code/01_data_collection/get_datasets.py"
)
ESPADOTO_PUBLICATION = (
    "M. Espadoto, R. M. Martins, A. Kerren, N. S. T. Hirata, A. C. Telea. "
    "Toward a Quantitative Survey of Dimension Reduction Techniques. "
    "IEEE Transactions on Visualization and Computer Graphics 27(3), 2021. "
    "doi:10.1109/TVCG.2019.2944182"
)

FOREST_MIRROR_REPO = "rupakc/UCI-Data-Analysis"
FOREST_MIRROR_COMMIT = "b2937bc0f36f570b5a6bb4049656f5239a1f334c"
FOREST_MIRROR_BASE = (
    f"https://raw.githubusercontent.com/{FOREST_MIRROR_REPO}/{FOREST_MIRROR_COMMIT}"
    "/Forest%20Mapping%20Dataset/Forest%20Type%20Mapping"
)
FOREST_UCI_PAGE = "https://archive.ics.uci.edu/dataset/333/forest+type+mapping"
FOREST_UCI_ZIP = "https://archive.ics.uci.edu/static/public/333/forest+type+mapping.zip"
FOREST_PUBLICATION = (
    "B. Johnson (2012). Forest type mapping [Dataset]. UCI Machine Learning Repository. "
    "doi:10.24432/C5QP56 (CC BY 4.0)"
)


@dataclass(frozen=True)
class SourceFile:
    file_name: str
    sha256: str
    urls: tuple[str, ...]
    text: bool = False  # normalize CRLF -> LF before hashing and storing


@dataclass(frozen=True)
class ArchiveSource:
    """A zip archive that contains several SourceFiles (matched by base name)."""

    url: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    display_name: str
    loader: str  # "espadoto_npy" or "uci_forest_csv"
    files: tuple[SourceFile, ...]
    source_name: str
    source_url: str
    source_publication: str
    upstream: str
    upstream_preprocessing: str
    archives: tuple[ArchiveSource, ...] = ()


def _espadoto_file(dataset_id: str, file_name: str, sha256: str) -> SourceFile:
    return SourceFile(
        file_name=file_name,
        sha256=sha256,
        urls=(
            f"{ESPADOTO_RAW_BASE}/{dataset_id}/{file_name}",
            f"{ESPADOTO_PAGES_BASE}/{dataset_id}/{file_name}",
        ),
    )


def _espadoto(
    dataset_id: str,
    display_name: str,
    x_sha256: str,
    y_sha256: str,
    upstream: str,
    upstream_preprocessing: str,
) -> DatasetSource:
    return DatasetSource(
        dataset_id=dataset_id,
        display_name=display_name,
        loader="espadoto_npy",
        files=(
            _espadoto_file(dataset_id, "X.npy", x_sha256),
            _espadoto_file(dataset_id, "y.npy", y_sha256),
        ),
        source_name="Espadoto et al. (2021) DR survey benchmark",
        source_url=ESPADOTO_LANDING_PAGE,
        source_publication=ESPADOTO_PUBLICATION,
        upstream=upstream,
        upstream_preprocessing=(
            upstream_preprocessing
            + " Features are then MinMax-scaled to [0, 1] as float32 "
            f"(see {ESPADOTO_PREPROCESSING_SCRIPT})."
        ),
    )


_STRATIFIED = "stratified subsample with sklearn train_test_split(train_size={frac}, random_state=42)"

DATASETS: dict[str, DatasetSource] = {
    "bank": _espadoto(
        "bank",
        "Bank Marketing",
        "692834099ce40c6b64f151ae687ec208fa6f33b8e43f0d9c9e55f6dff5496f20",
        "786cd4cdf46f53822b29a188ea6988dfff9a7fa33abaaa3572e5f2c301d0007f",
        "UCI Bank Marketing (id 222), bank-additional-full.csv, 41,188 rows",
        "Target y == 'yes'; remaining columns one-hot encoded with pandas.get_dummies; "
        + _STRATIFIED.format(frac=0.05)
        + " -> 2,059 rows.",
    ),
    "epileptic": _espadoto(
        "epileptic",
        "Epileptic Seizure Recognition",
        "1172ff7e720ebcde5b4ec5409fc2ac6c62e112e3672c9c465f31eea95d081491",
        "0bdf78ab781e496b55ca81aaef8a72d53a862b6a5525df6fcd5a58c38f6de8bd",
        "UCI Epileptic Seizure Recognition (id 388), data.csv, 11,500 x 178, 5 classes",
        "Index column dropped; " + _STRATIFIED.format(frac=0.5) + " -> 5,750 rows.",
    ),
    "hiva": _espadoto(
        "hiva",
        "HIVA",
        "244e1f7591cca5ecfeaed966d6faf72f2a2aa06b3209485942049d0a7d4d0e9b",
        "30b237c0a07f63936010614277fe24f0cc5605f02a7bbb0dfd0b439732da38b2",
        "HIVA, Agnostic Learning vs. Prior Knowledge challenge (2007), hiva_train, 3,845 x 1,617",
        _STRATIFIED.format(frac=0.8) + " -> 3,076 rows; labels are -1/+1.",
    ),
    "secom": _espadoto(
        "secom",
        "SECOM",
        "1cebcd793296abcd444191b5474508427d67388361cf206ad36384e9ad80ba9a",
        "57b8ef466725e3485074174478a9a10ff55436d82299bc28b912ef3f9ed5ac91",
        "UCI SECOM (id 179), secom.data + secom_labels.data, 1,567 x 590",
        "Missing values replaced by 0; no subsampling; labels are -1/+1.",
    ),
    "seismic": _espadoto(
        "seismic",
        "Seismic Bumps",
        "2855b449fe37839b5d1ffa9dc93004ebbe38ce60d62d3748806e661ff343abd1",
        "bbca356dd4ef89dbd5b87ec6a54fd65303409f49d7fd4d4c91d8191d69b80954",
        "UCI seismic-bumps (id 266), seismic-bumps.arff, 2,584 rows",
        "Categorical columns one-hot encoded with pandas.get_dummies; "
        + _STRATIFIED.format(frac=0.25)
        + " -> 646 rows.",
    ),
    "spambase": _espadoto(
        "spambase",
        "Spambase",
        "17bdd8c61aa4b2b10b752975528d45b72bd11a04298e12ca0e8522d73c2d921f",
        "a66e1124211f7df80e1e2bfd3dccb3e18e9dd93dd97bc8576f077ed84326ac6b",
        "UCI Spambase (id 94), spambase.data, 4,601 x 57",
        "No subsampling.",
    ),
    "foresttype": DatasetSource(
        dataset_id="foresttype",
        display_name="Forest Type Mapping",
        loader="uci_forest_csv",
        # Stored under the UCI names. Row order of the processed dataset:
        # training.csv (198 rows), then testing.csv (325 rows).
        files=(
            SourceFile(
                "training.csv",
                "d63da464655f4ec7c95bb310f560adecc7b716258bc379e4e0ca59f91a5ebf18",
                (f"{FOREST_MIRROR_BASE}/testing.csv",),  # mirror name is swapped
                text=True,
            ),
            SourceFile(
                "testing.csv",
                "8f1dc6b2e437122f958d0ec96b81c32f5c175b57ad901de8fe4052e2443bd315",
                (f"{FOREST_MIRROR_BASE}/training.csv",),  # mirror name is swapped
                text=True,
            ),
        ),
        archives=(ArchiveSource(FOREST_UCI_ZIP, ("training.csv", "testing.csv")),),
        source_name="UCI Machine Learning Repository: Forest type mapping (id 333)",
        source_url=FOREST_UCI_PAGE,
        source_publication=FOREST_PUBLICATION,
        upstream=(
            f"UCI Forest type mapping (id 333), {FOREST_UCI_ZIP}; "
            f"fallback mirror {FOREST_MIRROR_REPO}@{FOREST_MIRROR_COMMIT}"
        ),
        upstream_preprocessing=(
            "UCI training.csv (198 rows) and testing.csv (325 rows) concatenated in that order "
            "(523 rows, 27 numeric columns b1-b9, "
            "pred_minus_obs_H_b1-b9, pred_minus_obs_S_b1-b9); class letters (d, h, o, s) stripped of "
            "whitespace and encoded as sorted category codes 0-3; no scaling."
        ),
    ),
}

FOREST_CLASS_DESCRIPTIONS = {
    "d": "Mixed deciduous forest",
    "h": "Hinoki forest",
    "o": "Other (non-forest) land",
    "s": "Sugi forest",
}
