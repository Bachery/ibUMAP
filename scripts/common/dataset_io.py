from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from common.paths import CATALOG_PATH, PROCESSED_ROOT, repo_relative


@dataclass(frozen=True)
class LabelData:
    values: np.ndarray
    source: str
    column: str | None = None
    names: list[str] | None = None


@dataclass(frozen=True)
class ProcessedDataset:
    dataset_id: str
    dataset_dir: Path
    features_path: Path
    metadata: dict[str, Any]
    catalog_entry: dict[str, Any] | None = None

    @property
    def obs_path(self) -> Path:
        obs_file = self.metadata.get("obs_file", "obs.csv.gz")
        return self.dataset_dir / str(obs_file)

    @property
    def target_path(self) -> Path:
        target_file = self.metadata.get("target_file", "target.npy")
        return self.dataset_dir / str(target_file)

    @property
    def feature_shape(self) -> tuple[int, ...] | None:
        value = self.metadata.get("feature_shape")
        if not isinstance(value, list):
            return None
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None

    @property
    def feature_dtype(self) -> str | None:
        value = self.metadata.get("feature_dtype")
        return str(value) if value is not None else None

    def load_features(
        self,
        *,
        mmap: bool = False,
        dtype: np.dtype | str | None = None,
        order: str | None = None,
    ) -> np.ndarray:
        mode = "r" if mmap else None
        array = np.load(self.features_path, mmap_mode=mode)
        if dtype is not None or order is not None:
            array = np.asarray(array, dtype=dtype, order=order)
        return array

    def read_obs(self, *, max_rows: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
        return read_obs(self.obs_path, max_rows=max_rows)

    def load_extra_array(self, key: str, *, mmap: bool = False) -> np.ndarray:
        path = resolve_extra_array_path(self, key)
        return np.load(path, mmap_mode="r" if mmap else None)

    def load_labels(
        self,
        *,
        preferred_columns: Sequence[str] | None = None,
        sample_indices: np.ndarray | Sequence[int] | None = None,
    ) -> LabelData | None:
        return load_labels(self, preferred_columns=preferred_columns, sample_indices=sample_indices)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_catalog(catalog_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(catalog_path) if catalog_path is not None else CATALOG_PATH
    return read_json(path)


def catalog_entries(catalog_path: str | Path | None = None) -> list[dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    entries = catalog.get("datasets", [])
    if not isinstance(entries, list):
        raise ValueError(f"{catalog_path or CATALOG_PATH} does not contain a datasets list")
    return [entry for entry in entries if isinstance(entry, dict)]


def catalog_entry_by_id(
    dataset_id: str,
    *,
    catalog_path: str | Path | None = None,
    required: bool = False,
) -> dict[str, Any] | None:
    for entry in catalog_entries(catalog_path):
        if entry.get("dataset_id") == dataset_id:
            return entry
    if required:
        raise KeyError(f"Dataset {dataset_id!r} is not present in {catalog_path or CATALOG_PATH}")
    return None


def resolve_processed_dataset_dir(
    dataset: str | Path,
    *,
    processed_root: str | Path | None = None,
) -> Path:
    value = Path(dataset)
    if value.exists():
        return value.resolve()

    root = Path(processed_root) if processed_root is not None else PROCESSED_ROOT
    if root.name == str(dataset) and (root / "metadata.json").exists():
        return root.resolve()
    return (root / str(dataset)).resolve()


def load_processed_dataset(
    dataset: str | Path,
    *,
    processed_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
    require_catalog_entry: bool = False,
) -> ProcessedDataset:
    dataset_dir = resolve_processed_dataset_dir(dataset, processed_root=processed_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory does not exist: {dataset_dir}")

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Processed dataset is missing metadata.json: {dataset_dir}")

    metadata = read_json(metadata_path)
    dataset_id = str(metadata.get("dataset_id") or dataset_dir.name)
    primary_array = str(metadata.get("primary_array") or "features.npy")
    features_path = dataset_dir / primary_array
    if not features_path.exists():
        raise FileNotFoundError(f"Processed dataset is missing primary array {primary_array}: {dataset_dir}")

    catalog_entry = catalog_entry_by_id(
        dataset_id,
        catalog_path=catalog_path,
        required=require_catalog_entry,
    )
    return ProcessedDataset(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        features_path=features_path,
        metadata=metadata,
        catalog_entry=catalog_entry,
    )


def read_obs(path: str | Path, *, max_rows: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    obs_path = Path(path)
    if not obs_path.exists():
        return [], []
    opener = gzip.open if obs_path.name.endswith(".gz") else open
    rows: list[dict[str, str]] = []
    with opener(obs_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            rows.append(dict(row))
    return header, rows


def choose_label_column(
    metadata: dict[str, Any],
    obs_columns: Iterable[str],
    preferred_columns: Sequence[str] | None = None,
) -> str | None:
    available = set(obs_columns)
    candidates: list[str] = []
    if preferred_columns:
        candidates.extend(str(column) for column in preferred_columns)

    default_color = metadata.get("default_color_by")
    if isinstance(default_color, str):
        candidates.append(default_color)

    label_columns = metadata.get("label_columns")
    if isinstance(label_columns, list):
        candidates.extend(str(column) for column in label_columns)

    candidates.extend(
        [
            "label_name",
            "label",
            "target",
            "cell_type",
            "plot_cell_type",
            "cell_type_name",
            "tissue",
            "organ_tissue",
            "Disease",
            "Group",
            "Seurat.clusters",
        ]
    )
    for column in candidates:
        if column in available:
            return column
    return None


def load_obs_column(
    dataset: ProcessedDataset,
    column: str,
    *,
    sample_indices: np.ndarray | Sequence[int] | None = None,
) -> np.ndarray:
    header, rows = dataset.read_obs()
    if column not in header:
        raise KeyError(f"Column {column!r} is not present in {repo_relative(dataset.obs_path)}")
    values = np.asarray([row.get(column, "") for row in rows], dtype=object)
    if sample_indices is not None:
        values = values[np.asarray(sample_indices, dtype=np.int64)]
    return values


def load_labels(
    dataset: ProcessedDataset,
    *,
    preferred_columns: Sequence[str] | None = None,
    sample_indices: np.ndarray | Sequence[int] | None = None,
) -> LabelData | None:
    if dataset.target_path.exists():
        values = np.load(dataset.target_path, mmap_mode="r")
        if sample_indices is not None:
            values = values[np.asarray(sample_indices, dtype=np.int64)]
        return LabelData(values=np.asarray(values), source=repo_relative(dataset.target_path))

    header, _ = dataset.read_obs(max_rows=0)
    column = choose_label_column(dataset.metadata, header, preferred_columns=preferred_columns)
    if column is None:
        return None
    values = load_obs_column(dataset, column, sample_indices=sample_indices)
    names = sorted({str(value) for value in values.tolist()})
    return LabelData(values=values, source=repo_relative(dataset.obs_path), column=column, names=names)


def resolve_extra_array_path(dataset: ProcessedDataset, key: str) -> Path:
    extra_arrays = dataset.metadata.get("extra_arrays")
    if isinstance(extra_arrays, dict):
        value = extra_arrays.get(key)
        if isinstance(value, dict):
            file_name = value.get("file") or key
        elif isinstance(value, str):
            file_name = value
        else:
            file_name = key
    elif isinstance(extra_arrays, list):
        file_name = key
        for item in extra_arrays:
            if isinstance(item, dict) and (item.get("key") == key or item.get("name") == key):
                file_name = item.get("file") or key
                break
            if isinstance(item, str) and item == key:
                file_name = item
                break
    else:
        file_name = key

    path = dataset.dataset_dir / str(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Extra array {key!r} does not exist: {path}")
    return path


def select_row_indices(n_rows: int, n: int | None, *, seed: int = 42) -> np.ndarray:
    if n is None or n <= 0 or n >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=int(n), replace=False)).astype(np.int64)


def sample_rows(
    array: np.ndarray,
    n: int | None,
    *,
    seed: int = 42,
    dtype: np.dtype | str | None = np.float32,
    order: str | None = "C",
) -> tuple[np.ndarray, np.ndarray]:
    row_indices = select_row_indices(int(array.shape[0]), n, seed=seed)
    subset = array[row_indices]
    if dtype is not None or order is not None:
        subset = np.asarray(subset, dtype=dtype, order=order)
    return subset, row_indices

