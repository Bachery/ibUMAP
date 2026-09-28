from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
except ImportError:  # pragma: no cover - exercised only on minimal environments
    sp = None
    svds = None


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import write_checksums
from common.metadata import utc_now_iso, write_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "datasets/raw/c_elegans_embryogenesis"
RAW_PATH = RAW_ROOT / "packer2019.h5ad"
RAW_METADATA_SUMMARY_PATH = RAW_ROOT / "raw_metadata_summary.json"
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"

FAMILY = "c_elegans_embryogenesis"
SOURCE_NAME = "Packer 2019 C. elegans embryogenesis"
SOURCE_URL = "https://data.caltech.edu/records/b1kj4-nh475"
RAW_ROLE = "primary_source"
FEATURE_SOURCE = "normalized_log_expression_pca_svd"
DEFAULT_PCA_COMPONENTS = 32
SCRIPT_VERSION = 1

MISSING_CATEGORY_VALUES = {"", "nan", "none", "na", "not provided", "unknown", "null"}
CELL_METADATA_COLUMNS = {
    "barcode",
    "batch",
    "cell_subtype",
    "cell_type",
    "embryo_time",
    "embryo_time_bin",
    "lineage",
    "n_umi",
    "passed_qc",
    "plot_cell_type",
    "raw_embryo_time",
    "raw_embryo_time_bin",
    "sample",
    "sample_batch",
    "sample_description",
    "size_factor",
    "study",
    "time_point",
}
LABEL_PRIORITY = ["cell_type", "plot_cell_type", "lineage", "embryo_time_bin", "time_point"]
DEFAULT_COLOR_PRIORITY = ["cell_type", "plot_cell_type", "lineage"]

CELL_TYPE_TOP_4 = [
    "Body_wall_muscle",
    "Hypodermis",
    "Ciliated_amphid_neuron",
    "Ciliated_non_amphid_neuron",
]
PLOT_CELL_TYPE_TOP_8 = [
    "BWM_posterior",
    "BWM_anterior",
    "Seam_cell",
    "BWM_head_row_1",
    "Hypodermis",
    "BWM_head_row_2",
    "pm3_pm4_pm5",
    "hyp7_C_lineage",
]
PAIR_SPECS = [
    ("pair_body_wall_muscle__hypodermis", "Body_wall_muscle", "Hypodermis"),
    ("pair_body_wall_muscle__ciliated_amphid_neuron", "Body_wall_muscle", "Ciliated_amphid_neuron"),
    (
        "pair_body_wall_muscle__ciliated_non_amphid_neuron",
        "Body_wall_muscle",
        "Ciliated_non_amphid_neuron",
    ),
    ("pair_hypodermis__ciliated_amphid_neuron", "Hypodermis", "Ciliated_amphid_neuron"),
]


@dataclass(frozen=True)
class DatasetSpec:
    subset_id: str
    dataset_id: str
    display_name: str
    subset_kind: str
    filter_description: str
    label_field: str | None
    selected_labels: tuple[str, ...] | None = None
    require_passed_qc: bool = True
    require_informative_label: bool = False
    embryo_time_min: float | None = None
    embryo_time_max: float | None = None

    @property
    def summary_file(self) -> str:
        if self.subset_kind in {"full_qc", "full_all"}:
            return "dataset_summary.json"
        return "subset_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Packer 2019 C. elegans embryogenesis H5AD into processed datasets."
    )
    parser.add_argument("--list-subsets", action="store_true", help="List supported dataset/subset names.")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Read raw H5AD metadata and write raw_metadata_summary.json only.",
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Prepare one dataset key or dataset_id, e.g. qc_passed, all_cells, "
            "global_qc_annotated, or c_elegans_embryogenesis_qc_passed."
        ),
    )
    parser.add_argument(
        "--all-default-subsets",
        action="store_true",
        help="Prepare qc_passed and the default trajectory-oriented subsets, excluding all_cells.",
    )
    parser.add_argument("--all", action="store_true", help="Prepare qc_passed, all_cells, and all subsets.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild selected processed outputs.")
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def dataset_id_for(subset_id: str) -> str:
    return f"{FAMILY}_{subset_id}"


def build_specs() -> dict[str, DatasetSpec]:
    specs = [
        DatasetSpec(
            subset_id="qc_passed",
            dataset_id=dataset_id_for("qc_passed"),
            display_name="C. elegans embryogenesis QC-passed cells",
            subset_kind="full_qc",
            filter_description="passed_qc == True; no annotation requirement.",
            label_field=None,
            require_passed_qc=True,
        ),
        DatasetSpec(
            subset_id="all_cells",
            dataset_id=dataset_id_for("all_cells"),
            display_name="C. elegans embryogenesis all cells",
            subset_kind="full_all",
            filter_description="All cells from the raw H5AD; no QC or annotation filtering.",
            label_field=None,
            require_passed_qc=False,
        ),
        DatasetSpec(
            subset_id="global_qc_annotated",
            dataset_id=dataset_id_for("global_qc_annotated"),
            display_name="C. elegans embryogenesis global QC annotated",
            subset_kind="global",
            filter_description="passed_qc == True and informative cell_type annotation.",
            label_field="cell_type",
            selected_labels=None,
            require_passed_qc=True,
            require_informative_label=True,
        ),
        DatasetSpec(
            subset_id="stage_early_annotated",
            dataset_id=dataset_id_for("stage_early_annotated"),
            display_name="C. elegans embryogenesis early annotated stage",
            subset_kind="stage_window",
            filter_description="passed_qc == True, informative cell_type, embryo_time < 270.",
            label_field="cell_type",
            selected_labels=None,
            require_passed_qc=True,
            require_informative_label=True,
            embryo_time_max=270.0,
        ),
        DatasetSpec(
            subset_id="stage_mid_annotated",
            dataset_id=dataset_id_for("stage_mid_annotated"),
            display_name="C. elegans embryogenesis mid annotated stage",
            subset_kind="stage_window",
            filter_description=(
                "passed_qc == True, informative cell_type, "
                "270 <= embryo_time < 430."
            ),
            label_field="cell_type",
            selected_labels=None,
            require_passed_qc=True,
            require_informative_label=True,
            embryo_time_min=270.0,
            embryo_time_max=430.0,
        ),
        DatasetSpec(
            subset_id="stage_late_annotated",
            dataset_id=dataset_id_for("stage_late_annotated"),
            display_name="C. elegans embryogenesis late annotated stage",
            subset_kind="stage_window",
            filter_description="passed_qc == True, informative cell_type, embryo_time >= 430.",
            label_field="cell_type",
            selected_labels=None,
            require_passed_qc=True,
            require_informative_label=True,
            embryo_time_min=430.0,
        ),
        DatasetSpec(
            subset_id="cell_type_top_4_mixture",
            dataset_id=dataset_id_for("cell_type_top_4_mixture"),
            display_name="C. elegans embryogenesis top 4 cell-type mixture",
            subset_kind="cell_type_mixture",
            filter_description="passed_qc == True and cell_type in the fixed top-4 branch list.",
            label_field="cell_type",
            selected_labels=tuple(CELL_TYPE_TOP_4),
            require_passed_qc=True,
        ),
        DatasetSpec(
            subset_id="plot_cell_type_top_8_mixture",
            dataset_id=dataset_id_for("plot_cell_type_top_8_mixture"),
            display_name="C. elegans embryogenesis top 8 plot-cell-type mixture",
            subset_kind="plot_cell_type_mixture",
            filter_description="passed_qc == True and plot_cell_type in the fixed top-8 branch list.",
            label_field="plot_cell_type",
            selected_labels=tuple(PLOT_CELL_TYPE_TOP_8),
            require_passed_qc=True,
        ),
    ]
    for subset_id, left, right in PAIR_SPECS:
        specs.append(
            DatasetSpec(
                subset_id=subset_id,
                dataset_id=dataset_id_for(subset_id),
                display_name=(
                    "C. elegans embryogenesis "
                    f"{left.replace('_', ' ')} vs {right.replace('_', ' ')}"
                ),
                subset_kind="cell_type_pair",
                filter_description=f"passed_qc == True and cell_type in [{left}, {right}].",
                label_field="cell_type",
                selected_labels=(left, right),
                require_passed_qc=True,
            )
        )
    return {spec.subset_id: spec for spec in specs}


SPECS = build_specs()
DEFAULT_KEYS = [
    "qc_passed",
    "global_qc_annotated",
    "stage_early_annotated",
    "stage_mid_annotated",
    "stage_late_annotated",
    "cell_type_top_4_mixture",
    "plot_cell_type_top_8_mixture",
    "pair_body_wall_muscle__hypodermis",
    "pair_body_wall_muscle__ciliated_amphid_neuron",
    "pair_body_wall_muscle__ciliated_non_amphid_neuron",
    "pair_hypodermis__ciliated_amphid_neuron",
]
ALL_KEYS = ["qc_passed", "all_cells", *[key for key in DEFAULT_KEYS if key != "qc_passed"]]


def require_scipy() -> None:
    if sp is None or svds is None:
        raise ImportError(
            "scipy is required to read sparse H5AD X and run truncated SVD. "
            "Install scipy or prepare features in an environment where scipy is available."
        )


def decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.asarray(
            [value.decode("utf-8", errors="replace") for value in values],
            dtype=object,
        )
    if values.dtype.kind == "O":
        return np.asarray([decode_scalar(value) for value in values], dtype=object)
    return values


def attr_as_str(value: Any) -> str | None:
    if value is None:
        return None
    value = decode_scalar(value)
    return str(value)


def axis_length(group: h5py.Group | None) -> int | None:
    if group is None:
        return None
    index_name = attr_as_str(group.attrs.get("_index"))
    candidates = []
    if index_name:
        candidates.append(index_name)
    candidates.extend(["_index", "index"])
    for name in candidates:
        if name in group and isinstance(group[name], h5py.Dataset):
            obj = group[name]
            if obj.ndim == 1:
                return int(obj.shape[0])
    for name, obj in group.items():
        if name == "__categories":
            continue
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
            return int(obj.shape[0])
        if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj.keys()):
            return int(obj["codes"].shape[0])
    return None


def matrix_shape(handle: h5py.File) -> tuple[int, int]:
    if "X" not in handle:
        raise ValueError("H5AD file has no X matrix")
    x_obj = handle["X"]
    if isinstance(x_obj, h5py.Dataset):
        if len(x_obj.shape) != 2:
            raise ValueError(f"Expected X to be 2D, found shape {x_obj.shape}")
        return int(x_obj.shape[0]), int(x_obj.shape[1])
    if isinstance(x_obj, h5py.Group):
        shape_attr = x_obj.attrs.get("shape")
        if shape_attr is not None:
            shape = tuple(int(value) for value in shape_attr)
            if len(shape) == 2:
                return shape
        if "shape" in x_obj:
            shape = tuple(int(value) for value in np.asarray(x_obj["shape"][()]).tolist())
            if len(shape) == 2:
                return shape
    obs_len = axis_length(handle.get("obs"))
    var_len = axis_length(handle.get("var"))
    if obs_len is not None and var_len is not None:
        return obs_len, var_len
    raise ValueError("Could not determine X matrix shape from H5AD")


def read_categorical_group(group: h5py.Group) -> pd.Series:
    categories = decode_array(group["categories"][()])
    codes = np.asarray(group["codes"][()])
    values: list[Any] = []
    for code in codes:
        code_int = int(code)
        values.append("" if code_int < 0 else decode_scalar(categories[code_int]))
    return pd.Series(values)


def read_axis_column(
    axis_group: h5py.Group,
    name: str,
    obj: h5py.Dataset | h5py.Group,
    n_rows: int,
) -> tuple[pd.Series | None, str | None]:
    try:
        if isinstance(obj, h5py.Group):
            if {"categories", "codes"}.issubset(obj.keys()):
                series = read_categorical_group(obj)
            else:
                return None, "unsupported H5AD group without categories/codes"
        elif isinstance(obj, h5py.Dataset):
            if obj.ndim != 1:
                return None, f"unsupported non-1D dataset with shape {obj.shape}"
            values = obj[()]
            if (
                obj.dtype.kind in {"i", "u"}
                and "__categories" in axis_group
                and name in axis_group["__categories"]
            ):
                categories = decode_array(axis_group["__categories"][name][()])
                decoded: list[Any] = []
                for code in np.asarray(values).tolist():
                    code_int = int(code)
                    decoded.append("" if code_int < 0 else decode_scalar(categories[code_int]))
                series = pd.Series(decoded)
            else:
                series = pd.Series(decode_array(np.asarray(values)))
        else:
            return None, f"unsupported object type {type(obj).__name__}"
    except Exception as exc:
        return None, str(exc)

    if len(series) != n_rows:
        return None, f"row count mismatch: expected {n_rows}, got {len(series)}"
    return series, None


def read_axis_dataframe(
    handle: h5py.File,
    axis_name: str,
    n_rows: int,
) -> tuple[pd.DataFrame, list[dict[str, str]], str | None]:
    if axis_name not in handle or not isinstance(handle[axis_name], h5py.Group):
        raise ValueError(f"H5AD file has no {axis_name} metadata group")
    axis_group = handle[axis_name]
    skipped: list[dict[str, str]] = []
    columns: dict[str, pd.Series] = {}

    index_source = attr_as_str(axis_group.attrs.get("_index"))
    original_index_column = None
    if index_source and index_source in axis_group:
        series, reason = read_axis_column(axis_group, index_source, axis_group[index_source], n_rows)
        if series is not None:
            columns["original_obs_index"] = series.astype(str)
            original_index_column = "original_obs_index"
        else:
            skipped.append({"column": index_source, "reason": reason or "could not read index"})

    for column_name in axis_group.keys():
        if column_name == "__categories" or column_name == index_source:
            continue
        output_name = str(column_name)
        if output_name in columns:
            output_name = f"raw_{output_name}"
        series, reason = read_axis_column(axis_group, column_name, axis_group[column_name], n_rows)
        if series is None:
            skipped.append({"column": column_name, "reason": reason or "could not read column"})
            continue
        columns[output_name] = series

    frame = pd.DataFrame(columns)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_object_dtype(series):
            frame[column] = series.map(lambda value: "" if pd.isna(value) else str(decode_scalar(value)))
    return frame, skipped, original_index_column


def detect_cell_metadata_axis(handle: h5py.File) -> tuple[str, str, tuple[int, int]]:
    x_shape = matrix_shape(handle)
    obs = handle.get("obs")
    var = handle.get("var")
    obs_len = axis_length(obs if isinstance(obs, h5py.Group) else None)
    var_len = axis_length(var if isinstance(var, h5py.Group) else None)
    obs_score = len(CELL_METADATA_COLUMNS & set(obs.keys())) if isinstance(obs, h5py.Group) else 0
    var_score = len(CELL_METADATA_COLUMNS & set(var.keys())) if isinstance(var, h5py.Group) else 0

    if obs_score > 0 and obs_len == x_shape[0]:
        return "obs", "rows", x_shape
    if var_score > 0 and var_len == x_shape[1]:
        return "var", "columns", x_shape
    if obs_len == x_shape[0]:
        return "obs", "rows", x_shape
    if var_len == x_shape[1]:
        return "var", "columns", x_shape
    raise ValueError(
        "Could not identify the cell metadata axis. "
        f"X shape={x_shape}, obs_len={obs_len}, var_len={var_len}, "
        f"obs_cell_columns={obs_score}, var_cell_columns={var_score}"
    )


def prepare_cell_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "sample_id" in frame.columns:
        frame = frame.rename(columns={"sample_id": "raw_sample_id"})
    if "original_cell_index" in frame.columns:
        frame = frame.rename(columns={"original_cell_index": "raw_original_cell_index"})
    frame.insert(0, "original_cell_index", np.arange(frame.shape[0], dtype=np.int64))
    return frame


def load_context(raw_path: Path) -> dict[str, Any]:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw H5AD: {raw_path}. Run download.py or copy/link packer2019.h5ad "
            "into datasets/raw/c_elegans_embryogenesis/."
        )
    with h5py.File(raw_path, "r") as handle:
        metadata_axis, cell_axis, x_shape = detect_cell_metadata_axis(handle)
        n_cells = x_shape[0] if cell_axis == "rows" else x_shape[1]
        cell_metadata, skipped_obs_columns, original_index_column = read_axis_dataframe(
            handle,
            metadata_axis,
            n_cells,
        )
        cell_metadata = prepare_cell_metadata(cell_metadata)
        gene_axis = "var" if metadata_axis == "obs" else "obs"
        n_genes = x_shape[1] if cell_axis == "rows" else x_shape[0]
        gene_columns = sorted(str(key) for key in handle.get(gene_axis, {}).keys())

    return {
        "raw_path": raw_path,
        "cell_metadata_axis": metadata_axis,
        "cell_axis": cell_axis,
        "gene_metadata_axis": gene_axis,
        "x_shape": [int(x_shape[0]), int(x_shape[1])],
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "obs": cell_metadata,
        "skipped_obs_columns": skipped_obs_columns,
        "original_index_column": original_index_column,
        "gene_metadata_columns": gene_columns,
    }


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "t", "yes", "y"})


def is_informative_label(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).fillna("").str.strip().str.lower()
    return ~lowered.isin(MISSING_CATEGORY_VALUES)


def informative_labels(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    labels = frame.loc[is_informative_label(frame[column]), column].astype(str)
    counts = labels.value_counts(dropna=False)
    return [str(label) for label in counts.index.tolist()]


def selected_labels_for_spec(spec: DatasetSpec, frame: pd.DataFrame) -> list[str]:
    if spec.selected_labels is not None:
        return list(spec.selected_labels)
    if spec.label_field is None or spec.label_field not in frame.columns:
        return []
    mask = pd.Series(np.ones(frame.shape[0], dtype=bool), index=frame.index)
    if spec.require_passed_qc and "passed_qc" in frame.columns:
        mask &= as_bool_series(frame["passed_qc"])
    mask &= is_informative_label(frame[spec.label_field])
    return informative_labels(frame.loc[mask], spec.label_field)


def apply_spec(frame: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    mask = pd.Series(np.ones(frame.shape[0], dtype=bool), index=frame.index)
    if spec.require_passed_qc:
        if "passed_qc" not in frame.columns:
            raise KeyError(f"{spec.subset_id} requires obs column passed_qc")
        mask &= as_bool_series(frame["passed_qc"])
    if spec.label_field is not None:
        if spec.label_field not in frame.columns:
            raise KeyError(f"{spec.subset_id} requires obs column {spec.label_field}")
        if spec.require_informative_label:
            mask &= is_informative_label(frame[spec.label_field])
        if spec.selected_labels is not None:
            mask &= frame[spec.label_field].astype(str).isin([str(value) for value in spec.selected_labels])
    if spec.embryo_time_min is not None or spec.embryo_time_max is not None:
        if "embryo_time" not in frame.columns:
            raise KeyError(f"{spec.subset_id} requires obs column embryo_time")
        embryo_time = pd.to_numeric(frame["embryo_time"], errors="coerce")
        if spec.embryo_time_min is not None:
            mask &= embryo_time >= float(spec.embryo_time_min)
        if spec.embryo_time_max is not None:
            mask &= embryo_time < float(spec.embryo_time_max)
    return mask


def sparse_encoding(x_group: h5py.Group, shape: tuple[int, int]) -> str:
    encoding = attr_as_str(x_group.attrs.get("encoding-type")) or attr_as_str(x_group.attrs.get("h5sparse_format"))
    if encoding:
        encoding = encoding.lower()
        if "csc" in encoding:
            return "csc"
        if "csr" in encoding:
            return "csr"
    if "indptr" in x_group:
        indptr_len = int(x_group["indptr"].shape[0])
        if indptr_len == shape[0] + 1:
            return "csr"
        if indptr_len == shape[1] + 1:
            return "csc"
    return "csr"


def read_csr_rows(x_group: h5py.Group, row_indices: np.ndarray, shape: tuple[int, int]) -> Any:
    require_scipy()
    row_indices = np.asarray(row_indices, dtype=np.int64)
    indptr = np.asarray(x_group["indptr"], dtype=np.int64)
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    out_indptr = np.zeros(row_indices.shape[0] + 1, dtype=np.int64)
    for offset, row in enumerate(row_indices.tolist(), start=1):
        start = int(indptr[row])
        stop = int(indptr[row + 1])
        if stop > start:
            data_parts.append(np.asarray(x_group["data"][start:stop], dtype=np.float32))
            index_parts.append(np.asarray(x_group["indices"][start:stop], dtype=np.int32))
        out_indptr[offset] = out_indptr[offset - 1] + (stop - start)

    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32)
    indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
    return sp.csr_matrix((data, indices, out_indptr), shape=(row_indices.shape[0], shape[1]), dtype=np.float32)


def load_sparse_matrix(x_group: h5py.Group, shape: tuple[int, int]) -> Any:
    require_scipy()
    data = np.asarray(x_group["data"], dtype=np.float32)
    indices = np.asarray(x_group["indices"], dtype=np.int32)
    indptr = np.asarray(x_group["indptr"], dtype=np.int64)
    encoding = sparse_encoding(x_group, shape)
    if encoding == "csc":
        return sp.csc_matrix((data, indices, indptr), shape=shape, dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)


def read_expression_subset(raw_path: Path, source_indices: np.ndarray, cell_axis: str) -> Any:
    require_scipy()
    with h5py.File(raw_path, "r") as handle:
        shape = matrix_shape(handle)
        x_obj = handle["X"]
        if isinstance(x_obj, h5py.Dataset):
            if cell_axis == "rows":
                dense = np.asarray(x_obj[source_indices, :], dtype=np.float32)
            else:
                dense = np.asarray(x_obj[:, source_indices], dtype=np.float32).T
            return sp.csr_matrix(dense)
        if not isinstance(x_obj, h5py.Group):
            raise ValueError(f"Unsupported H5AD X object type: {type(x_obj).__name__}")
        if not {"data", "indices", "indptr"}.issubset(x_obj.keys()):
            raise ValueError("Sparse H5AD X group must contain data, indices, and indptr")
        encoding = sparse_encoding(x_obj, shape)
        if cell_axis == "rows" and encoding == "csr":
            return read_csr_rows(x_obj, source_indices, shape)
        matrix = load_sparse_matrix(x_obj, shape)
        if cell_axis == "rows":
            return matrix[source_indices, :].tocsr().astype(np.float32)
        return matrix[:, source_indices].T.tocsr().astype(np.float32)


def normalize_log1p(matrix: Any, size_factor: np.ndarray | None) -> Any:
    require_scipy()
    processed = matrix.astype(np.float32, copy=True).tocsr()
    if size_factor is not None:
        factors = np.asarray(size_factor, dtype=np.float32)
        factors[~np.isfinite(factors)] = 1.0
        factors[factors <= 0] = 1.0
        positive = factors[factors > 0]
        median = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(median) or median <= 0:
            median = 1.0
        processed = processed.multiply((median / factors).astype(np.float32)[:, None]).tocsr()
    else:
        row_sums = np.asarray(processed.sum(axis=1)).ravel().astype(np.float32)
        row_sums[~np.isfinite(row_sums)] = 1.0
        row_sums[row_sums <= 0] = 1.0
        positive = row_sums[row_sums > 0]
        median = float(np.median(positive)) if positive.size else 1.0
        processed = processed.multiply((median / row_sums).astype(np.float32)[:, None]).tocsr()
    if processed.nnz:
        processed.data = np.log1p(processed.data).astype(np.float32, copy=False)
    return processed


def truncated_svd_features(matrix: Any, n_components: int, random_state: int) -> tuple[np.ndarray, dict[str, Any]]:
    require_scipy()
    requested = int(n_components)
    if requested <= 0:
        raise ValueError("--pca-components must be positive")
    max_components = min(requested, int(matrix.shape[0]) - 1, int(matrix.shape[1]) - 1)
    if max_components < 1:
        raise ValueError(f"Cannot run SVD on matrix with shape {matrix.shape}")

    try:
        u, singular_values, _vt = svds(matrix, k=max_components, random_state=int(random_state))
    except TypeError:
        u, singular_values, _vt = svds(matrix, k=max_components)
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    u = u[:, order]
    features = (u * singular_values).astype(np.float32, copy=False)
    return features, {
        "representation": "truncated_svd",
        "requested_components": requested,
        "n_output_features": int(features.shape[1]),
        "n_input_features": int(matrix.shape[1]),
        "singular_values": [float(value) for value in singular_values.astype(np.float64)],
        "size_factor_normalization": True,
        "log1p": True,
    }


def schema_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    return "string"


def role_for_column(column: str, series: pd.Series) -> str:
    lower = column.lower()
    if column == "sample_id":
        return "identifier"
    if column == "original_cell_index":
        return "original_index"
    if column in {"cell_type", "plot_cell_type", "lineage", "cell_subtype"}:
        return "biological_label"
    if "time" in lower:
        return "developmental_time"
    if "batch" in lower or column in {"study", "sample", "sample_description"}:
        return "batch_or_source"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous_metadata"
    return "metadata"


def build_obs_schema(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        examples = [str(value) for value in non_null.astype(str).unique()[:3]]
        schema[column] = {
            "dtype": schema_dtype(series),
            "role": role_for_column(column, series),
            "description": f"Observation column {column} from the Packer 2019 H5AD cell metadata.",
            "n_unique": int(series.nunique(dropna=True)),
            "example_values": examples,
        }
    return schema


def metadata_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], str | None]:
    schema = build_obs_schema(frame)
    columns = set(frame.columns)
    label_columns = [column for column in LABEL_PRIORITY if column in columns]
    default_color_by = next((column for column in DEFAULT_COLOR_PRIORITY if column in columns), None)

    colorable_columns: list[str] = []
    filterable_columns: list[str] = []
    searchable_columns: list[str] = []
    for column, config in schema.items():
        if column == "sample_id":
            continue
        n_unique = int(config.get("n_unique", 0))
        dtype = str(config.get("dtype"))
        if column in {"original_cell_index", "original_obs_index", "barcode"}:
            searchable_columns.append(column)
        if dtype in {"string", "bool"} and 1 < n_unique <= 2000:
            colorable_columns.append(column)
            filterable_columns.append(column)

    for column in label_columns:
        if column not in colorable_columns:
            colorable_columns.insert(0, column)
        if column not in filterable_columns:
            filterable_columns.insert(0, column)
    if default_color_by is not None and default_color_by not in colorable_columns:
        colorable_columns.insert(0, default_color_by)
    return label_columns, colorable_columns, searchable_columns, filterable_columns, default_color_by


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}. Use --force to rebuild it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def embryo_time_stats(values: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return float(numeric.min()), float(numeric.max())
    return None, None


def write_raw_metadata_summary(context: dict[str, Any]) -> dict[str, Any]:
    obs = context["obs"]
    summary: dict[str, Any] = {
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "raw_path": repo_relative(context["raw_path"]),
        "cell_metadata_axis": context["cell_metadata_axis"],
        "gene_metadata_axis": context["gene_metadata_axis"],
        "cell_axis_in_X": context["cell_axis"],
        "h5ad_x_shape": context["x_shape"],
        "n_cells": int(context["n_cells"]),
        "n_genes": int(context["n_genes"]),
        "obs_columns": [str(column) for column in obs.columns],
        "gene_metadata_columns": context["gene_metadata_columns"],
        "skipped_obs_columns": context["skipped_obs_columns"],
        "generated_at": utc_now_iso(),
    }
    if "passed_qc" in obs.columns:
        summary["n_passed_qc"] = int(as_bool_series(obs["passed_qc"]).sum())
    if "embryo_time" in obs.columns:
        embryo_time = pd.to_numeric(obs["embryo_time"], errors="coerce")
        summary["embryo_time_min"] = float(embryo_time.min()) if embryo_time.notna().any() else None
        summary["embryo_time_max"] = float(embryo_time.max()) if embryo_time.notna().any() else None
        summary["embryo_time_quantiles"] = {
            str(q): float(np.nanquantile(embryo_time.to_numpy(dtype=np.float64), q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            if embryo_time.notna().any()
        }
    for column in ["cell_type", "plot_cell_type", "lineage", "embryo_time_bin", "time_point"]:
        if column in obs.columns:
            summary[f"{column}_top"] = (
                obs[column].astype(str).value_counts(dropna=False).head(20).to_dict()
            )
    RAW_METADATA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(RAW_METADATA_SUMMARY_PATH, summary)
    return summary


def metadata_payload(
    *,
    spec: DatasetSpec,
    raw_path: Path,
    obs_frame: pd.DataFrame,
    feature_shape: list[int],
    pca_components: int,
    selected_labels: list[str],
    summary_file: str,
    skipped_obs_columns: list[dict[str, str]],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    label_columns, colorable_columns, searchable_columns, filterable_columns, default_color_by = (
        metadata_columns(obs_frame)
    )
    n_cells = int(feature_shape[0])
    return {
        "dataset_id": spec.dataset_id,
        "display_name": spec.display_name,
        "family": FAMILY,
        "kind": "single_cell",
        "source_type": "h5ad",
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "raw_path": repo_relative(raw_path),
        "raw_role": RAW_ROLE,
        "primary_array": "features.npy",
        "feature_source": FEATURE_SOURCE,
        "feature_shape": feature_shape,
        "feature_dtype": "float32",
        "pca_components": int(pca_components),
        "obs_file": "obs.csv.gz",
        "pseudotime_file": "pseudotime.npy",
        "pseudotime_source": "obs/embryo_time",
        "source_indices_file": "source_indices.npy",
        "summary_file": summary_file,
        "identifier_column": "sample_id",
        "original_index_column": "original_cell_index",
        "default_color_by": default_color_by,
        "label_columns": label_columns,
        "colorable_columns": colorable_columns,
        "searchable_columns": searchable_columns,
        "filterable_columns": filterable_columns,
        "obs_schema": build_obs_schema(obs_frame),
        "subset_id": spec.subset_id,
        "subset_kind": spec.subset_kind,
        "subset_filter": spec.filter_description,
        "selected_labels": selected_labels,
        "label_field": spec.label_field,
        "n_cells": n_cells,
        "generated_by": repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": SCRIPT_VERSION,
        "preprocessing": preprocessing,
        "skipped_obs_columns": skipped_obs_columns,
    }


def prepare_one(context: dict[str, Any], spec: DatasetSpec, args: argparse.Namespace) -> None:
    require_scipy()
    raw_path = Path(context["raw_path"])
    frame = context["obs"]
    mask = apply_spec(frame, spec)
    subset_obs = frame.loc[mask].copy().reset_index(drop=True)
    if subset_obs.empty:
        raise RuntimeError(f"Subset {spec.subset_id} is empty after filtering")
    if "embryo_time" not in subset_obs.columns:
        raise KeyError("embryo_time is required to write pseudotime.npy")

    selected_labels = selected_labels_for_spec(spec, frame)
    source_indices = subset_obs["original_cell_index"].to_numpy(dtype=np.int64)
    obs_out = subset_obs.copy()
    obs_out.insert(0, "sample_id", np.arange(obs_out.shape[0], dtype=np.int64))

    output_dir = PROCESSED_ROOT / spec.dataset_id
    prepare_output_dir(output_dir, args.force)

    try:
        expression = read_expression_subset(raw_path, source_indices, context["cell_axis"])
        size_factor = (
            pd.to_numeric(subset_obs["size_factor"], errors="coerce").to_numpy(dtype=np.float32)
            if "size_factor" in subset_obs.columns
            else None
        )
        normalized = normalize_log1p(expression, size_factor)
        features, preprocessing = truncated_svd_features(
            normalized,
            n_components=args.pca_components,
            random_state=args.random_state,
        )
        feature_shape = [int(value) for value in features.shape]
        pseudotime = pd.to_numeric(subset_obs["embryo_time"], errors="coerce").to_numpy(dtype=np.float32)
        embryo_min, embryo_max = embryo_time_stats(subset_obs["embryo_time"])

        features_path = output_dir / "features.npy"
        obs_path = output_dir / "obs.csv.gz"
        pseudotime_path = output_dir / "pseudotime.npy"
        source_indices_path = output_dir / "source_indices.npy"
        summary_path = output_dir / spec.summary_file
        metadata_path = output_dir / "metadata.json"

        np.save(features_path, features.astype(np.float32, copy=False))
        obs_out.to_csv(obs_path, index=False, compression="gzip", quoting=csv.QUOTE_MINIMAL)
        np.save(pseudotime_path, pseudotime)
        np.save(source_indices_path, source_indices)

        summary = {
            "subset_id": spec.subset_id,
            "dataset_id": spec.dataset_id,
            "subset_kind": spec.subset_kind,
            "filter_description": spec.filter_description,
            "label_field": spec.label_field,
            "selected_labels": selected_labels,
            "n_cells": int(obs_out.shape[0]),
            "n_genes": int(context["n_genes"]),
            "feature_shape": feature_shape,
            "embryo_time_min": embryo_min,
            "embryo_time_max": embryo_max,
            "source_indices_file": "source_indices.npy",
            "pseudotime_file": "pseudotime.npy",
            "obs_file": "obs.csv.gz",
            "feature_file": "features.npy",
            "raw_path": repo_relative(raw_path),
            "cell_metadata_axis": context["cell_metadata_axis"],
            "cell_axis_in_X": context["cell_axis"],
            "preprocessing": preprocessing,
            "generated_at": utc_now_iso(),
        }
        write_json(summary_path, summary)

        metadata = metadata_payload(
            spec=spec,
            raw_path=raw_path,
            obs_frame=obs_out,
            feature_shape=feature_shape,
            pca_components=args.pca_components,
            selected_labels=selected_labels,
            summary_file=spec.summary_file,
            skipped_obs_columns=context["skipped_obs_columns"],
            preprocessing=preprocessing,
        )
        write_metadata(metadata_path, metadata)
        write_checksums(
            output_dir / "checksums.txt",
            [features_path, obs_path, pseudotime_path, source_indices_path, summary_path, metadata_path],
            base_dir=output_dir,
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print(f"Prepared {repo_relative(output_dir)}")


def resolve_dataset_key(query: str) -> str:
    normalized = query.strip()
    if normalized in SPECS:
        return normalized
    for key, spec in SPECS.items():
        if normalized == spec.dataset_id:
            return key
    prefix = f"{FAMILY}_"
    if normalized.startswith(prefix):
        candidate = normalized[len(prefix) :]
        if candidate in SPECS:
            return candidate
    raise KeyError(f"Unknown dataset/subset {query!r}. Run --list-subsets.")


def print_subset_listing() -> None:
    for key in ALL_KEYS:
        spec = SPECS[key]
        labels = ",".join(spec.selected_labels or ())
        print(
            "\t".join(
                [
                    key,
                    spec.dataset_id,
                    spec.subset_kind,
                    spec.label_field or "",
                    labels or "dynamic_or_none",
                ]
            )
        )


def selected_keys(args: argparse.Namespace) -> list[str]:
    if args.all:
        return ALL_KEYS
    if args.all_default_subsets:
        return DEFAULT_KEYS
    if args.dataset:
        return [resolve_dataset_key(args.dataset)]
    return []


def main() -> int:
    args = parse_args()
    try:
        if args.list_subsets:
            print_subset_listing()
            return 0

        keys = selected_keys(args)
        if not args.summary and not keys:
            print("FAIL: provide --summary, --dataset, --all-default-subsets, --all, or --list-subsets", file=sys.stderr)
            return 2

        context = load_context(args.raw_path)
        if args.summary:
            summary = write_raw_metadata_summary(context)
            print(
                "Wrote "
                f"{repo_relative(RAW_METADATA_SUMMARY_PATH)} "
                f"(n_cells={summary.get('n_cells')}, n_genes={summary.get('n_genes')})"
            )

        for key in keys:
            prepare_one(context, SPECS[key], args)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
