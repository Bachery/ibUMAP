from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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
COLLECTION_ID = "0cca8620-8dee-45d0-aef5-23f032a5cf09"
COLLECTION_API = (
    "https://api.cellxgene.cziscience.com/curation/v1/collections/"
    f"{COLLECTION_ID}"
)
RAW_ROOT = REPO_ROOT / "datasets/raw/whole_mouse_brain_merfish"
H5AD_ROOT = RAW_ROOT / "h5ad"
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"

FAMILY = "whole_mouse_brain_merfish"
DEFAULT_SVD_COMPONENTS = 50
DEFAULT_BATCH_ROWS = 100_000
LABEL_PRIORITY = ["cell_type", "subclass_transfer", "cluster_id_transfer", "major_brain_region"]
DEFAULT_COLOR_PRIORITY = ["cell_type", "subclass_transfer", "major_brain_region"]
ANATOMY_COLUMNS = {"ccf_region_name", "major_brain_region", "brain_section_label", "tissue"}

RAW_FILE_NAMES = (
    "WB_MERFISH_animal1_coronal.h5ad",
    "WB_MERFISH_animal2_coronal.h5ad",
    "WB_MERFISH_animal3_sagittal.h5ad",
    "WB_MERFISH_animal4_sagittal.h5ad",
)


@dataclass(frozen=True)
class ObsColumn:
    source_name: str
    output_name: str
    kind: str
    dtype: str
    categories: np.ndarray | None = None

    @property
    def n_unique(self) -> int | None:
        return None if self.categories is None else int(self.categories.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare whole mouse brain MERFISH H5AD files into compact, "
            "dimension-reduction-ready datasets."
        )
    )
    parser.add_argument("--all", action="store_true", help="Prepare all four downloaded H5AD files.")
    parser.add_argument("--list", action="store_true", help="List available raw H5AD files.")
    parser.add_argument(
        "--dataset",
        help="Prepare one dataset by raw stem, H5AD file name, or processed dataset_id.",
    )
    parser.add_argument("--force", action="store_true", help="Delete and rebuild existing output.")
    parser.add_argument(
        "--svd-components",
        type=int,
        default=DEFAULT_SVD_COMPONENTS,
        help=f"Number of truncated-SVD components (default: {DEFAULT_SVD_COMPONENTS}).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sparse SVD.")
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=f"Rows per output-writing batch (default: {DEFAULT_BATCH_ROWS}).",
    )
    return parser.parse_args()


def require_scipy() -> None:
    if sp is None or svds is None:
        raise RuntimeError("scipy is required to read sparse H5AD X and run truncated SVD.")


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def decode_text_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(decode_scalar(value)) for value in values], dtype=object)


def dataset_id_for(path: Path) -> str:
    suffix = path.stem.removeprefix("WB_MERFISH_").lower()
    return f"{FAMILY}_{suffix}"


def display_name_for(path: Path) -> str:
    suffix = path.stem.removeprefix("WB_MERFISH_").replace("_", " ")
    return f"Whole mouse brain MERFISH {suffix}"


def list_h5ad_files() -> list[Path]:
    return [H5AD_ROOT / name for name in RAW_FILE_NAMES if (H5AD_ROOT / name).exists()]


def select_h5ad_files(args: argparse.Namespace) -> list[Path]:
    files = list_h5ad_files()
    if args.list:
        return files
    if args.all:
        return files
    if args.dataset:
        query = args.dataset
        query_stem = Path(query).stem
        selected = [
            path
            for path in files
            if path.name == query or path.stem == query_stem or dataset_id_for(path) == query
        ]
        if not selected:
            raise FileNotFoundError(
                f"No raw H5AD matched {query!r}. Run download.py first or use --list."
            )
        return selected
    raise ValueError("Provide --all, --list, or --dataset")


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}. Use --force to rebuild it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def matrix_shape(x_group: h5py.Group) -> tuple[int, int]:
    raw_shape = x_group.attrs.get("shape")
    if raw_shape is None or len(raw_shape) != 2:
        raise ValueError("Sparse H5AD X is missing a two-dimensional shape attribute")
    return int(raw_shape[0]), int(raw_shape[1])


def load_csr_expression(h5ad_path: Path) -> tuple[Any, tuple[int, int]]:
    require_scipy()
    with h5py.File(h5ad_path, "r") as handle:
        if "X" not in handle or not isinstance(handle["X"], h5py.Group):
            raise ValueError("Expected H5AD X to be a sparse group")
        x_group = handle["X"]
        encoding = str(decode_scalar(x_group.attrs.get("encoding-type", ""))).lower()
        if "csr" not in encoding:
            raise ValueError(f"Expected CSR-encoded H5AD X, found {encoding or 'unknown'}")
        if not {"data", "indices", "indptr"}.issubset(x_group.keys()):
            raise ValueError("Sparse H5AD X must contain data, indices, and indptr")

        shape = matrix_shape(x_group)
        data = np.asarray(x_group["data"], dtype=np.float32)
        indices = np.asarray(x_group["indices"], dtype=np.int32)
        indptr = np.asarray(x_group["indptr"], dtype=np.int64)

    if indptr.shape[0] != shape[0] + 1:
        raise ValueError(f"CSR indptr has length {indptr.shape[0]}, expected {shape[0] + 1}")
    if int(indptr[-1]) != data.shape[0] or data.shape[0] != indices.shape[0]:
        raise ValueError("CSR data, indices, and indptr lengths are inconsistent")
    return sp.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32), shape


def sparse_svd_components(
    expression: Any,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    require_scipy()
    if n_components <= 0:
        raise ValueError("--svd-components must be positive")
    max_components = min(expression.shape) - 1
    if n_components > max_components:
        raise ValueError(
            f"--svd-components={n_components} exceeds the sparse SVD limit of {max_components}"
        )

    _left, singular_values, right_vectors = svds(
        expression,
        k=n_components,
        which="LM",
        return_singular_vectors="vh",
        random_state=random_state,
    )
    if right_vectors is None:
        raise RuntimeError("Sparse SVD did not return right singular vectors")
    order = np.argsort(singular_values)[::-1]
    return (
        np.asarray(right_vectors[order], dtype=np.float32),
        np.asarray(singular_values[order], dtype=np.float32),
    )


def write_svd_features(
    expression: Any,
    components: np.ndarray,
    output_path: Path,
    batch_rows: int,
) -> list[int]:
    if batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    n_rows = int(expression.shape[0])
    feature_shape = (n_rows, int(components.shape[0]))
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=feature_shape)
    try:
        for start in range(0, n_rows, batch_rows):
            stop = min(start + batch_rows, n_rows)
            output[start:stop] = np.asarray(
                expression[start:stop] @ components.T,
                dtype=np.float32,
            )
            print(f"  wrote SVD features for rows {start:,}:{stop:,}", flush=True)
    finally:
        output.flush()
        del output
    return [int(value) for value in feature_shape]


def obs_index_source(obs_group: h5py.Group) -> str:
    attr_value = decode_scalar(obs_group.attrs.get("_index"))
    if isinstance(attr_value, str) and attr_value in obs_group:
        return attr_value
    for candidate in ("_index", "index", "feature_id"):
        if candidate in obs_group:
            return candidate
    raise ValueError("H5AD obs has no usable index column")


def obs_source_order(obs_group: h5py.Group, index_source: str) -> list[str]:
    raw_order = obs_group.attrs.get("column-order", [])
    ordered = [str(decode_scalar(value)) for value in raw_order]
    ordered = [name for name in ordered if name in obs_group and name != index_source]
    remaining = sorted(name for name in obs_group.keys() if name not in ordered and name != index_source)
    return [index_source, *ordered, *remaining]


def describe_obs_columns(obs_group: h5py.Group, n_rows: int) -> list[ObsColumn]:
    index_source = obs_index_source(obs_group)
    columns: list[ObsColumn] = []
    for source_name in obs_source_order(obs_group, index_source):
        obj = obs_group[source_name]
        output_name = "original_obs_index" if source_name == index_source else source_name
        if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj.keys()):
            if int(obj["codes"].shape[0]) != n_rows:
                raise ValueError(f"obs/{source_name} row count does not match X")
            columns.append(
                ObsColumn(
                    source_name=source_name,
                    output_name=output_name,
                    kind="categorical",
                    dtype="string",
                    categories=decode_text_array(obj["categories"][:]),
                )
            )
        elif isinstance(obj, h5py.Dataset):
            if obj.ndim != 1 or int(obj.shape[0]) != n_rows:
                raise ValueError(f"obs/{source_name} is not a one-dimensional column with {n_rows} rows")
            dtype = "string" if obj.dtype.kind in {"O", "S", "U"} else str(obj.dtype)
            columns.append(ObsColumn(source_name, output_name, "dataset", dtype))
        else:
            raise ValueError(f"Unsupported obs/{source_name} type: {type(obj).__name__}")
    return columns


def obs_values(column: ObsColumn, obs_group: h5py.Group, start: int, stop: int) -> np.ndarray:
    obj = obs_group[column.source_name]
    if column.kind == "categorical":
        codes = np.asarray(obj["codes"][start:stop], dtype=np.int64)
        values = np.full(codes.shape[0], "", dtype=object)
        valid = (codes >= 0) & (codes < int(column.categories.shape[0]))
        values[valid] = column.categories[codes[valid]]
        return values
    values = obj[start:stop]
    if values.dtype.kind in {"O", "S", "U"}:
        return decode_text_array(values)
    return np.asarray(values)


def role_for_obs_column(column: ObsColumn) -> str:
    name = column.output_name
    if name == "sample_id":
        return "identifier"
    if name == "original_obs_index":
        return "original_identifier"
    if name in {"cell_type", "subclass_transfer", "cluster_id_transfer", "cell_type_ontology_term_id"}:
        return "biological_label"
    if name in ANATOMY_COLUMNS:
        return "anatomical_or_spatial_annotation"
    if name in {"donor_id", "assay", "assay_ontology_term_id"}:
        return "batch_or_source"
    if "confidence" in name:
        return "continuous_metadata"
    if column.dtype in {"float32", "float64", "int32", "int64"}:
        return "continuous_metadata"
    return "metadata"


def build_obs_schema(columns: list[ObsColumn]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {
        "sample_id": {
            "dtype": "int64",
            "role": "identifier",
            "description": "Stable row identifier matching features.npy.",
        }
    }
    for column in columns:
        record: dict[str, Any] = {
            "dtype": column.dtype,
            "role": role_for_obs_column(column),
            "description": f"Observation column {column.source_name} from the source H5AD obs table.",
        }
        if column.n_unique is not None:
            record["n_unique"] = column.n_unique
        schema[column.output_name] = record
    return schema


def obs_metadata_fields(columns: list[ObsColumn]) -> tuple[list[str], list[str], list[str], list[str], str | None]:
    available = {column.output_name for column in columns}
    label_columns = [column for column in LABEL_PRIORITY if column in available]
    default_color_by = next((column for column in DEFAULT_COLOR_PRIORITY if column in available), None)
    colorable_columns = [
        column.output_name
        for column in columns
        if column.kind == "categorical" and 1 < (column.n_unique or 0) <= 1000
    ]
    filterable_columns = list(colorable_columns)
    searchable_columns = [
        column
        for column in ("original_obs_index", "observation_joinid")
        if column in available or column == "original_obs_index"
    ]
    if default_color_by is not None and default_color_by not in colorable_columns:
        colorable_columns.insert(0, default_color_by)
    return label_columns, colorable_columns, searchable_columns, filterable_columns, default_color_by


def write_obs_csv(
    h5ad_path: Path,
    columns: list[ObsColumn],
    output_path: Path,
    n_rows: int,
    batch_rows: int,
) -> None:
    if batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    header = ["sample_id", *[column.output_name for column in columns]]
    with h5py.File(h5ad_path, "r") as handle, gzip.open(
        output_path, "wt", newline="", encoding="utf-8"
    ) as text_handle:
        writer = csv.writer(text_handle)
        writer.writerow(header)
        obs_group = handle["obs"]
        for start in range(0, n_rows, batch_rows):
            stop = min(start + batch_rows, n_rows)
            values = [np.arange(start, stop, dtype=np.int64)]
            values.extend(obs_values(column, obs_group, start, stop) for column in columns)
            writer.writerows(zip(*values))
            print(f"  wrote obs rows {start:,}:{stop:,}", flush=True)


def write_h5_array(
    h5ad_path: Path,
    source_key: str,
    output_path: Path,
    n_rows: int,
    batch_rows: int,
) -> list[int]:
    if batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    with h5py.File(h5ad_path, "r") as handle:
        source = handle[source_key]
        if not isinstance(source, h5py.Dataset) or source.ndim != 2 or int(source.shape[0]) != n_rows:
            raise ValueError(f"{source_key} must be a two-dimensional array with {n_rows} rows")
        shape = tuple(int(value) for value in source.shape)
        output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
        try:
            for start in range(0, n_rows, batch_rows):
                stop = min(start + batch_rows, n_rows)
                output[start:stop] = np.asarray(source[start:stop], dtype=np.float32)
        finally:
            output.flush()
            del output
    return [int(value) for value in shape]


def raw_h5ad_keys_summary(h5ad_path: Path) -> dict[str, Any]:
    with h5py.File(h5ad_path, "r") as handle:
        x_group = handle["X"]
        summary: dict[str, Any] = {
            "root_keys": sorted(str(key) for key in handle.keys()),
            "X": {
                "encoding_type": str(decode_scalar(x_group.attrs.get("encoding-type", ""))),
                "shape": [int(value) for value in matrix_shape(x_group)],
                "data_dtype": str(x_group["data"].dtype),
                "nnz": int(x_group["data"].shape[0]),
            },
            "obs_columns": sorted(str(key) for key in handle["obs"].keys()),
            "obsm": {},
        }
        for key, obj in handle.get("obsm", {}).items():
            if isinstance(obj, h5py.Dataset):
                summary["obsm"][str(key)] = {
                    "shape": [int(value) for value in obj.shape],
                    "dtype": str(obj.dtype),
                }
    return summary


def metadata_payload(
    *,
    h5ad_path: Path,
    feature_shape: list[int],
    singular_values: np.ndarray,
    obs_columns: list[ObsColumn],
    reference_shape: list[int],
    spatial_shape: list[int],
    ccf_shape: list[int],
    raw_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    label_columns, colorable_columns, searchable_columns, filterable_columns, default_color_by = (
        obs_metadata_fields(obs_columns)
    )
    return {
        "dataset_id": dataset_id_for(h5ad_path),
        "display_name": display_name_for(h5ad_path),
        "family": FAMILY,
        "kind": "single_cell",
        "source_type": "cellxgene_collection",
        "collection_id": COLLECTION_ID,
        "collection_api": COLLECTION_API,
        "raw_path": repo_relative(h5ad_path),
        "raw_role": "primary_source",
        "primary_array": "features.npy",
        "feature_source": "X (H5AD CSR expression matrix) transformed by truncated SVD",
        "feature_shape": feature_shape,
        "feature_dtype": "float32",
        "svd_components": int(args.svd_components),
        "svd_singular_values": [float(value) for value in singular_values],
        "obs_file": "obs.csv.gz",
        "reference_embedding_file": "reference_embedding.npy",
        "reference_embedding_source": "obsm/X_umap",
        "reference_embedding_shape": reference_shape,
        "reference_embeddings": {
            "umap": {
                "file": "reference_embedding.npy",
                "shape": reference_shape,
                "dtype": "float32",
                "role": "reference_embedding",
                "source": "obsm/X_umap",
            }
        },
        "extra_arrays": {
            "spatial_coordinates": {
                "file": "spatial_coordinates.npy",
                "shape": spatial_shape,
                "dtype": "float32",
                "role": "spatial_coordinates",
                "source": "obsm/X_spatial_coords",
            },
            "ccf_coordinates": {
                "file": "ccf_coordinates.npy",
                "shape": ccf_shape,
                "dtype": "float32",
                "role": "common_coordinate_framework_coordinates",
                "source": "obsm/X_CCF",
            },
        },
        "metric": "euclidean",
        "identifier_column": "sample_id",
        "original_index_column": "original_obs_index",
        "default_color_by": default_color_by,
        "label_columns": label_columns,
        "colorable_columns": colorable_columns,
        "searchable_columns": searchable_columns,
        "filterable_columns": filterable_columns,
        "obs_schema": build_obs_schema(obs_columns),
        "n_cells": int(feature_shape[0]),
        "generated_by": repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": 1,
        "preprocessing": (
            "Read the already normalized log-expression H5AD X CSR matrix without "
            "densifying it, fitted truncated SVD with scipy.sparse.linalg.svds, and "
            "wrote row-batched float32 SVD projections. Exported obs and stored the "
            "source X_umap, X_spatial_coords, and X_CCF arrays as float32."
        ),
        "raw_h5ad_keys_summary": raw_summary,
    }


def prepare_one(h5ad_path: Path, args: argparse.Namespace) -> None:
    require_scipy()
    output_dir = PROCESSED_ROOT / dataset_id_for(h5ad_path)
    prepare_output_dir(output_dir, args.force)
    try:
        with h5py.File(h5ad_path, "r") as handle:
            if "obs" not in handle or not isinstance(handle["obs"], h5py.Group):
                raise ValueError("H5AD is missing the obs group")
            if "obsm/X_umap" not in handle:
                raise ValueError("H5AD is missing required obsm/X_umap")
            if "obsm/X_spatial_coords" not in handle or "obsm/X_CCF" not in handle:
                raise ValueError("H5AD is missing required spatial coordinate arrays")
            x_shape = matrix_shape(handle["X"])
            obs_columns = describe_obs_columns(handle["obs"], x_shape[0])

        print(f"[{dataset_id_for(h5ad_path)}] loading sparse expression matrix", flush=True)
        expression, x_shape = load_csr_expression(h5ad_path)
        print(
            f"[{dataset_id_for(h5ad_path)}] fitting truncated SVD on "
            f"{x_shape[0]:,} cells x {x_shape[1]:,} genes ({expression.nnz:,} nonzeros)",
            flush=True,
        )
        components, singular_values = sparse_svd_components(
            expression,
            args.svd_components,
            args.random_state,
        )

        features_path = output_dir / "features.npy"
        print(f"[{dataset_id_for(h5ad_path)}] writing SVD features", flush=True)
        feature_shape = write_svd_features(expression, components, features_path, args.batch_rows)
        del expression, components
        gc.collect()

        obs_path = output_dir / "obs.csv.gz"
        print(f"[{dataset_id_for(h5ad_path)}] writing observation metadata", flush=True)
        write_obs_csv(h5ad_path, obs_columns, obs_path, feature_shape[0], args.batch_rows)

        reference_path = output_dir / "reference_embedding.npy"
        spatial_path = output_dir / "spatial_coordinates.npy"
        ccf_path = output_dir / "ccf_coordinates.npy"
        print(f"[{dataset_id_for(h5ad_path)}] writing reference coordinate arrays", flush=True)
        reference_shape = write_h5_array(
            h5ad_path,
            "obsm/X_umap",
            reference_path,
            feature_shape[0],
            args.batch_rows,
        )
        spatial_shape = write_h5_array(
            h5ad_path,
            "obsm/X_spatial_coords",
            spatial_path,
            feature_shape[0],
            args.batch_rows,
        )
        ccf_shape = write_h5_array(
            h5ad_path,
            "obsm/X_CCF",
            ccf_path,
            feature_shape[0],
            args.batch_rows,
        )

        metadata_path = output_dir / "metadata.json"
        metadata = metadata_payload(
            h5ad_path=h5ad_path,
            feature_shape=feature_shape,
            singular_values=singular_values,
            obs_columns=obs_columns,
            reference_shape=reference_shape,
            spatial_shape=spatial_shape,
            ccf_shape=ccf_shape,
            raw_summary=raw_h5ad_keys_summary(h5ad_path),
            args=args,
        )
        write_metadata(metadata_path, metadata)
        write_checksums(
            output_dir / "checksums.txt",
            [features_path, obs_path, reference_path, spatial_path, ccf_path, metadata_path],
            base_dir=output_dir,
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(f"[{dataset_id_for(h5ad_path)}] prepared {repo_relative(output_dir)}")


def main() -> int:
    args = parse_args()
    try:
        selected = select_h5ad_files(args)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.list:
        if not selected:
            print(f"No expected H5AD files found under {H5AD_ROOT}.")
        for path in selected:
            print(f"{path.name}\t{dataset_id_for(path)}")
        return 0
    if not selected:
        print(f"FAIL: no expected H5AD files found under {H5AD_ROOT}", file=sys.stderr)
        return 1

    failures: list[tuple[Path, str]] = []
    for path in selected:
        try:
            prepare_one(path, args)
        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"[{dataset_id_for(path)}] FAIL: {exc}", file=sys.stderr)

    if failures:
        print(
            "Failed: " + "; ".join(f"{path.name} ({message})" for path, message in failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
