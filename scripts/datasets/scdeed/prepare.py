from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
import h5py
import numpy as np
import pandas as pd
import pyreadr


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import write_checksums
from common.metadata import build_obs_schema, obs_metadata_fields, utc_now_iso, write_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "datasets/raw/scdeed"
MANIFEST_PATH = RAW_ROOT / "inspect_manifest.json"
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"
ZENODO_RECORD_ID = "7216361"
SOURCE_URL = "https://zenodo.org/records/7216361"
SOURCE_PUBLICATION = (
    "Statistical method scDEED for detecting dubious 2D single-cell embeddings and "
    "optimizing t-SNE and UMAP hyperparameters"
)
PREFERRED_LABEL_COLUMNS = [
    "cell_type",
    "cell_ontology_class",
    "free_annotation",
    "annotation",
    "label",
    "cluster_ids",
    "res_0_5",
    "subtissue",
    "tissue",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare scDEED datasets from downloaded raw data.")
    parser.add_argument("--all", action="store_true", help="Prepare all recognized candidates.")
    parser.add_argument("--dataset", help="Prepare one recognized candidate, e.g. marrow.")
    parser.add_argument("--list", action="store_true", help="List recognized candidates.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed output.")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing inspect manifest: {MANIFEST_PATH}. Run inspect.py first.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def safe_dataset_id(value: str) -> str:
    import re

    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "dataset"


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to overwrite it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [
                value.decode("utf-8", errors="replace")
                if isinstance(value, (bytes, bytearray))
                else value
                for value in values
            ],
            dtype=object,
        )
    return values


def read_h5ad_obs_column(item: Any) -> np.ndarray:
    if hasattr(item, "keys") and {"categories", "codes"}.issubset(set(item.keys())):
        categories = decode_array(item["categories"][()])
        codes = np.asarray(item["codes"][()])
        values: list[Any] = []
        for code in codes:
            code_int = int(code)
            if code_int < 0:
                values.append("")
            else:
                values.append(categories[code_int])
        return np.asarray(values, dtype=object)
    return decode_array(np.asarray(item[()]))


def read_h5ad_obs(handle: h5py.File, n_obs: int) -> pd.DataFrame:
    if "obs" not in handle:
        raise ValueError("H5AD file has no obs group")
    data: dict[str, Any] = {"sample_id": np.arange(n_obs, dtype=np.int64)}
    for name, item in handle["obs"].items():
        values = read_h5ad_obs_column(item)
        if values.ndim != 1 or int(values.shape[0]) != n_obs:
            continue
        data[str(name)] = values
    return pd.DataFrame(data)


def read_loom_col_attrs(handle: h5py.File, n_obs: int) -> pd.DataFrame:
    if "col_attrs" not in handle:
        raise ValueError("Loom file has no col_attrs group")
    data: dict[str, Any] = {"sample_id": np.arange(n_obs, dtype=np.int64)}
    for name, item in handle["col_attrs"].items():
        values = decode_array(np.asarray(item[()]))
        if values.ndim != 1 or int(values.shape[0]) != n_obs:
            continue
        data[str(name)] = values
    return pd.DataFrame(data)


def dtype_name(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    return "string"


def choose_label_columns(obs: pd.DataFrame) -> list[str]:
    labels = [column for column in PREFERRED_LABEL_COLUMNS if column in obs.columns]
    for column in obs.columns:
        lower = column.lower()
        if column not in labels and ("label" in lower or "cluster" in lower):
            labels.append(column)
    return labels[:8]


def choose_default_color_by(label_columns: list[str]) -> str | None:
    for column in PREFERRED_LABEL_COLUMNS:
        if column in label_columns:
            return column
    return label_columns[0] if label_columns else None


def obs_schema_from_frame(obs: pd.DataFrame, label_columns: list[str]) -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for column in obs.columns:
        role = "identifier" if column == "sample_id" else "metadata"
        if column in label_columns:
            role = "biological_label"
        columns[column] = {
            "dtype": dtype_name(obs[column]),
            "role": role,
            "description": f"Observation column {column} from the downloaded scDEED raw data.",
        }
    return build_obs_schema(columns)


def write_obs(path: Path, obs: pd.DataFrame) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        obs.to_csv(handle, index=False)


def label_name(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def encode_labels(values: pd.Series) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    names = [label_name(value) for value in values]
    unique = sorted(set(names))
    mapping = {name: index for index, name in enumerate(unique)}
    labels = np.asarray([mapping[name] for name in names], dtype=np.int32)
    label_mapping = [
        {"label": int(index), "label_name": name, "original_value": name}
        for index, name in enumerate(unique)
    ]
    return labels, label_mapping, names


def read_r_object(path: Path, object_name: str) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    key: str | None = None if object_name == "null" else object_name
    if key in result:
        value = result[key]
    elif len(result) == 1:
        value = next(iter(result.values()))
    else:
        raise KeyError(f"Object {object_name!r} not found in {path}")
    if not isinstance(value, pd.DataFrame):
        value = pd.DataFrame(value)
    return value


def first_column(frame: pd.DataFrame, preferred: str | None = None) -> pd.Series:
    if preferred and preferred in frame.columns:
        return frame[preferred]
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one-column table, found columns={list(frame.columns)}")
    return frame.iloc[:, 0]


def candidate_by_id(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any] | None:
    for candidate in manifest.get("prepare_candidates", []):
        if safe_dataset_id(str(candidate.get("dataset_id"))) == safe_dataset_id(dataset_id):
            return candidate
    return None


def skipped_sources(
    manifest: dict[str, Any],
    prepared_dataset_id: str,
    used_paths: str | list[str] | set[str],
) -> list[dict[str, Any]]:
    if isinstance(used_paths, str):
        used_path_set = {used_paths}
    else:
        used_path_set = set(used_paths)
    records = []
    for record in manifest.get("files", []):
        roles = record.get("candidate_roles", [])
        if record.get("path") in used_path_set:
            continue
        if "unknown" in roles or record.get("status") in {
            "unsupported_without_pyreadr",
            "unsupported_without_h5py",
            "inspect_failed",
        }:
            records.append(
                {
                    "path": record.get("path"),
                    "file_type": record.get("file_type"),
                    "status": record.get("status", "unrecognized"),
                    "candidate_roles": roles,
                    "reason": record.get("error") or "not paired into a prepared dataset",
                }
            )
    for candidate in manifest.get("prepare_candidates", []):
        if safe_dataset_id(str(candidate.get("dataset_id"))) != safe_dataset_id(prepared_dataset_id):
            records.append(
                {
                    "dataset_id": candidate.get("dataset_id"),
                    "status": "not_selected",
                    "reason": "candidate was not selected for this processed dataset",
                }
            )
    return records


def h5_key(handle: h5py.File, key: str) -> Any:
    item: Any = handle
    for part in key.split("/"):
        item = item[part]
    return item


def prepare_h5ad_candidate(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    force: bool,
) -> tuple[bool, str]:
    dataset_id = safe_dataset_id(str(candidate["dataset_id"]))
    raw_path_value = str(candidate["raw_path"])
    raw_path = REPO_ROOT / raw_path_value
    if not raw_path.exists():
        return False, f"raw source file is missing: {raw_path_value}"

    output_dir = PROCESSED_ROOT / f"scdeed_{dataset_id}"
    try:
        prepare_output_dir(output_dir, force)
        with h5py.File(raw_path, "r") as handle:
            feature_key = str(candidate["feature_key"])
            source_features = h5_key(handle, feature_key)
            features = np.asarray(source_features[()], dtype=np.float32)
            if features.ndim != 2:
                raise ValueError(f"features must be 2D, found shape {features.shape}")

            obs = read_h5ad_obs(handle, int(features.shape[0]))
            label_columns = choose_label_columns(obs)
            default_color_by = choose_default_color_by(label_columns)
            obs_schema = obs_schema_from_frame(obs, label_columns)

            features_out = output_dir / "features.npy"
            obs_out = output_dir / "obs.csv.gz"
            metadata_out = output_dir / "metadata.json"
            np.save(features_out, features)
            write_obs(obs_out, obs)

            reference_records: dict[str, dict[str, Any]] = {}
            reference_files: list[Path] = []
            reference_root = output_dir / "reference_embeddings"
            for name, record in sorted(candidate.get("reference_embeddings", {}).items()):
                key = str(record["key"])
                reference = np.asarray(h5_key(handle, key)[()], dtype=np.float32)
                if reference.ndim != 2 or int(reference.shape[0]) != int(features.shape[0]):
                    continue
                reference_root.mkdir(parents=True, exist_ok=True)
                reference_path = reference_root / f"{safe_dataset_id(name)}.npy"
                np.save(reference_path, reference)
                reference_files.append(reference_path)
                reference_records[safe_dataset_id(name)] = {
                    "file": reference_path.relative_to(output_dir).as_posix(),
                    "source": key,
                    "shape": [int(value) for value in reference.shape],
                    "dtype": "float32",
                    "role": "reference_embedding",
                }

            metadata = {
                "dataset_id": f"scdeed_{dataset_id}",
                "display_name": f"scDEED {dataset_id}",
                "family": "scdeed",
                "kind": "single_cell",
                "source_type": "zenodo_record",
                "zenodo_record_id": ZENODO_RECORD_ID,
                "source_url": SOURCE_URL,
                "source_publication": SOURCE_PUBLICATION,
                "raw_path": repo_relative(raw_path),
                "raw_paths": [repo_relative(raw_path)],
                "raw_role": "primary_source",
                "primary_array": "features.npy",
                "feature_source": feature_key,
                "feature_source_path": repo_relative(raw_path),
                "feature_shape": [int(value) for value in features.shape],
                "feature_dtype": "float32",
                "source_feature_dtype": str(source_features.dtype),
                "obs_file": "obs.csv.gz",
                "identifier_column": "sample_id",
                "default_color_by": default_color_by,
                "label_columns": label_columns,
                "colorable_columns": label_columns,
                "searchable_columns": [
                    column
                    for column in label_columns
                    if column in obs.columns and dtype_name(obs[column]) == "string"
                ],
                "filterable_columns": label_columns,
                "reference_embeddings": reference_records,
                "reference_metrics": {},
                "skipped_or_unrecognized_sources": skipped_sources(
                    manifest,
                    dataset_id,
                    raw_path_value,
                ),
                "metric": "euclidean",
                "generated_by": repo_relative(Path(__file__)),
                "generated_at": utc_now_iso(),
                "version": 1,
                "preprocessing": (
                    "Prepared from downloaded and extracted scDEED Zenodo raw data. "
                    f"Features were read from {feature_key} and converted to float32; "
                    "observation metadata was read from the H5AD obs group."
                ),
            }
            metadata.update(
                obs_metadata_fields(
                    obs_schema=obs_schema,
                    identifier_column="sample_id",
                    default_color_by=default_color_by,
                    label_columns=label_columns,
                    colorable_columns=label_columns,
                    searchable_columns=metadata["searchable_columns"],
                    filterable_columns=label_columns,
                )
            )
            write_metadata(metadata_out, metadata)
            write_checksums(
                output_dir / "checksums.txt",
                [features_out, obs_out, metadata_out, *reference_files],
                base_dir=output_dir,
            )
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, str(exc)
    return True, f"prepared {repo_relative(output_dir)}"


def prepare_loom_candidate(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    force: bool,
) -> tuple[bool, str]:
    dataset_id = safe_dataset_id(str(candidate["dataset_id"]))
    raw_path_value = str(candidate["raw_path"])
    raw_path = REPO_ROOT / raw_path_value
    if not raw_path.exists():
        return False, f"raw source file is missing: {raw_path_value}"

    output_dir = PROCESSED_ROOT / f"scdeed_{dataset_id}"
    try:
        prepare_output_dir(output_dir, force)
        with h5py.File(raw_path, "r") as handle:
            matrix = handle["matrix"]
            features = np.asarray(matrix[()], dtype=np.float32).T
            if features.ndim != 2:
                raise ValueError(f"features must be 2D, found shape {features.shape}")

            obs = read_loom_col_attrs(handle, int(features.shape[0]))
            label_columns = choose_label_columns(obs)
            default_color_by = choose_default_color_by(label_columns)
            obs_schema = obs_schema_from_frame(obs, label_columns)

            features_out = output_dir / "features.npy"
            obs_out = output_dir / "obs.csv.gz"
            metadata_out = output_dir / "metadata.json"
            np.save(features_out, features)
            write_obs(obs_out, obs)

            reference_records: dict[str, dict[str, Any]] = {}
            reference_files: list[Path] = []
            reference_root = output_dir / "reference_embeddings"
            for name, record in sorted(candidate.get("reference_embeddings", {}).items()):
                columns = [str(column) for column in record.get("columns", [])]
                arrays = []
                for column in columns:
                    _, attr_name = column.split("/", 1)
                    arrays.append(np.asarray(handle["col_attrs"][attr_name][()], dtype=np.float32))
                reference = np.column_stack(arrays).astype(np.float32, copy=False)
                if reference.ndim != 2 or int(reference.shape[0]) != int(features.shape[0]):
                    continue
                reference_root.mkdir(parents=True, exist_ok=True)
                reference_path = reference_root / f"{safe_dataset_id(name)}.npy"
                np.save(reference_path, reference)
                reference_files.append(reference_path)
                reference_records[safe_dataset_id(name)] = {
                    "file": reference_path.relative_to(output_dir).as_posix(),
                    "source": ",".join(columns),
                    "shape": [int(value) for value in reference.shape],
                    "dtype": "float32",
                    "role": "reference_embedding",
                }

            metadata = {
                "dataset_id": f"scdeed_{dataset_id}",
                "display_name": f"scDEED {dataset_id}",
                "family": "scdeed",
                "kind": "single_cell",
                "source_type": "zenodo_record",
                "zenodo_record_id": ZENODO_RECORD_ID,
                "source_url": SOURCE_URL,
                "source_publication": SOURCE_PUBLICATION,
                "raw_path": repo_relative(raw_path),
                "raw_paths": [repo_relative(raw_path)],
                "raw_role": "primary_source",
                "primary_array": "features.npy",
                "feature_source": f"{raw_path_value}:matrix:transposed_cells_by_genes",
                "feature_source_path": repo_relative(raw_path),
                "feature_shape": [int(value) for value in features.shape],
                "feature_dtype": "float32",
                "source_feature_dtype": str(matrix.dtype),
                "obs_file": "obs.csv.gz",
                "identifier_column": "sample_id",
                "default_color_by": default_color_by,
                "label_columns": label_columns,
                "colorable_columns": label_columns,
                "searchable_columns": [
                    column
                    for column in label_columns
                    if column in obs.columns and dtype_name(obs[column]) == "string"
                ],
                "filterable_columns": label_columns,
                "reference_embeddings": reference_records,
                "reference_metrics": {},
                "skipped_or_unrecognized_sources": skipped_sources(
                    manifest,
                    dataset_id,
                    raw_path_value,
                ),
                "metric": "euclidean",
                "generated_by": repo_relative(Path(__file__)),
                "generated_at": utc_now_iso(),
                "version": 1,
                "preprocessing": (
                    "Prepared from downloaded and extracted scDEED Loom raw data. "
                    "The source matrix is genes by cells and was transposed to cells by genes."
                ),
            }
            metadata.update(
                obs_metadata_fields(
                    obs_schema=obs_schema,
                    identifier_column="sample_id",
                    default_color_by=default_color_by,
                    label_columns=label_columns,
                    colorable_columns=label_columns,
                    searchable_columns=metadata["searchable_columns"],
                    filterable_columns=label_columns,
                )
            )
            write_metadata(metadata_out, metadata)
            write_checksums(
                output_dir / "checksums.txt",
                [features_out, obs_out, metadata_out, *reference_files],
                base_dir=output_dir,
            )
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, str(exc)
    return True, f"prepared {repo_relative(output_dir)}"


def prepare_r_table_candidate(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    force: bool,
) -> tuple[bool, str]:
    dataset_id = safe_dataset_id(str(candidate["dataset_id"]))
    feature_path_value = str(candidate["feature_path"])
    label_path_value = str(candidate["obs_path"])
    feature_path = REPO_ROOT / feature_path_value
    label_path = REPO_ROOT / label_path_value
    if not feature_path.exists() or not label_path.exists():
        return False, "feature or label raw source file is missing"

    output_dir = PROCESSED_ROOT / f"scdeed_{dataset_id}"
    try:
        prepare_output_dir(output_dir, force)
        feature_frame = read_r_object(feature_path, str(candidate["feature_object"]))
        features = feature_frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, found shape {features.shape}")

        label_frame = read_r_object(label_path, str(candidate["obs_object"]))
        label_series = first_column(label_frame, str(candidate.get("obs_column") or ""))
        if int(label_series.shape[0]) != int(features.shape[0]):
            raise ValueError(
                f"row mismatch: features={features.shape[0]}, labels={label_series.shape[0]}"
            )

        labels, label_mapping, label_names = encode_labels(label_series)
        obs = pd.DataFrame(
            {
                "sample_id": np.arange(features.shape[0], dtype=np.int64),
                "label": labels,
                "label_name": label_names,
                "cluster_label": label_names,
            }
        )

        raw_paths = [repo_relative(feature_path), repo_relative(label_path)]
        for extra in candidate.get("obs_extra", []):
            extra_path = REPO_ROOT / str(extra["path"])
            extra_frame = read_r_object(extra_path, str(extra["object"]))
            extra_values = first_column(extra_frame)
            if int(extra_values.shape[0]) != int(features.shape[0]):
                continue
            column = str(extra["column"])
            try:
                obs[column] = pd.to_numeric(extra_values, errors="raise")
            except (TypeError, ValueError):
                obs[column] = extra_values
            raw_paths.append(repo_relative(extra_path))

        label_columns = ["label", "label_name", "cluster_label"]
        for column in ["population_numeric"]:
            if column in obs.columns:
                label_columns.append(column)
        obs_schema = obs_schema_from_frame(obs, label_columns)

        features_out = output_dir / "features.npy"
        obs_out = output_dir / "obs.csv.gz"
        metadata_out = output_dir / "metadata.json"
        np.save(features_out, features)
        write_obs(obs_out, obs)

        reference_records: dict[str, dict[str, Any]] = {}
        reference_files: list[Path] = []
        reference_root = output_dir / "reference_embeddings"
        for name, record in sorted(candidate.get("reference_embeddings", {}).items()):
            reference_path_value = str(record["path"])
            reference_path = REPO_ROOT / reference_path_value
            reference_frame = read_r_object(reference_path, str(record["object"]))
            reference = reference_frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
            if reference.ndim != 2 or int(reference.shape[0]) != int(features.shape[0]):
                continue
            reference_root.mkdir(parents=True, exist_ok=True)
            out_path = reference_root / f"{safe_dataset_id(name)}.npy"
            np.save(out_path, reference)
            reference_files.append(out_path)
            raw_paths.append(repo_relative(reference_path))
            reference_records[safe_dataset_id(name)] = {
                "file": out_path.relative_to(output_dir).as_posix(),
                "source": f"{reference_path_value}:{record['object']}",
                "shape": [int(value) for value in reference.shape],
                "dtype": "float32",
                "role": "reference_embedding",
            }

        metric_records: dict[str, dict[str, Any]] = {}
        metric_files: list[Path] = []
        metric_root = output_dir / "reference_metrics"
        for name, record in sorted(candidate.get("reference_metrics", {}).items()):
            metric_path_value = str(record["path"])
            metric_path = REPO_ROOT / metric_path_value
            metric_frame = read_r_object(metric_path, str(record["object"]))
            metric_root.mkdir(parents=True, exist_ok=True)
            out_path = metric_root / f"{safe_dataset_id(name)}.csv.gz"
            with gzip.open(out_path, "wt", newline="", encoding="utf-8") as handle:
                metric_frame.to_csv(handle, index=False)
            metric_files.append(out_path)
            raw_paths.append(repo_relative(metric_path))
            metric_records[safe_dataset_id(name)] = {
                "file": out_path.relative_to(output_dir).as_posix(),
                "source": f"{metric_path_value}:{record['object']}",
                "shape": [int(value) for value in metric_frame.shape],
                "role": "scdeed_output",
            }

        metadata = {
            "dataset_id": f"scdeed_{dataset_id}",
            "display_name": f"scDEED {dataset_id}",
            "family": "scdeed",
            "kind": "single_cell",
            "source_type": "zenodo_record",
            "zenodo_record_id": ZENODO_RECORD_ID,
            "source_url": SOURCE_URL,
            "source_publication": SOURCE_PUBLICATION,
            "raw_path": repo_relative(feature_path.parent),
            "raw_paths": sorted(set(raw_paths)),
            "raw_role": "primary_source",
            "primary_array": "features.npy",
            "feature_source": f"{feature_path_value}:{candidate['feature_object']}",
            "feature_source_path": repo_relative(feature_path),
            "feature_shape": [int(value) for value in features.shape],
            "feature_dtype": "float32",
            "source_feature_dtype": "R data.frame",
            "obs_file": "obs.csv.gz",
            "identifier_column": "sample_id",
            "default_color_by": "label_name",
            "label_columns": label_columns,
            "colorable_columns": label_columns,
            "searchable_columns": ["label_name", "cluster_label"],
            "filterable_columns": label_columns,
            "reference_embeddings": reference_records,
            "reference_metrics": metric_records,
            "label_mapping": label_mapping,
            "n_classes": len(label_mapping),
            "skipped_or_unrecognized_sources": skipped_sources(
                manifest,
                dataset_id,
                set(raw_paths),
            ),
            "metric": "euclidean",
            "generated_by": repo_relative(Path(__file__)),
            "generated_at": utc_now_iso(),
            "version": 1,
            "preprocessing": (
                "Prepared from downloaded and extracted scDEED Zenodo raw RData/Rds files. "
                f"Features were read from {feature_path_value}:{candidate['feature_object']} "
                "and converted to float32."
            ),
        }
        metadata.update(
            obs_metadata_fields(
                obs_schema=obs_schema,
                identifier_column="sample_id",
                default_color_by="label_name",
                label_columns=label_columns,
                colorable_columns=label_columns,
                searchable_columns=["label_name", "cluster_label"],
                filterable_columns=label_columns,
            )
        )
        write_metadata(metadata_out, metadata)
        write_checksums(
            output_dir / "checksums.txt",
            [features_out, obs_out, metadata_out, *reference_files, *metric_files],
            base_dir=output_dir,
        )
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, str(exc)
    return True, f"prepared {repo_relative(output_dir)}"


def normalized_cart_cell_id(value: Any) -> str:
    text = str(value)
    if text.endswith("-1"):
        return text[:-2] + ".1"
    return text.replace("-", ".")


def prepare_cart_candidate(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    force: bool,
) -> tuple[bool, str]:
    dataset_id = safe_dataset_id(str(candidate["dataset_id"]))
    feature_path_value = str(candidate["feature_path"])
    obs_path_value = str(candidate["obs_path"])
    feature_path = REPO_ROOT / feature_path_value
    obs_path = REPO_ROOT / obs_path_value
    full_metadata_path_value = candidate.get("full_metadata_path")
    full_metadata_path = REPO_ROOT / str(full_metadata_path_value) if full_metadata_path_value else None
    if not feature_path.exists() or not obs_path.exists():
        return False, "CART feature or metadata source file is missing"

    output_dir = PROCESSED_ROOT / f"scdeed_{dataset_id}"
    try:
        prepare_output_dir(output_dir, force)
        counts = pd.read_csv(feature_path, sep="\t", index_col=0)
        features = counts.T.to_numpy(dtype=np.float32, copy=True)
        cell_ids = pd.Index([str(value) for value in counts.columns], name="cell_id")
        del counts

        metadata = pd.read_csv(obs_path, sep="\t", index_col=0)
        metadata.index = metadata.index.map(str)
        metadata = metadata.reindex(cell_ids)
        if metadata.isna().all(axis=None):
            raise ValueError("CART metadata could not be aligned to count matrix cell IDs")
        obs = metadata.reset_index(names="cell_id")
        obs.insert(0, "sample_id", np.arange(obs.shape[0], dtype=np.int64))
        if "label" in obs.columns:
            obs["label_name"] = obs["label"].map(label_name)

        raw_paths = [repo_relative(feature_path), repo_relative(obs_path)]
        reference_records: dict[str, dict[str, Any]] = {}
        reference_files: list[Path] = []
        if full_metadata_path is not None and full_metadata_path.exists():
            full_metadata = pd.read_csv(full_metadata_path)
            cell_column = "Cell" if "Cell" in full_metadata.columns else None
            if cell_column is not None:
                full_metadata["cell_id"] = full_metadata[cell_column].map(normalized_cart_cell_id)
                full_metadata = full_metadata.set_index("cell_id").reindex(cell_ids)
                for column in [
                    "nGene",
                    "nUMI",
                    "percent.mito",
                    "Patient",
                    "Group",
                    "Disease",
                    "Seurat.clusters",
                ]:
                    if column in full_metadata.columns:
                        obs[column] = full_metadata[column].to_numpy()
                if {"tSNE_1", "tSNE_2"}.issubset(full_metadata.columns):
                    tsne = full_metadata[["tSNE_1", "tSNE_2"]].to_numpy(dtype=np.float32)
                    reference_root = output_dir / "reference_embeddings"
                    reference_root.mkdir(parents=True, exist_ok=True)
                    tsne_path = reference_root / "tsne.npy"
                    np.save(tsne_path, tsne)
                    reference_files.append(tsne_path)
                    reference_records["tsne"] = {
                        "file": tsne_path.relative_to(output_dir).as_posix(),
                        "source": f"{full_metadata_path_value}:tSNE_1,tSNE_2",
                        "shape": [int(value) for value in tsne.shape],
                        "dtype": "float32",
                        "role": "reference_embedding",
                    }
                raw_paths.append(repo_relative(full_metadata_path))

        metric_records: dict[str, dict[str, Any]] = {}
        metric_files: list[Path] = []
        metric_root = output_dir / "reference_metrics"
        for name, record in sorted(candidate.get("reference_metrics", {}).items()):
            metric_path_value = str(record["path"])
            metric_path = REPO_ROOT / metric_path_value
            metric_frame = read_r_object(metric_path, str(record["object"]))
            metric_root.mkdir(parents=True, exist_ok=True)
            out_path = metric_root / f"{safe_dataset_id(name)}.csv.gz"
            with gzip.open(out_path, "wt", newline="", encoding="utf-8") as handle:
                metric_frame.to_csv(handle, index=False)
            metric_files.append(out_path)
            raw_paths.append(repo_relative(metric_path))
            metric_records[safe_dataset_id(name)] = {
                "file": out_path.relative_to(output_dir).as_posix(),
                "source": f"{metric_path_value}:{record['object']}",
                "shape": [int(value) for value in metric_frame.shape],
                "role": "scdeed_output",
            }

        label_columns = [
            column
            for column in ["label", "label_name", "Disease", "Group", "Patient", "Seurat.clusters"]
            if column in obs.columns
        ]
        obs_schema = obs_schema_from_frame(obs, label_columns)

        features_out = output_dir / "features.npy"
        obs_out = output_dir / "obs.csv.gz"
        metadata_out = output_dir / "metadata.json"
        np.save(features_out, features)
        write_obs(obs_out, obs)

        payload = {
            "dataset_id": f"scdeed_{dataset_id}",
            "display_name": "scDEED CART",
            "family": "scdeed",
            "kind": "single_cell",
            "source_type": "zenodo_record",
            "zenodo_record_id": ZENODO_RECORD_ID,
            "source_url": SOURCE_URL,
            "source_publication": SOURCE_PUBLICATION,
            "raw_path": repo_relative(feature_path.parent),
            "raw_paths": sorted(set(raw_paths)),
            "raw_role": "primary_source",
            "primary_array": "features.npy",
            "feature_source": f"{feature_path_value}:transposed_cells_by_genes",
            "feature_source_path": repo_relative(feature_path),
            "feature_shape": [int(value) for value in features.shape],
            "feature_dtype": "float32",
            "source_feature_dtype": "normalized_counts_tsv",
            "obs_file": "obs.csv.gz",
            "identifier_column": "sample_id",
            "default_color_by": "label_name" if "label_name" in obs.columns else None,
            "label_columns": label_columns,
            "colorable_columns": label_columns,
            "searchable_columns": [
                column
                for column in ["label_name", "Disease", "Group", "Patient"]
                if column in obs.columns
            ],
            "filterable_columns": label_columns,
            "reference_embeddings": reference_records,
            "reference_metrics": metric_records,
            "skipped_or_unrecognized_sources": skipped_sources(
                manifest,
                dataset_id,
                set(raw_paths),
            ),
            "metric": "euclidean",
            "generated_by": repo_relative(Path(__file__)),
            "generated_at": utc_now_iso(),
            "version": 1,
            "preprocessing": (
                "Prepared from downloaded and extracted CART normalized count TSV. "
                "The source matrix is genes by cells and was transposed to cells by genes."
            ),
        }
        payload.update(
            obs_metadata_fields(
                obs_schema=obs_schema,
                identifier_column="sample_id",
                default_color_by=payload["default_color_by"],
                label_columns=label_columns,
                colorable_columns=label_columns,
                searchable_columns=payload["searchable_columns"],
                filterable_columns=label_columns,
            )
        )
        write_metadata(metadata_out, payload)
        write_checksums(
            output_dir / "checksums.txt",
            [features_out, obs_out, metadata_out, *reference_files, *metric_files],
            base_dir=output_dir,
        )
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, str(exc)
    return True, f"prepared {repo_relative(output_dir)}"


def prepare_candidate(candidate: dict[str, Any], manifest: dict[str, Any], force: bool) -> tuple[bool, str]:
    source_format = candidate.get("source_format")
    if source_format == "h5ad":
        return prepare_h5ad_candidate(candidate, manifest, force)
    if source_format == "loom":
        return prepare_loom_candidate(candidate, manifest, force)
    if source_format == "r_table":
        return prepare_r_table_candidate(candidate, manifest, force)
    if source_format == "cart_tsv":
        return prepare_cart_candidate(candidate, manifest, force)
    return False, f"unsupported candidate source format: {source_format}"


def main() -> int:
    args = parse_args()
    try:
        manifest = read_manifest()
        candidates = manifest.get("prepare_candidates", [])
        if args.list:
            if not candidates:
                print("No recognized prepare candidates. Run inspect.py first.")
            for candidate in candidates:
                raw_source = candidate.get("raw_path") or candidate.get("feature_path")
                feature_source = (
                    candidate.get("feature_key")
                    or candidate.get("feature_object")
                    or candidate.get("feature_path")
                )
                print(
                    f"{candidate.get('dataset_id')}\t{candidate.get('status')}\t"
                    f"{candidate.get('source_format')}\t{raw_source}\t"
                    f"{feature_source}"
                )
            return 0
        if args.all:
            selected = candidates
        elif args.dataset:
            candidate = candidate_by_id(manifest, args.dataset)
            selected = [candidate] if candidate is not None else []
            if not selected:
                raise ValueError(f"Unknown dataset candidate: {args.dataset}")
        else:
            raise ValueError("Provide --list, --all, or --dataset")

        prepared: list[str] = []
        skipped: list[tuple[str, str]] = []
        for candidate in selected:
            ok, message = prepare_candidate(candidate, manifest, args.force)
            dataset_id = str(candidate.get("dataset_id"))
            if ok:
                prepared.append(dataset_id)
                print(f"[{dataset_id}] {message}")
            else:
                skipped.append((dataset_id, message))
                print(f"[{dataset_id}] SKIP: {message}", file=sys.stderr)
        if prepared:
            print("Prepared: " + ", ".join(prepared))
        if skipped:
            print(
                "Skipped: "
                + ", ".join(f"{dataset_id} ({reason})" for dataset_id, reason in skipped),
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
