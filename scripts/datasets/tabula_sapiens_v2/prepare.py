from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import write_checksums
from common.metadata import utc_now_iso, write_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTION_ID = "e5f58829-1a66-40b5-a624-9046778e74f5"
COLLECTION_API = (
    "https://api.cellxgene.cziscience.com/curation/v1/collections/"
    f"{COLLECTION_ID}"
)
RAW_ROOT = REPO_ROOT / "datasets/raw/tabula_sapiens_v2"
H5AD_ROOT = RAW_ROOT / "h5ad"
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"

DEFAULT_COLOR_PRIORITY = [
    "cell_type",
    "cell_type_ontology_term_id",
    "cell_ontology_class",
    "free_annotation",
    "organ_tissue",
    "tissue",
]

LABEL_PRIORITY = [
    "cell_type",
    "cell_type_ontology_term_id",
    "cell_ontology_class",
    "free_annotation",
    "organ_tissue",
    "tissue",
    "assay",
    "method",
    "donor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Tabula Sapiens v2 H5AD files into processed datasets."
    )
    parser.add_argument("--all", action="store_true", help="Prepare all raw H5AD files.")
    parser.add_argument("--list", action="store_true", help="List available raw H5AD files.")
    parser.add_argument(
        "--dataset",
        help="Prepare one dataset by raw stem, H5AD file name, or processed dataset_id.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N selected files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed output.")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def safe_dataset_id(raw_name: str) -> str:
    stem = Path(raw_name).stem
    stem = re.sub(r"^TS2[_\-\s]*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^Tabula[_\-\s]*Sapiens[_\-\s]*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_").lower()
    if not stem:
        stem = "dataset"
    return f"tabula_sapiens_v2_{stem}"


def display_name_from_file(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^TS2[_\-\s]*", "", stem, flags=re.IGNORECASE)
    return f"Tabula Sapiens v2 {stem.replace('_', ' ')}"


def list_h5ad_files() -> list[Path]:
    if not H5AD_ROOT.exists():
        return []
    return sorted(H5AD_ROOT.glob("*.h5ad"))


def select_h5ad_files(args: argparse.Namespace) -> list[Path]:
    files = list_h5ad_files()
    if args.list:
        return files[: args.limit] if args.limit is not None else files
    if args.all:
        selected = files
    elif args.dataset:
        query = args.dataset
        query_stem = Path(query).stem
        selected = [
            path
            for path in files
            if path.name == query
            or path.stem == query_stem
            or safe_dataset_id(path.stem) == query
        ]
        if not selected:
            raise FileNotFoundError(
                f"No raw H5AD matched {query!r}. Run download.py first or use --list."
            )
    else:
        raise ValueError("Provide --all, --list, or --dataset")

    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to overwrite it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8", errors="replace")
    if values.dtype.kind == "O":
        return np.asarray([decode_scalar(value) for value in values], dtype=object)
    return values


def read_categorical_group(group: h5py.Group) -> pd.Series:
    categories = decode_array(group["categories"][:])
    codes = group["codes"][:]
    values: list[Any] = []
    for code in codes:
        code_int = int(code)
        values.append("" if code_int < 0 else decode_scalar(categories[code_int]))
    return pd.Series(pd.Categorical(values))


def read_obs_column(name: str, obj: h5py.Dataset | h5py.Group, n_rows: int) -> tuple[pd.Series | None, str | None]:
    try:
        if isinstance(obj, h5py.Group):
            if "categories" in obj and "codes" in obj:
                series = read_categorical_group(obj)
            else:
                return None, "unsupported h5py group without categories/codes"
        elif isinstance(obj, h5py.Dataset):
            values = obj[:]
            if values.ndim != 1:
                return None, f"unsupported non-1D dataset with shape {values.shape}"
            series = pd.Series(decode_array(values))
        else:
            return None, f"unsupported h5py object type {type(obj).__name__}"
    except Exception as exc:
        return None, str(exc)

    if len(series) != n_rows:
        return None, f"row count mismatch: expected {n_rows}, got {len(series)}"
    return series, None


def obs_index_column_name(obs_group: h5py.Group) -> str | None:
    attr_value = obs_group.attrs.get("_index")
    if attr_value is not None:
        attr_value = decode_scalar(attr_value)
        if isinstance(attr_value, str) and attr_value in obs_group:
            return attr_value
    if "_index" in obs_group:
        return "_index"
    if "index" in obs_group:
        return "index"
    return None


def read_obs_dataframe(obs_group: h5py.Group, n_rows: int) -> tuple[pd.DataFrame, list[dict[str, str]], str | None]:
    skipped: list[dict[str, str]] = []
    columns: dict[str, pd.Series] = {"sample_id": pd.Series(np.arange(n_rows, dtype=np.int64))}

    original_index_source = obs_index_column_name(obs_group)
    original_index_column = None
    if original_index_source is not None:
        series, reason = read_obs_column(original_index_source, obs_group[original_index_source], n_rows)
        if series is not None:
            columns["original_obs_index"] = series.astype(str)
            original_index_column = "original_obs_index"
        else:
            skipped.append({"column": original_index_source, "reason": reason or "could not read index"})

    for column_name in obs_group.keys():
        if column_name == original_index_source:
            continue
        series, reason = read_obs_column(column_name, obs_group[column_name], n_rows)
        if series is None:
            skipped.append({"column": column_name, "reason": reason or "could not read column"})
            continue
        columns[column_name] = series

    frame = pd.DataFrame(columns)
    for column in frame.columns:
        if pd.api.types.is_object_dtype(frame[column]) or isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else str(value))
    return frame, skipped, original_index_column


def role_for_column(column: str, series: pd.Series) -> str:
    lower = column.lower()
    if column == "sample_id":
        return "identifier"
    if column == "original_obs_index":
        return "original_identifier"
    if "cell" in lower or "annotation" in lower or "ontology" in lower:
        return "biological_label"
    if any(token in lower for token in ["donor", "batch", "assay", "method"]):
        return "batch_or_source"
    if "organ" in lower or "tissue" in lower:
        return "tissue_or_organ"
    if any(token in lower for token in ["disease", "development", "stage", "time"]):
        return "biological_covariate"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous_metadata"
    return "metadata"


def schema_dtype(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    return "string"


def build_obs_schema(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        examples = [str(value) for value in non_null.astype(str).unique()[:3]]
        schema[column] = {
            "dtype": schema_dtype(series),
            "role": role_for_column(column, series),
            "description": f"Column {column} from the H5AD obs table.",
            "n_unique": int(series.nunique(dropna=True)),
            "example_values": examples,
        }
    return schema


def choose_default_color_by(columns: set[str]) -> str | None:
    for column in DEFAULT_COLOR_PRIORITY:
        if column in columns:
            return column
    return None


def metadata_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], str | None]:
    schema = build_obs_schema(frame)
    columns = set(frame.columns)
    default_color_by = choose_default_color_by(columns)

    label_columns = [column for column in LABEL_PRIORITY if column in columns]
    if default_color_by is not None and default_color_by not in label_columns:
        label_columns.insert(0, default_color_by)

    colorable_columns: list[str] = []
    filterable_columns: list[str] = []
    searchable_columns: list[str] = []
    for column, config in schema.items():
        if column == "sample_id":
            continue
        dtype = config["dtype"]
        n_unique = int(config.get("n_unique", 0))
        if column == "original_obs_index":
            searchable_columns.append(column)
            continue
        if dtype in {"string", "category", "bool"}:
            filterable_columns.append(column)
            if 1 < n_unique <= 1000:
                colorable_columns.append(column)

    if default_color_by is not None and default_color_by not in colorable_columns:
        colorable_columns.insert(0, default_color_by)
    return label_columns, colorable_columns, searchable_columns, filterable_columns, default_color_by


def keys_summary(handle: h5py.File) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ["obsm/X_scvi", "obsm/X_umap", "obs"]:
        exists = key in handle
        record: dict[str, Any] = {"exists": exists}
        if exists and isinstance(handle[key], h5py.Dataset):
            record["shape"] = [int(value) for value in handle[key].shape]
            record["dtype"] = str(handle[key].dtype)
        elif exists and isinstance(handle[key], h5py.Group):
            record["keys"] = sorted(str(value) for value in handle[key].keys())
        summary[key] = record
    return summary


def metadata_payload(
    *,
    h5ad_path: Path,
    dataset_id: str,
    display_name: str,
    feature_shape: list[int],
    reference_shape: list[int] | None,
    reference_file: str | None,
    obs_frame: pd.DataFrame,
    skipped_obs_columns: list[dict[str, str]],
    raw_h5ad_keys_summary: dict[str, Any],
    original_index_column: str | None,
) -> dict[str, Any]:
    obs_schema = build_obs_schema(obs_frame)
    label_columns, colorable_columns, searchable_columns, filterable_columns, default_color_by = (
        metadata_columns(obs_frame)
    )
    return {
        "dataset_id": dataset_id,
        "display_name": display_name,
        "family": "tabula_sapiens_v2",
        "kind": "single_cell",
        "source_type": "cellxgene_collection",
        "collection_id": COLLECTION_ID,
        "collection_api": COLLECTION_API,
        "raw_path": repo_relative(h5ad_path),
        "raw_role": "primary_source",
        "primary_array": "features.npy",
        "feature_source": "obsm/X_scvi",
        "feature_shape": feature_shape,
        "feature_dtype": "float32",
        "obs_file": "obs.csv.gz",
        "reference_embedding_file": reference_file,
        "reference_embedding_source": "obsm/X_umap" if reference_file else None,
        "reference_embedding_shape": reference_shape,
        "metric": "euclidean",
        "generated_by": repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": 1,
        "preprocessing": (
            "Read Tabula Sapiens v2 H5AD with h5py, saved obsm/X_scvi as float32 "
            "features.npy, exported obs metadata to obs.csv.gz, and saved obsm/X_umap "
            "as reference_embedding.npy when available."
        ),
        "obs_schema": obs_schema,
        "identifier_column": "sample_id",
        "original_index_column": original_index_column,
        "default_color_by": default_color_by,
        "label_columns": label_columns,
        "colorable_columns": colorable_columns,
        "searchable_columns": searchable_columns,
        "filterable_columns": filterable_columns,
        "skipped_obs_columns": skipped_obs_columns,
        "raw_h5ad_keys_summary": raw_h5ad_keys_summary,
    }


def prepare_one(h5ad_path: Path, force: bool) -> tuple[bool, str]:
    dataset_id = safe_dataset_id(h5ad_path.stem)
    output_dir = PROCESSED_ROOT / dataset_id
    try:
        prepare_output_dir(output_dir, force)
    except Exception as exc:
        return False, str(exc)

    try:
        with h5py.File(h5ad_path, "r") as handle:
            raw_h5ad_keys_summary = keys_summary(handle)
            if "obsm/X_scvi" not in handle:
                shutil.rmtree(output_dir, ignore_errors=True)
                return False, "missing required obsm/X_scvi"

            features = np.asarray(handle["obsm/X_scvi"][:], dtype=np.float32)
            n_rows = int(features.shape[0])
            if "obs" not in handle or not isinstance(handle["obs"], h5py.Group):
                shutil.rmtree(output_dir, ignore_errors=True)
                return False, "missing required obs group"
            obs_frame, skipped_obs_columns, original_index_column = read_obs_dataframe(handle["obs"], n_rows)

            reference = None
            reference_shape = None
            reference_file = None
            if "obsm/X_umap" in handle:
                reference = np.asarray(handle["obsm/X_umap"][:], dtype=np.float32)
                if int(reference.shape[0]) != n_rows:
                    skipped_obs_columns.append(
                        {
                            "column": "obsm/X_umap",
                            "reason": (
                                "reference embedding row count mismatch: "
                                f"{reference.shape[0]} vs {n_rows}"
                            ),
                        }
                    )
                    reference = None
                else:
                    reference_shape = [int(value) for value in reference.shape]
                    reference_file = "reference_embedding.npy"

        features_path = output_dir / "features.npy"
        obs_path = output_dir / "obs.csv.gz"
        metadata_path = output_dir / "metadata.json"
        np.save(features_path, features)
        obs_frame.to_csv(obs_path, index=False, compression="gzip", quoting=csv.QUOTE_MINIMAL)

        checksum_files = [features_path, obs_path, metadata_path]
        if reference is not None:
            reference_path = output_dir / "reference_embedding.npy"
            np.save(reference_path, reference)
            checksum_files.insert(2, reference_path)

        metadata = metadata_payload(
            h5ad_path=h5ad_path,
            dataset_id=dataset_id,
            display_name=display_name_from_file(h5ad_path),
            feature_shape=[int(value) for value in features.shape],
            reference_shape=reference_shape,
            reference_file=reference_file,
            obs_frame=obs_frame,
            skipped_obs_columns=skipped_obs_columns,
            raw_h5ad_keys_summary=raw_h5ad_keys_summary,
            original_index_column=original_index_column,
        )
        write_metadata(metadata_path, metadata)
        write_checksums(output_dir / "checksums.txt", checksum_files, base_dir=output_dir)
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        return False, str(exc)

    return True, f"prepared {repo_relative(output_dir)}"


def main() -> int:
    args = parse_args()
    try:
        selected = select_h5ad_files(args)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.list:
        if not selected:
            print(f"No H5AD files found under {H5AD_ROOT}. Run download.py first.")
        for path in selected:
            print(f"{path.name}\t{safe_dataset_id(path.stem)}")
        return 0

    if not selected:
        print(f"FAIL: no H5AD files found under {H5AD_ROOT}. Run download.py first.", file=sys.stderr)
        return 1

    skipped: list[tuple[str, str]] = []
    prepared: list[str] = []
    for path in selected:
        ok, message = prepare_one(path, args.force)
        dataset_id = safe_dataset_id(path.stem)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
