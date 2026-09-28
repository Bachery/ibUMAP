"""Prepare the tabular ML datasets in the shared processed format.

Reads the verified raw files written by download.py from
datasets/raw_cache/tabular_ml/<dataset_id>/ and writes

    datasets/processed/<dataset_id>/
      features.npy   float32
      target.npy     int32
      obs.csv.gz     sample_id, label, label_name (gzip with mtime=0, byte-stable)
      metadata.json
      checksums.txt
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import sha256_file, write_checksums
from common.metadata import build_obs_schema, obs_metadata_fields, utc_now_iso, write_metadata

from _sources import DATASETS, FOREST_CLASS_DESCRIPTIONS, DatasetSource


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CACHE_ROOT = REPO_ROOT / "datasets/raw_cache/tabular_ml"
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"


OBS_SCHEMA = build_obs_schema(
    {
        "sample_id": {
            "dtype": "int64",
            "role": "identifier",
            "description": "Stable row identifier matching the row index of features.npy.",
        },
        "label": {
            "dtype": "int32",
            "role": "class_label",
            "description": "Stable int32 class label encoded from the source label values.",
        },
        "label_name": {
            "dtype": "string",
            "role": "class_name",
            "description": "Source label value as a string (see label_mapping in metadata.json).",
        },
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tabular ML datasets from datasets/raw_cache/tabular_ml/<dataset_id>/."
    )
    parser.add_argument("dataset_id", nargs="?", choices=sorted(DATASETS))
    parser.add_argument("--all", action="store_true", help="Prepare all supported tabular ML datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed output.")
    parser.add_argument("--raw-root", type=Path, default=RAW_CACHE_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    return parser.parse_args()


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(DATASETS)
    if args.dataset_id:
        return [args.dataset_id]
    raise ValueError("Provide a dataset_id or --all")


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to overwrite it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def verify_raw_files(config: DatasetSource, raw_dir: Path) -> None:
    for source in config.files:
        path = raw_dir / source.file_name
        if not path.exists():
            raise FileNotFoundError(
                f"missing raw file {repo_relative(path)}; run "
                f"scripts/datasets/tabular_ml/download.py {config.dataset_id}"
            )
        actual = sha256_file(path)
        if actual != source.sha256:
            raise ValueError(f"{repo_relative(path)} has SHA-256 {actual}, expected {source.sha256}")


# --------------------------------------------------------------------------- loaders


def load_espadoto_npy(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(raw_dir / "X.npy", allow_pickle=False)
    y = np.load(raw_dir / "y.npy", allow_pickle=False)
    return x, y


def load_uci_forest_csv(config: DatasetSource, raw_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Concatenate the CSV files in the order listed in _sources.py.

    Returns Fortran-ordered float64 features, int8 sorted-category codes (as pandas category
    codes would give), and the sorted class letters.
    """
    header: list[str] | None = None
    classes: list[str] = []
    rows: list[list[float]] = []
    for source in config.files:
        with (raw_dir / source.file_name).open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            file_header = [column.strip() for column in next(reader)]
            if header is None:
                header = file_header
            elif file_header != header:
                raise ValueError(f"column mismatch in {source.file_name}")
            for record in reader:
                if not record:
                    continue
                classes.append(record[0].strip())
                rows.append([float(value) for value in record[1:]])
    assert header is not None and header[0] == "class"
    # Fortran order matches the historical arrays byte for byte (they came from
    # a pandas DataFrame's .values); np.asarray(..., float32) below keeps it.
    x = np.asfortranarray(np.asarray(rows, dtype=np.float64))
    categories = sorted(set(classes))
    codes = np.asarray([categories.index(value) for value in classes], dtype=np.int8)
    return x, codes, categories


# --------------------------------------------------------------------------- encoding


def _label_name(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def build_label_encoding(labels: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"labels must be a non-empty 1-D array, found shape {labels.shape}")
    if np.issubdtype(labels.dtype, np.floating) and not np.isfinite(labels).all():
        raise ValueError("labels contain non-finite values")

    label_names = [_label_name(value) for value in labels]
    unique_names = sorted(set(label_names))
    name_to_label = {name: index for index, name in enumerate(unique_names)}
    encoded = np.asarray([name_to_label[name] for name in label_names], dtype=np.int32)
    mapping = [
        {"label": index, "label_name": name, "original_value": name}
        for index, name in enumerate(unique_names)
    ]
    return encoded, mapping


def write_obs(path: Path, labels: np.ndarray, label_mapping: list[dict[str, Any]]) -> None:
    names_by_label = {int(item["label"]): str(item["label_name"]) for item in label_mapping}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["sample_id", "label", "label_name"])
    writer.writeheader()
    for sample_id, label in enumerate(labels):
        writer.writerow(
            {"sample_id": sample_id, "label": int(label), "label_name": names_by_label[int(label)]}
        )
    # filename="" and mtime=0 keep the gzip header free of paths and timestamps.
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(buffer.getvalue().encode("utf-8"))


# --------------------------------------------------------------------------- metadata


def metadata_payload(
    config: DatasetSource,
    raw_dir: Path,
    feature_shape: list[int],
    original_feature_dtype: str,
    original_label_dtype: str,
    label_mapping: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_id": config.dataset_id,
        "display_name": config.display_name,
        "kind": "tabular_ml",
        "family": "tabular_ml",
        "source_type": "direct_download",
        "source_name": config.source_name,
        "source_url": config.source_url,
        "source_publication": config.source_publication,
        "upstream_source": config.upstream,
        # In the order download.py tries them: archives first, then per-file mirrors.
        "download_urls": [archive.url for archive in config.archives]
        + [url for source in config.files for url in source.urls],
        "raw_role": "raw_cache",
        "raw_path": repo_relative(raw_dir),
        "raw_paths": [repo_relative(raw_dir / source.file_name) for source in config.files],
        "raw_sha256": {source.file_name: source.sha256 for source in config.files},
        "original_feature_dtype": original_feature_dtype,
        "original_label_dtype": original_label_dtype,
        "label_encoding_strategy": "sorted_unique_label_name",
        "label_mapping": label_mapping,
        "n_classes": len(label_mapping),
        "primary_array": "features.npy",
        "feature_shape": feature_shape,
        "feature_dtype": "float32",
        "obs_file": "obs.csv.gz",
        "target_file": "target.npy",
        "target_dtype": "int32",
        "target_source_column": "label",
        "task_type": "classification",
        "identifier_column": "sample_id",
        "metric": "euclidean",
        "generated_by": repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": 1,
        "preprocessing": (
            "Upstream: "
            + config.upstream_preprocessing
            + " Here: features cast to float32, labels encoded by sorted unique label name to int32."
        ),
    }
    payload.update(
        obs_metadata_fields(
            obs_schema=OBS_SCHEMA,
            identifier_column="sample_id",
            default_color_by="label_name",
            label_columns=["label", "label_name"],
            colorable_columns=["label", "label_name"],
            searchable_columns=[],
            filterable_columns=["label", "label_name"],
        )
    )
    return payload


# --------------------------------------------------------------------------- main


def prepare_one(dataset_id: str, raw_root: Path, processed_root: Path, force: bool) -> str:
    config = DATASETS[dataset_id]
    raw_dir = raw_root / dataset_id
    verify_raw_files(config, raw_dir)

    categories: list[str] | None = None
    if config.loader == "espadoto_npy":
        x, y = load_espadoto_npy(raw_dir)
    elif config.loader == "uci_forest_csv":
        x, y, categories = load_uci_forest_csv(config, raw_dir)
    else:
        raise ValueError(f"unknown loader {config.loader!r}")

    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"bad shapes: X={x.shape}, y={y.shape}")
    if not np.isfinite(x).all():
        raise ValueError("features contain non-finite values")

    labels, label_mapping = build_label_encoding(y)
    if categories is not None:
        for item in label_mapping:
            letter = categories[int(item["original_value"])]
            item["original_value"] = letter
            item["description"] = FOREST_CLASS_DESCRIPTIONS.get(letter, "")

    output_dir = processed_root / dataset_id
    prepare_output_dir(output_dir, force)
    features_path = output_dir / "features.npy"
    target_path = output_dir / "target.npy"
    obs_path = output_dir / "obs.csv.gz"
    metadata_path = output_dir / "metadata.json"

    features = np.asarray(x, dtype=np.float32)
    np.save(features_path, features)
    np.save(target_path, labels.astype(np.int32, copy=False))
    write_obs(obs_path, labels, label_mapping)
    write_metadata(
        metadata_path,
        metadata_payload(
            config=config,
            raw_dir=raw_dir,
            feature_shape=[int(v) for v in features.shape],
            original_feature_dtype=str(x.dtype),
            original_label_dtype=str(y.dtype),
            label_mapping=label_mapping,
        ),
    )
    write_checksums(
        output_dir / "checksums.txt",
        [features_path, obs_path, target_path, metadata_path],
        base_dir=output_dir,
    )
    return f"prepared {repo_relative(output_dir)} {tuple(features.shape)}"


def main() -> int:
    args = parse_args()
    try:
        dataset_ids = selected_dataset_ids(args)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    failed: list[str] = []
    for dataset_id in dataset_ids:
        try:
            print(f"[{dataset_id}] " + prepare_one(dataset_id, args.raw_root, args.processed_root, args.force))
        except Exception as exc:
            failed.append(dataset_id)
            print(f"[{dataset_id}] FAIL: {exc}", file=sys.stderr)
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
