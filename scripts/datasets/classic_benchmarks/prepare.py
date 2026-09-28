from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import datasets as sklearn_datasets


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import write_checksums
from common.metadata import build_obs_schema, obs_metadata_fields, utc_now_iso, write_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    display_name: str
    loader_name: str
    task_type: str

    @property
    def source_loader(self) -> str:
        return f"sklearn.datasets.{self.loader_name}"


DATASETS: dict[str, DatasetConfig] = {
    "iris": DatasetConfig(
        dataset_id="iris",
        display_name="Iris",
        loader_name="load_iris",
        task_type="classification",
    ),
    "wine": DatasetConfig(
        dataset_id="wine",
        display_name="Wine",
        loader_name="load_wine",
        task_type="classification",
    ),
    "breast_cancer": DatasetConfig(
        dataset_id="breast_cancer",
        display_name="Breast Cancer Wisconsin Diagnostic",
        loader_name="load_breast_cancer",
        task_type="classification",
    ),
    "digits": DatasetConfig(
        dataset_id="digits",
        display_name="Digits",
        loader_name="load_digits",
        task_type="classification",
    ),
    "diabetes": DatasetConfig(
        dataset_id="diabetes",
        display_name="Diabetes",
        loader_name="load_diabetes",
        task_type="regression",
    ),
}


CLASSIFICATION_OBS_SCHEMA = build_obs_schema(
    {
        "sample_id": {
            "dtype": "int64",
            "role": "identifier",
            "description": "Stable row identifier matching the row index of features.npy.",
        },
        "label": {
            "dtype": "int32",
            "role": "class_label",
            "description": "Integer class label from the scikit-learn dataset.",
        },
        "label_name": {
            "dtype": "string",
            "role": "class_name",
            "description": "Human-readable class name from the scikit-learn dataset.",
        },
    }
)


REGRESSION_OBS_SCHEMA = build_obs_schema(
    {
        "sample_id": {
            "dtype": "int64",
            "role": "identifier",
            "description": "Stable row identifier matching the row index of features.npy.",
        },
        "target": {
            "dtype": "float32",
            "role": "continuous_target",
            "description": "Continuous regression target from the scikit-learn dataset.",
        },
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare small classic benchmark datasets from scikit-learn loaders."
    )
    parser.add_argument("dataset_id", nargs="?", choices=sorted(DATASETS))
    parser.add_argument("--all", action="store_true", help="Prepare all supported datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed output.")
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


def load_sklearn_dataset(config: DatasetConfig) -> Any:
    loader = getattr(sklearn_datasets, config.loader_name)
    return loader()


def write_classification_obs(path: Path, labels: np.ndarray, target_names: np.ndarray) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "label_name"])
        writer.writeheader()
        for sample_id, label in enumerate(labels):
            class_id = int(label)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "label": class_id,
                    "label_name": str(target_names[class_id]),
                }
            )


def write_regression_obs(path: Path, targets: np.ndarray) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "target"])
        writer.writeheader()
        for sample_id, target in enumerate(targets):
            writer.writerow({"sample_id": sample_id, "target": float(target)})


def base_metadata(
    config: DatasetConfig,
    feature_shape: list[int],
    feature_dtype: str,
) -> dict[str, Any]:
    return {
        "dataset_id": config.dataset_id,
        "display_name": config.display_name,
        "kind": "classic_benchmark",
        "source": "scikit-learn",
        "source_type": "python_library",
        "source_library": "scikit-learn",
        "source_loader": config.source_loader,
        "raw_role": "not_persisted",
        "raw_path": None,
        "download_url": None,
        "primary_array": "features.npy",
        "feature_shape": feature_shape,
        "feature_dtype": feature_dtype,
        "obs_file": "obs.csv.gz",
        "identifier_column": "sample_id",
        "metric": "euclidean",
        "generated_by": repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": 1,
    }


def classification_metadata(
    config: DatasetConfig,
    feature_shape: list[int],
    feature_dtype: str,
) -> dict[str, Any]:
    payload = base_metadata(config, feature_shape, feature_dtype)
    payload.update(
        {
            "target_file": "target.npy",
            "target_dtype": "int32",
            "target_source_column": "label",
            "preprocessing": (
                "Loaded from scikit-learn, converted data matrix to float32 features.npy, "
                "and stored sample labels in obs.csv.gz with int32 target.npy cache."
            ),
        }
    )
    payload.update(
        obs_metadata_fields(
            obs_schema=CLASSIFICATION_OBS_SCHEMA,
            identifier_column="sample_id",
            default_color_by="label_name",
            label_columns=["label", "label_name"],
            colorable_columns=["label", "label_name"],
            searchable_columns=[],
            filterable_columns=["label", "label_name"],
        )
    )
    return payload


def regression_metadata(
    config: DatasetConfig,
    feature_shape: list[int],
    feature_dtype: str,
) -> dict[str, Any]:
    payload = base_metadata(config, feature_shape, feature_dtype)
    payload.update(
        {
            "task_type": "regression",
            "target_file": None,
            "preprocessing": (
                "Loaded from scikit-learn, converted data matrix to float32 features.npy, "
                "and stored the continuous regression target in obs.csv.gz."
            ),
        }
    )
    payload.update(
        obs_metadata_fields(
            obs_schema=REGRESSION_OBS_SCHEMA,
            identifier_column="sample_id",
            default_color_by="target",
            label_columns=[],
            colorable_columns=["target"],
            searchable_columns=[],
            filterable_columns=[],
        )
    )
    return payload


def prepare_one(dataset_id: str, force: bool) -> None:
    config = DATASETS[dataset_id]
    output_dir = PROCESSED_ROOT / dataset_id
    prepare_output_dir(output_dir, force)

    bunch = load_sklearn_dataset(config)
    features = np.asarray(bunch.data, dtype=np.float32)
    features_path = output_dir / "features.npy"
    obs_path = output_dir / "obs.csv.gz"
    metadata_path = output_dir / "metadata.json"
    np.save(features_path, features)

    checksum_files = [features_path, obs_path]
    if config.task_type == "classification":
        labels = np.asarray(bunch.target, dtype=np.int32)
        target_names = np.asarray(bunch.target_names)
        target_path = output_dir / "target.npy"
        write_classification_obs(obs_path, labels, target_names)
        np.save(target_path, labels)
        metadata = classification_metadata(config, [int(v) for v in features.shape], str(features.dtype))
        checksum_files.append(target_path)
    elif config.task_type == "regression":
        targets = np.asarray(bunch.target, dtype=np.float32)
        write_regression_obs(obs_path, targets)
        metadata = regression_metadata(config, [int(v) for v in features.shape], str(features.dtype))
    else:
        raise ValueError(f"Unsupported task type for {dataset_id}: {config.task_type}")

    write_metadata(metadata_path, metadata)
    checksum_files.append(metadata_path)
    write_checksums(output_dir / "checksums.txt", checksum_files, base_dir=output_dir)
    print(f"[{dataset_id}] Prepared dataset: {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        for dataset_id in selected_dataset_ids(args):
            prepare_one(dataset_id, args.force)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
