from __future__ import annotations

import argparse
import csv
import gzip
import pickle
import shutil
import struct
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import write_checksums
from common.metadata import build_obs_schema, obs_metadata_fields, utc_now_iso, write_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CACHE_ROOT = REPO_ROOT / "datasets/raw_cache/image_benchmarks"
PROCESSED_ROOT = REPO_ROOT / "datasets/processed"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    display_name: str
    raw_subdir: str
    feature_shape: tuple[int, int]
    label_names: tuple[str, ...]
    parser: str
    raw_files: tuple[str, ...]
    download_urls: tuple[str, ...]


IDX_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

FASHION_LABEL_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

CIFAR10_LABEL_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

DATASETS: dict[str, DatasetConfig] = {
    "mnist784": DatasetConfig(
        dataset_id="mnist784",
        display_name="MNIST 784",
        raw_subdir="mnist",
        feature_shape=(70_000, 784),
        label_names=tuple(str(value) for value in range(10)),
        parser="idx",
        raw_files=IDX_FILES,
        download_urls=tuple(
            f"https://storage.googleapis.com/cvdf-datasets/mnist/{file_name}"
            for file_name in IDX_FILES
        ),
    ),
    "fashion_mnist": DatasetConfig(
        dataset_id="fashion_mnist",
        display_name="Fashion-MNIST",
        raw_subdir="fashion_mnist",
        feature_shape=(70_000, 784),
        label_names=FASHION_LABEL_NAMES,
        parser="idx",
        raw_files=IDX_FILES,
        download_urls=tuple(
            f"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{file_name}"
            for file_name in IDX_FILES
        ),
    ),
    "cifar10": DatasetConfig(
        dataset_id="cifar10",
        display_name="CIFAR-10",
        raw_subdir="cifar10",
        feature_shape=(60_000, 3072),
        label_names=CIFAR10_LABEL_NAMES,
        parser="cifar10",
        raw_files=("cifar-10-python.tar.gz",),
        download_urls=("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",),
    ),
}

OBS_SCHEMA = build_obs_schema(
    {
        "sample_id": {
            "dtype": "int64",
            "role": "identifier",
            "description": "Stable row identifier matching the row index of features.npy.",
        },
        "split": {
            "dtype": "string",
            "role": "split",
            "description": "Dataset split: train or test.",
        },
        "label": {
            "dtype": "int32",
            "role": "class_label",
            "description": "Integer class label from the original benchmark dataset.",
        },
        "label_name": {
            "dtype": "string",
            "role": "class_name",
            "description": "Human-readable class name for the integer label.",
        },
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare image benchmark datasets from downloaded raw cache files."
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


def require_raw_cache(config: DatasetConfig) -> Path:
    raw_dir = RAW_CACHE_ROOT / config.raw_subdir
    missing = [file_name for file_name in config.raw_files if not (raw_dir / file_name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Raw cache is incomplete for {config.dataset_id}: {missing_text}. "
            f"Run scripts/datasets/image_benchmarks/download.py {config.dataset_id} first."
        )
    return raw_dir


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to overwrite it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image magic for {path}: {magic}")
        raw = handle.read()
    expected = n_images * n_rows * n_cols
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size != expected:
        raise ValueError(f"IDX image size mismatch for {path}: expected {expected}, got {data.size}")
    return data.reshape(n_images, n_rows * n_cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, n_labels = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label magic for {path}: {magic}")
        raw = handle.read()
    labels = np.frombuffer(raw, dtype=np.uint8)
    if labels.size != n_labels:
        raise ValueError(f"IDX label size mismatch for {path}: expected {n_labels}, got {labels.size}")
    return labels.astype(np.int32)


def write_obs_rows(
    writer: csv.DictWriter[str],
    start_sample_id: int,
    split: str,
    labels: np.ndarray,
    label_names: tuple[str, ...],
) -> None:
    for offset, label_value in enumerate(labels):
        label = int(label_value)
        writer.writerow(
            {
                "sample_id": start_sample_id + offset,
                "split": split,
                "label": label,
                "label_name": label_names[label],
            }
        )


def prepare_idx_dataset(
    config: DatasetConfig,
    raw_dir: Path,
    features: np.memmap[Any, Any],
    target: np.memmap[Any, Any],
    obs_writer: csv.DictWriter[str],
) -> None:
    splits = [
        ("train", raw_dir / "train-images-idx3-ubyte.gz", raw_dir / "train-labels-idx1-ubyte.gz"),
        ("test", raw_dir / "t10k-images-idx3-ubyte.gz", raw_dir / "t10k-labels-idx1-ubyte.gz"),
    ]

    row_start = 0
    for split, image_path, label_path in splits:
        images = read_idx_images(image_path)
        labels = read_idx_labels(label_path)
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{config.dataset_id} {split} row mismatch: "
                f"images={images.shape[0]}, labels={labels.shape[0]}"
            )
        row_stop = row_start + int(images.shape[0])
        features[row_start:row_stop] = images.astype(np.float32) / np.float32(255.0)
        target[row_start:row_stop] = labels
        write_obs_rows(obs_writer, row_start, split, labels, config.label_names)
        print(f"  wrote {config.dataset_id} {split}: {images.shape[0]} rows", flush=True)
        row_start = row_stop

    if row_start != config.feature_shape[0]:
        raise ValueError(
            f"{config.dataset_id} row count mismatch: expected {config.feature_shape[0]}, got {row_start}"
        )


def load_cifar_member(handle: tarfile.TarFile, member_name: str) -> dict[str, Any]:
    member = handle.getmember(f"cifar-10-batches-py/{member_name}")
    extracted = handle.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract CIFAR-10 member: {member_name}")
    with extracted:
        return pickle.load(extracted, encoding="latin1")


def cifar_data_to_row_major(data: np.ndarray) -> np.ndarray:
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images.reshape(data.shape[0], 32 * 32 * 3)


def prepare_cifar10_dataset(
    config: DatasetConfig,
    raw_dir: Path,
    features: np.memmap[Any, Any],
    target: np.memmap[Any, Any],
    obs_writer: csv.DictWriter[str],
) -> None:
    tar_path = raw_dir / "cifar-10-python.tar.gz"
    row_start = 0
    with tarfile.open(tar_path, "r:gz") as handle:
        meta = load_cifar_member(handle, "batches.meta")
        label_names = tuple(str(value) for value in meta.get("label_names", config.label_names))
        if label_names != config.label_names:
            print("  warning: CIFAR-10 label names differ from local constants", file=sys.stderr)

        for batch_name in [f"data_batch_{index}" for index in range(1, 6)]:
            batch = load_cifar_member(handle, batch_name)
            data = np.asarray(batch["data"], dtype=np.uint8)
            labels = np.asarray(batch["labels"], dtype=np.int32)
            row_stop = row_start + int(data.shape[0])
            features[row_start:row_stop] = cifar_data_to_row_major(data).astype(np.float32) / np.float32(255.0)
            target[row_start:row_stop] = labels
            write_obs_rows(obs_writer, row_start, "train", labels, config.label_names)
            print(f"  wrote {config.dataset_id} {batch_name}: {data.shape[0]} rows", flush=True)
            row_start = row_stop

        batch = load_cifar_member(handle, "test_batch")
        data = np.asarray(batch["data"], dtype=np.uint8)
        labels = np.asarray(batch["labels"], dtype=np.int32)
        row_stop = row_start + int(data.shape[0])
        features[row_start:row_stop] = cifar_data_to_row_major(data).astype(np.float32) / np.float32(255.0)
        target[row_start:row_stop] = labels
        write_obs_rows(obs_writer, row_start, "test", labels, config.label_names)
        print(f"  wrote {config.dataset_id} test_batch: {data.shape[0]} rows", flush=True)
        row_start = row_stop

    if row_start != config.feature_shape[0]:
        raise ValueError(
            f"{config.dataset_id} row count mismatch: expected {config.feature_shape[0]}, got {row_start}"
        )


def metadata_payload(config: DatasetConfig, raw_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_id": config.dataset_id,
        "display_name": config.display_name,
        "kind": "image_benchmark",
        "source_type": "direct_download",
        "download_urls": list(config.download_urls),
        "raw_role": "download_cache",
        "raw_path": repo_relative(raw_dir),
        "download_url": config.download_urls[0] if len(config.download_urls) == 1 else None,
        "primary_array": "features.npy",
        "feature_shape": [int(value) for value in config.feature_shape],
        "feature_dtype": "float32",
        "obs_file": "obs.csv.gz",
        "target_file": "target.npy",
        "target_dtype": "int32",
        "target_source_column": "label",
        "identifier_column": "sample_id",
        "metric": "euclidean",
        "generated_by": repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": 1,
        "preprocessing": (
            "Downloaded original benchmark files, parsed IDX/CIFAR formats, flattened images "
            "into float32 feature vectors normalized to [0,1], and stored labels in obs.csv.gz "
            "with int32 target.npy cache."
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
            filterable_columns=["split", "label", "label_name"],
        )
    )
    return payload


def prepare_one(dataset_id: str, force: bool) -> None:
    config = DATASETS[dataset_id]
    raw_dir = require_raw_cache(config)
    output_dir = PROCESSED_ROOT / dataset_id
    prepare_output_dir(output_dir, force)

    features_path = output_dir / "features.npy"
    obs_path = output_dir / "obs.csv.gz"
    target_path = output_dir / "target.npy"
    metadata_path = output_dir / "metadata.json"

    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=config.feature_shape,
    )
    target = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.int32,
        shape=(config.feature_shape[0],),
    )

    with gzip.open(obs_path, "wt", newline="", encoding="utf-8") as obs_file:
        writer = csv.DictWriter(
            obs_file,
            fieldnames=["sample_id", "split", "label", "label_name"],
        )
        writer.writeheader()
        if config.parser == "idx":
            prepare_idx_dataset(config, raw_dir, features, target, writer)
        elif config.parser == "cifar10":
            prepare_cifar10_dataset(config, raw_dir, features, target, writer)
        else:
            raise ValueError(f"Unsupported parser for {dataset_id}: {config.parser}")

    features.flush()
    target.flush()

    metadata = metadata_payload(config, raw_dir)
    write_metadata(metadata_path, metadata)
    write_checksums(
        output_dir / "checksums.txt",
        [features_path, obs_path, target_path, metadata_path],
        base_dir=output_dir,
    )
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
