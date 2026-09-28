from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import write_checksums
from common.metadata import build_obs_schema, obs_metadata_fields, utc_now_iso, write_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]

DATASETS: dict[str, dict[str, Any]] = {
    "gist_960_euclidean": {
        "display_name": "GIST 960 Euclidean",
        "raw_path": REPO_ROOT / "datasets/raw/gist_960_euclidean/gist-960-euclidean.hdf5",
        "download_url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "output_dir": REPO_ROOT / "datasets/processed/gist_960_euclidean",
        "metric": "euclidean",
    },
}


HDF5_KEY_CANDIDATES = {
    "features.npy": ("train", "base"),
    "queries.npy": ("test", "query", "queries"),
    "neighbors.npy": ("neighbors",),
    "distances.npy": ("distances",),
}

OUTPUT_DTYPES = {
    "features.npy": np.float32,
    "queries.npy": np.float32,
    "neighbors.npy": np.int32,
    "distances.npy": np.float32,
}
OBS_SCHEMA = build_obs_schema(
    {
        "sample_id": {
            "dtype": "int64",
            "role": "identifier",
            "description": "Stable row identifier matching the row index of features.npy.",
        },
    }
)
SOURCE_KEYS = {
    "features.npy": "train",
    "queries.npy": "test",
    "neighbors.npy": "neighbors",
    "distances.npy": "distances",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ANN Benchmark HDF5 datasets.")
    parser.add_argument("dataset_id", nargs="?", choices=sorted(DATASETS))
    parser.add_argument("--all", action="store_true", help="Prepare all supported datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite known processed outputs.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only refresh metadata.json and checksums.txt from existing processed files.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=8192,
        help="Row chunk size used when copying HDF5 arrays.",
    )
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
    known_outputs = [
        "features.npy",
        "queries.npy",
        "neighbors.npy",
        "distances.npy",
        "obs.csv.gz",
        "metadata.json",
        "checksums.txt",
    ]
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --force to overwrite known processed files."
            )
        for name in known_outputs:
            (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def find_hdf5_key(handle: h5py.File, output_name: str) -> str:
    for key in HDF5_KEY_CANDIDATES[output_name]:
        if key in handle:
            return key
    candidates = ", ".join(HDF5_KEY_CANDIDATES[output_name])
    raise KeyError(f"None of the expected HDF5 keys exist for {output_name}: {candidates}")


def copy_hdf5_to_npy(
    source: h5py.Dataset,
    destination: Path,
    dtype: np.dtype[Any],
    chunk_rows: int,
) -> tuple[list[int], str]:
    shape = [int(value) for value in source.shape]
    target_dtype = np.dtype(dtype)
    array = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=target_dtype,
        shape=tuple(shape),
    )

    if not shape:
        array[...] = source[...]
        array.flush()
        return shape, str(target_dtype)

    rows = shape[0]
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        array[start:stop] = source[start:stop]
        if start == 0 or stop == rows or stop % (chunk_rows * 128) == 0:
            print(f"  wrote {destination.name}: {stop}/{rows} rows", flush=True)

    array.flush()
    return shape, str(target_dtype)


def write_obs(path: Path, n_rows: int) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        for sample_id in range(n_rows):
            writer.writerow({"sample_id": sample_id})
            if sample_id == 0 or sample_id + 1 == n_rows or (sample_id + 1) % 1_000_000 == 0:
                print(f"  wrote obs.csv.gz: {sample_id + 1}/{n_rows} rows", flush=True)


def array_record(path: Path, source_key: str) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    return {
        "file": path.name,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "source_key": source_key,
    }


def metadata_payload(
    dataset_id: str,
    raw_path: Path,
    copied: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    config = DATASETS[dataset_id]
    metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "display_name": config["display_name"],
        "kind": "ann_benchmark",
        "raw_path": repo_relative(raw_path),
        "download_url": config["download_url"],
        "source": "ann-benchmarks",
        "primary_array": "features.npy",
        "obs_file": "obs.csv.gz",
        "extra_arrays": {
            "queries.npy": copied["queries.npy"],
            "neighbors.npy": copied["neighbors.npy"],
            "distances.npy": copied["distances.npy"],
        },
        "feature_shape": copied["features.npy"]["shape"],
        "feature_dtype": copied["features.npy"]["dtype"],
        "query_shape": copied["queries.npy"]["shape"],
        "neighbor_shape": copied["neighbors.npy"]["shape"],
        "distance_shape": copied["distances.npy"]["shape"],
        "metric": config["metric"],
        "generated_by": repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": 1,
        "preprocessing": (
            "Converted ANN Benchmarks HDF5 arrays to standardized NumPy files: "
            "features.npy from train/base, queries.npy from test/query, "
            "neighbors.npy from neighbors, distances.npy from distances, plus "
            "a gzip-compressed observation table with one sample_id per feature row."
        ),
    }
    metadata.update(
        obs_metadata_fields(
            obs_schema=OBS_SCHEMA,
            identifier_column="sample_id",
            default_color_by=None,
            label_columns=[],
            colorable_columns=[],
            searchable_columns=[],
            filterable_columns=[],
        )
    )
    return metadata


def write_metadata_and_checksums(
    dataset_id: str,
    raw_path: Path,
    output_dir: Path,
    copied: dict[str, dict[str, Any]],
) -> None:
    metadata = metadata_payload(dataset_id, raw_path, copied)
    write_metadata(output_dir / "metadata.json", metadata)
    write_checksums(
        output_dir / "checksums.txt",
        [
            output_dir / "features.npy",
            output_dir / "queries.npy",
            output_dir / "neighbors.npy",
            output_dir / "distances.npy",
            output_dir / "obs.csv.gz",
            output_dir / "metadata.json",
        ],
        base_dir=output_dir,
    )


def refresh_metadata_and_checksums(dataset_id: str) -> None:
    config = DATASETS[dataset_id]
    raw_path = Path(config["raw_path"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw HDF5 file not found: {raw_path}")

    required = ["features.npy", "queries.npy", "neighbors.npy", "distances.npy", "obs.csv.gz"]
    for file_name in required:
        path = output_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Required processed file not found: {path}")

    copied = {
        file_name: array_record(output_dir / file_name, SOURCE_KEYS[file_name])
        for file_name in ("features.npy", "queries.npy", "neighbors.npy", "distances.npy")
    }
    write_metadata_and_checksums(dataset_id, raw_path, output_dir, copied)
    print(f"[{dataset_id}] Refreshed metadata and checksums: {output_dir}")


def prepare_one(dataset_id: str, force: bool, chunk_rows: int) -> None:
    config = DATASETS[dataset_id]
    raw_path = Path(config["raw_path"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw HDF5 file not found: {raw_path}")

    print(f"[{dataset_id}] Reading raw HDF5: {raw_path}")
    prepare_output_dir(output_dir, force)

    copied: dict[str, dict[str, Any]] = {}
    with h5py.File(raw_path, "r") as handle:
        for output_name in ("features.npy", "queries.npy", "neighbors.npy", "distances.npy"):
            source_key = find_hdf5_key(handle, output_name)
            shape, dtype = copy_hdf5_to_npy(
                handle[source_key],
                output_dir / output_name,
                OUTPUT_DTYPES[output_name],
                chunk_rows,
            )
            copied[output_name] = {
                "file": output_name,
                "shape": shape,
                "dtype": dtype,
                "source_key": source_key,
            }

    n_features = int(copied["features.npy"]["shape"][0])
    write_obs(output_dir / "obs.csv.gz", n_features)

    write_metadata_and_checksums(dataset_id, raw_path, output_dir, copied)
    print(f"[{dataset_id}] Prepared dataset: {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        for dataset_id in selected_dataset_ids(args):
            if args.metadata_only:
                refresh_metadata_and_checksums(dataset_id)
            else:
                prepare_one(dataset_id, args.force, args.chunk_rows)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
