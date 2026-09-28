from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.checksums import write_checksums
from common.metadata import build_obs_schema, obs_metadata_fields, utc_now_iso, write_metadata


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_URL = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
DEFAULT_RAW_PATH = REPO_ROOT / "datasets/raw/google_news/GoogleNews-vectors-negative300.bin.gz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets/processed/google_news_300d"
DATASET_ID = "google_news_300d"
OBS_SCHEMA = build_obs_schema(
    {
        "sample_id": {
            "dtype": "int64",
            "role": "identifier",
            "description": "Stable row identifier matching the row index of features.npy.",
        },
        "token": {
            "dtype": "string",
            "role": "semantic_identifier",
            "description": "Word or phrase token from the original GoogleNews word2vec vocabulary.",
        },
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Google News Word2Vec dataset.")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed files.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only refresh metadata.json and checksums.txt from existing processed files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Print progress every N rows.",
    )
    return parser.parse_args()


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_token(stream: BinaryIO) -> bytes:
    token = bytearray()
    while True:
        chunk = stream.peek(4096)
        if not chunk:
            raise EOFError("Unexpected EOF while reading word2vec token")

        delimiter = chunk.find(b" ")
        if delimiter >= 0:
            if delimiter:
                token.extend(stream.read(delimiter))
            stream.read(1)
            return bytes(token.lstrip(b"\n"))

        token.extend(stream.read(len(chunk)))


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    known_outputs = ["features.npy", "obs.csv.gz", "metadata.json", "checksums.txt"]
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --force to overwrite known processed files."
            )
        for name in known_outputs:
            (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def _array_shape_dtype(path: Path) -> tuple[list[int], str]:
    array = np.load(path, mmap_mode="r")
    return [int(value) for value in array.shape], str(array.dtype)


def _metadata_payload(raw_path: Path, feature_shape: list[int], feature_dtype: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "display_name": "Google News Word2Vec 300D",
        "kind": "word_embedding",
        "raw_path": _repo_relative(raw_path),
        "download_url": DOWNLOAD_URL,
        "primary_array": "features.npy",
        "obs_file": "obs.csv.gz",
        "feature_shape": feature_shape,
        "feature_dtype": feature_dtype,
        "metric": "cosine",
        "generated_by": _repo_relative(Path(__file__)),
        "generated_at": utc_now_iso(),
        "version": 1,
        "preprocessing": (
            "Converted GoogleNews word2vec binary gzip archive to a float32 "
            "NumPy memmap-compatible .npy feature matrix and gzip-compressed "
            "token observation table."
        ),
    }
    payload.update(
        obs_metadata_fields(
            obs_schema=OBS_SCHEMA,
            identifier_column="sample_id",
            default_color_by=None,
            label_columns=[],
            colorable_columns=[],
            searchable_columns=["token"],
            filterable_columns=[],
        )
    )
    return payload


def refresh_metadata_and_checksums(raw_path: Path, output_dir: Path) -> None:
    raw_path = raw_path.resolve()
    output_dir = output_dir.resolve()
    features_path = output_dir / "features.npy"
    obs_path = output_dir / "obs.csv.gz"
    metadata_path = output_dir / "metadata.json"
    checksums_path = output_dir / "checksums.txt"

    for path in (features_path, obs_path):
        if not path.exists():
            raise FileNotFoundError(f"Required processed file not found: {path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw archive not found: {raw_path}")

    feature_shape, feature_dtype = _array_shape_dtype(features_path)
    metadata = _metadata_payload(raw_path, feature_shape, feature_dtype)
    write_metadata(metadata_path, metadata)
    write_checksums(
        checksums_path,
        [features_path, obs_path, metadata_path],
        base_dir=output_dir,
    )
    print(f"Refreshed metadata and checksums: {output_dir}")


def prepare_google_news(raw_path: Path, output_dir: Path, force: bool, progress_every: int) -> None:
    raw_path = raw_path.resolve()
    output_dir = output_dir.resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw archive not found: {raw_path}")

    _prepare_output_dir(output_dir, force)
    features_path = output_dir / "features.npy"
    obs_path = output_dir / "obs.csv.gz"
    metadata_path = output_dir / "metadata.json"
    checksums_path = output_dir / "checksums.txt"

    print(f"Reading raw archive: {raw_path}")
    with gzip.open(raw_path, "rb") as raw_file:
        header = raw_file.readline().decode("utf-8").strip()
        try:
            vocab_size, vector_size = (int(part) for part in header.split())
        except Exception as exc:
            raise ValueError(f"Invalid word2vec header: {header!r}") from exc

        if vector_size != 300:
            print(f"Warning: expected 300 dimensions, found {vector_size}", file=sys.stderr)

        print(f"Header: vocab_size={vocab_size}, vector_size={vector_size}")
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(vocab_size, vector_size),
        )
        vector_bytes = vector_size * np.dtype(np.float32).itemsize

        with gzip.open(obs_path, "wt", newline="", encoding="utf-8") as obs_file:
            writer = csv.DictWriter(obs_file, fieldnames=["sample_id", "token"])
            writer.writeheader()

            for row in range(vocab_size):
                token = _read_token(raw_file).decode("utf-8", errors="replace")
                raw_vector = raw_file.read(vector_bytes)
                if len(raw_vector) != vector_bytes:
                    raise EOFError(
                        f"Unexpected EOF while reading vector row {row} from {raw_path}"
                    )
                features[row] = np.frombuffer(raw_vector, dtype=np.float32, count=vector_size)
                writer.writerow({"sample_id": row, "token": token})

                if row == 0 or row + 1 == vocab_size or (row + 1) % progress_every == 0:
                    print(f"  wrote {row + 1}/{vocab_size} rows", flush=True)

        features.flush()

    metadata = _metadata_payload(raw_path, [vocab_size, vector_size], "float32")
    write_metadata(metadata_path, metadata)
    write_checksums(
        checksums_path,
        [features_path, obs_path, metadata_path],
        base_dir=output_dir,
    )
    print(f"Prepared dataset: {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        if args.metadata_only:
            refresh_metadata_and_checksums(args.raw_path, args.output_dir)
        else:
            prepare_google_news(args.raw_path, args.output_dir, args.force, args.progress_every)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
