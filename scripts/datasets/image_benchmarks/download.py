from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CACHE_ROOT = REPO_ROOT / "datasets/raw_cache/image_benchmarks"
CHUNK_SIZE = 1024 * 1024 * 8


@dataclass(frozen=True)
class DownloadFile:
    url: str
    file_name: str


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    raw_subdir: str
    files: tuple[DownloadFile, ...]


MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

DATASETS: dict[str, DatasetConfig] = {
    "mnist784": DatasetConfig(
        dataset_id="mnist784",
        raw_subdir="mnist",
        files=tuple(
            DownloadFile(
                url=f"https://storage.googleapis.com/cvdf-datasets/mnist/{file_name}",
                file_name=file_name,
            )
            for file_name in MNIST_FILES
        ),
    ),
    "fashion_mnist": DatasetConfig(
        dataset_id="fashion_mnist",
        raw_subdir="fashion_mnist",
        files=tuple(
            DownloadFile(
                url=f"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/{file_name}",
                file_name=file_name,
            )
            for file_name in MNIST_FILES
        ),
    ),
    "cifar10": DatasetConfig(
        dataset_id="cifar10",
        raw_subdir="cifar10",
        files=(
            DownloadFile(
                url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                file_name="cifar-10-python.tar.gz",
            ),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw cache files for image benchmark datasets."
    )
    parser.add_argument("dataset_id", nargs="?", choices=sorted(DATASETS))
    parser.add_argument("--all", action="store_true", help="Download all supported datasets.")
    parser.add_argument("--force", action="store_true", help="Redownload files that already exist.")
    return parser.parse_args()


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(DATASETS)
    if args.dataset_id:
        return [args.dataset_id]
    raise ValueError("Provide a dataset_id or --all")


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown size"
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GiB"


def download_one_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"  skip existing: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    print(f"  download: {url}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        next_report = CHUNK_SIZE * 4
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report or (total is not None and downloaded == total):
                print(
                    f"    wrote {_format_bytes(downloaded)} / {_format_bytes(total)}",
                    flush=True,
                )
                next_report += CHUNK_SIZE * 4

    tmp_path.replace(destination)
    print(f"  saved: {destination} ({_format_bytes(destination.stat().st_size)})")


def download_dataset(dataset_id: str, force: bool) -> None:
    config = DATASETS[dataset_id]
    output_dir = RAW_CACHE_ROOT / config.raw_subdir
    print(f"[{dataset_id}] raw cache: {output_dir}")
    for file_config in config.files:
        download_one_file(file_config.url, output_dir / file_config.file_name, force)


def main() -> int:
    args = parse_args()
    try:
        for dataset_id in selected_dataset_ids(args):
            download_dataset(dataset_id, args.force)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
