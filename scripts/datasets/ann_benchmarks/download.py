from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[3]

DATASETS = {
    "gist_960_euclidean": {
        "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "raw_dir": REPO_ROOT / "datasets/raw/gist_960_euclidean",
        "filename": "gist-960-euclidean.hdf5",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ANN Benchmark HDF5 datasets.")
    parser.add_argument("dataset_id", nargs="?", choices=sorted(DATASETS))
    parser.add_argument("--all", action="store_true", help="Download all supported datasets.")
    parser.add_argument("--force", action="store_true", help="Re-download existing raw files.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024 * 8,
        help="Streaming download chunk size in bytes.",
    )
    return parser.parse_args()


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(DATASETS)
    if args.dataset_id:
        return [args.dataset_id]
    raise ValueError("Provide a dataset_id or --all")


def download_one(dataset_id: str, force: bool, chunk_size: int) -> None:
    config = DATASETS[dataset_id]
    url = str(config["url"])
    raw_dir = Path(config["raw_dir"])
    output = raw_dir / str(config["filename"])

    if output.exists() and not force:
        print(f"[{dataset_id}] Raw file already exists, skipping: {output}")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            print(f"[{dataset_id}] Downloading {url}")
            print(f"[{dataset_id}] Destination: {output}")
            with urlopen(url) as response:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header else None
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"[{dataset_id}] {downloaded / (1024 ** 2):.1f} MiB / "
                            f"{total / (1024 ** 2):.1f} MiB ({pct:.1f}%)"
                        )
                    else:
                        print(f"[{dataset_id}] {downloaded / (1024 ** 2):.1f} MiB")
            tmp_file.flush()
            tmp_file.close()
            shutil.move(str(tmp_path), output)
            print(f"[{dataset_id}] Download complete: {output}")
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def main() -> int:
    args = parse_args()
    try:
        for dataset_id in selected_dataset_ids(args):
            download_one(dataset_id, args.force, args.chunk_size)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
