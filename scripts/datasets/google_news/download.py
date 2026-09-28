from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_URL = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/raw/google_news/GoogleNews-vectors-negative300.bin.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Google News Word2Vec raw archive.")
    parser.add_argument("--url", default=DOWNLOAD_URL, help="Download URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination raw archive path.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if output exists.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024 * 8,
        help="Streaming download chunk size in bytes.",
    )
    return parser.parse_args()


def download(url: str, output: Path, force: bool, chunk_size: int) -> None:
    output = output.resolve()
    if output.exists() and not force:
        print(f"Raw file already exists, skipping download: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            print(f"Downloading {url}")
            print(f"Destination: {output}")
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
                        print(f"  {downloaded / (1024 ** 2):.1f} MiB / {total / (1024 ** 2):.1f} MiB ({pct:.1f}%)")
                    else:
                        print(f"  {downloaded / (1024 ** 2):.1f} MiB")
            tmp_file.flush()
            tmp_file.close()
            shutil.move(str(tmp_path), output)
            print(f"Download complete: {output}")
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def main() -> int:
    args = parse_args()
    download(args.url, args.output, args.force, args.chunk_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
