from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


DATASET_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPT_ROOT))

from common.metadata import utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "datasets/raw/c_elegans_embryogenesis"
RAW_PATH = RAW_ROOT / "packer2019.h5ad"
SOURCE_METADATA_PATH = RAW_ROOT / "source_metadata.json"
DOWNLOAD_PAGE = "https://data.caltech.edu/records/b1kj4-nh475"
SOURCE_URL = "https://data.caltech.edu/api/records/b1kj4-nh475/files/packer2019.h5ad/content"
SOURCE_NAME = "Packer 2019 C. elegans embryogenesis"
FILE_NAME = "packer2019.h5ad"
EXPECTED_MD5 = "e2cda8a6cee91d1a326dea5b9e4d2539"
EXPECTED_SIZE_BYTES = 682_250_280
CHUNK_SIZE = 1024 * 1024 * 8
REQUEST_HEADERS = {
    "Accept": "application/octet-stream,*/*",
    "User-Agent": "curl/8.7.1",
}
MAX_ATTEMPTS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Packer 2019 C. elegans embryogenesis H5AD "
            "from CaltechDATA into datasets/raw."
        )
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing raw target.")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print source and target paths without downloading.",
    )
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GiB"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path) -> None:
    actual = md5_file(path)
    if actual.lower() != EXPECTED_MD5.lower():
        raise ValueError(f"MD5 mismatch for {path}: expected {EXPECTED_MD5}, got {actual}")


def verify_size(path: Path) -> None:
    actual = Path(path).stat().st_size
    if actual != EXPECTED_SIZE_BYTES:
        raise ValueError(
            f"Size mismatch for {path}: expected {EXPECTED_SIZE_BYTES}, got {actual}"
        )


def write_source_metadata(
    *,
    download_status: str,
    source_url: str = SOURCE_URL,
) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source_name": SOURCE_NAME,
        "file_name": FILE_NAME,
        "source_url": source_url,
        "download_page": DOWNLOAD_PAGE,
        "target_path": repo_relative(RAW_PATH),
        "download_status": download_status,
        "expected_md5": EXPECTED_MD5,
        "expected_size_bytes": EXPECTED_SIZE_BYTES,
        "generated_at": utc_now_iso(),
    }
    SOURCE_METADATA_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_partial_path(tmp_path: Path, force: bool) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if not (RAW_PATH.exists() or RAW_PATH.is_symlink()):
        return

    if not force:
        try:
            verify_size(RAW_PATH)
            verify_md5(RAW_PATH)
        except ValueError as exc:
            actual_size = RAW_PATH.stat().st_size
            if actual_size < EXPECTED_SIZE_BYTES:
                if tmp_path.exists() and tmp_path.stat().st_size >= actual_size:
                    RAW_PATH.unlink()
                else:
                    RAW_PATH.replace(tmp_path)
                print(f"Found incomplete target ({format_bytes(actual_size)}); resuming download.")
                return
            raise exc
        else:
            write_source_metadata(download_status="already_present")
            print(f"Raw H5AD already exists: {RAW_PATH}")
            return

    RAW_PATH.unlink()
    tmp_path.unlink(missing_ok=True)


def make_request(offset: int) -> urllib.request.Request:
    headers = dict(REQUEST_HEADERS)
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(SOURCE_URL, headers=headers)


def stream_attempt(tmp_path: Path) -> int:
    offset = tmp_path.stat().st_size if tmp_path.exists() else 0
    if offset > EXPECTED_SIZE_BYTES:
        raise ValueError(f"Partial file is larger than expected: {tmp_path}")

    request = make_request(offset)
    with urllib.request.urlopen(request) as response:
        status = int(getattr(response, "status", response.getcode()))
        append = offset > 0 and status == 206
        if offset > 0 and not append:
            print("  server did not honor Range request; restarting from byte 0")
            offset = 0
        mode = "ab" if append else "wb"
        with tmp_path.open(mode) as handle:
            downloaded = offset
            next_report = ((downloaded // (CHUNK_SIZE * 8)) + 1) * CHUNK_SIZE * 8
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  wrote {format_bytes(downloaded)}", flush=True)
                    next_report += CHUNK_SIZE * 8
    return tmp_path.stat().st_size


def download(force: bool) -> None:
    tmp_path = RAW_PATH.with_suffix(RAW_PATH.suffix + ".part")
    prepare_partial_path(tmp_path, force)

    if RAW_PATH.exists():
        verify_md5(RAW_PATH)
        write_source_metadata(download_status="already_present")
        print(f"Raw H5AD already exists: {RAW_PATH}")
        return

    print(f"Downloading {SOURCE_URL}")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        current_size = tmp_path.stat().st_size if tmp_path.exists() else 0
        print(f"attempt {attempt}/{MAX_ATTEMPTS}: starting at {format_bytes(current_size)}")
        new_size = stream_attempt(tmp_path)
        if new_size == EXPECTED_SIZE_BYTES:
            break
        if new_size > EXPECTED_SIZE_BYTES:
            raise ValueError(
                f"Downloaded file is larger than expected: {new_size} > {EXPECTED_SIZE_BYTES}"
            )
        print(
            f"  partial download is {format_bytes(new_size)} / "
            f"{format_bytes(EXPECTED_SIZE_BYTES)}; retrying"
        )
    else:
        raise RuntimeError(
            f"Download incomplete after {MAX_ATTEMPTS} attempts: "
            f"{format_bytes(tmp_path.stat().st_size if tmp_path.exists() else 0)} / "
            f"{format_bytes(EXPECTED_SIZE_BYTES)}"
        )

    verify_size(tmp_path)
    verify_md5(tmp_path)
    tmp_path.replace(RAW_PATH)
    write_source_metadata(download_status="downloaded")
    print(f"Saved {RAW_PATH} ({format_bytes(RAW_PATH.stat().st_size)})")


def print_listing() -> None:
    print(f"source_name\t{SOURCE_NAME}")
    print(f"download_page\t{DOWNLOAD_PAGE}")
    print(f"source_url\t{SOURCE_URL}")
    print(f"file_name\t{FILE_NAME}")
    print(f"expected_md5\t{EXPECTED_MD5}")
    print(f"expected_size\t{format_bytes(EXPECTED_SIZE_BYTES)}")
    print(f"target_path\t{RAW_PATH}")


def main() -> int:
    args = parse_args()
    try:
        if args.list_only:
            print_listing()
            return 0
        download(args.force)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(
            "Manual download fallback:\n"
            f"  1. Open {DOWNLOAD_PAGE}\n"
            f"  2. Download {FILE_NAME}\n"
            f"  3. Place it at {RAW_PATH}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
