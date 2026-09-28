from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
from dataclasses import dataclass


REPO_ROOT = Path(__file__).resolve().parents[3]
ZENODO_RECORD_ID = "7216361"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
RAW_ROOT = REPO_ROOT / "datasets/raw/scdeed"
DOWNLOAD_ROOT = RAW_ROOT / "downloads"
METADATA_PATH = RAW_ROOT / "zenodo_metadata.json"
CHUNK_SIZE = 1024 * 1024 * 8


@dataclass(frozen=True)
class ZenodoFile:
    file_name: str
    size: int | None
    checksum: str | None
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download scDEED files from Zenodo record 7216361.")
    parser.add_argument("--force", action="store_true", help="Redownload files that already exist.")
    parser.add_argument("--list-only", action="store_true", help="List files without downloading them.")
    parser.add_argument("--limit", type=int, help="Only process the first N files.")
    return parser.parse_args()


def fetch_record_metadata() -> dict[str, Any]:
    request = urllib.request.Request(ZENODO_API, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def write_record_metadata(metadata: dict[str, Any]) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_files(metadata: dict[str, Any]) -> list[ZenodoFile]:
    records: list[ZenodoFile] = []
    for item in metadata.get("files", []):
        links = item.get("links", {})
        url = links.get("self") or links.get("download")
        key = item.get("key")
        if not key or not url:
            continue
        records.append(
            ZenodoFile(
                file_name=str(key),
                size=int(item["size"]) if item.get("size") is not None else None,
                checksum=str(item.get("checksum")) if item.get("checksum") else None,
                url=str(url),
            )
        )
    return records


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


def checksum_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, checksum: str | None) -> None:
    if not checksum:
        return
    if ":" in checksum:
        algorithm, expected = checksum.split(":", 1)
    else:
        algorithm, expected = "md5", checksum
    actual = checksum_file(path, algorithm)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {checksum}, got {algorithm}:{actual}"
        )


def download_file(record: ZenodoFile, force: bool) -> None:
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    destination = DOWNLOAD_ROOT / record.file_name
    if destination.exists() and not force:
        print(f"skip existing: {destination}")
        verify_checksum(destination, record.checksum)
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    print(f"download {record.file_name}: {record.url}")
    with urllib.request.urlopen(record.url) as response, tmp_path.open("wb") as handle:
        downloaded = 0
        next_report = CHUNK_SIZE * 8
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report or (
                record.size is not None and downloaded == record.size
            ):
                print(
                    f"  wrote {format_bytes(downloaded)} / {format_bytes(record.size)}",
                    flush=True,
                )
                next_report += CHUNK_SIZE * 8

    tmp_path.replace(destination)
    verify_checksum(destination, record.checksum)
    print(f"saved: {destination} ({format_bytes(destination.stat().st_size)})")


def main() -> int:
    args = parse_args()
    try:
        metadata = fetch_record_metadata()
        write_record_metadata(metadata)
        files = parse_files(metadata)
        if args.limit is not None:
            files = files[: args.limit]

        if args.list_only:
            for record in files:
                print(
                    "\t".join(
                        [
                            record.file_name,
                            format_bytes(record.size),
                            record.checksum or "",
                            record.url,
                        ]
                    )
                )
            print(f"Found {len(files)} file(s)")
            return 0

        for record in files:
            download_file(record, args.force)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
