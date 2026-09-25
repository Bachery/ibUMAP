#!/usr/bin/env python3
"""Download and verify the public BRAQUE supplementary dataset.

The script resolves file metadata from Mendeley Data at run time, downloads via
the repository-provided public URLs, verifies the official size and SHA-256,
and writes an auditable source manifest.  It is a dry run unless --run is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CASE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = CASE_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
LOG_ROOT = CASE_ROOT / "logs"
DATASET_ID = "j8xbwb93x9"
DATASET_VERSION = 1
DATASET_DOI = "10.17632/j8xbwb93x9.1"
METADATA_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
DEFAULT_FILES = ("BRAQUE-RawCSVdata.zip", "ListOfPrimaryAntibodies.xlsx")
USER_AGENT = "ibUMAP-BRAQUE-case/1.0 (+research reproducibility)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Exact repository filename to download; may be repeated.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download every file in the dataset, including the CyBorgh MTA template.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Resolve and save metadata without downloading files (requires --run).",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing files.")
    parser.add_argument(
        "--output-dir", type=Path, default=RAW_ROOT, help="Raw download directory."
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--run", action="store_true", help="Perform writes; otherwise print a dry run."
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def request_json(url: str, timeout: int) -> tuple[dict[str, Any], dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
        headers = {key.lower(): value for key, value in response.headers.items()}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object from {url}, received {type(payload).__name__}")
    return payload, headers


def safe_filename(value: str) -> str:
    name = Path(value).name
    if not name or name != value or name in {".", ".."}:
        raise ValueError(f"Unsafe repository filename: {value!r}")
    return name


def repository_files(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in metadata.get("files", []):
        if not isinstance(entry, dict) or "filename" not in entry:
            continue
        name = safe_filename(str(entry["filename"]))
        if name in result:
            raise ValueError(f"Duplicate repository filename in metadata: {name}")
        result[name] = entry
    if not result:
        raise ValueError("Mendeley metadata contains no downloadable files")
    return result


def expected_file_fields(entry: dict[str, Any]) -> tuple[str, int, str]:
    details = entry.get("content_details") or {}
    url = str(details.get("download_url") or "")
    expected_size = int(details.get("size") or entry.get("size") or 0)
    expected_sha256 = str(details.get("sha256_hash") or "").lower()
    if not url.startswith("https://"):
        raise ValueError(f"Missing HTTPS download URL for {entry.get('filename')}")
    if expected_size <= 0:
        raise ValueError(f"Invalid official size for {entry.get('filename')}")
    if len(expected_sha256) != 64:
        raise ValueError(f"Invalid official SHA-256 for {entry.get('filename')}")
    return url, expected_size, expected_sha256


def verify_existing(path: Path, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if size != expected_size or actual_sha256 != expected_sha256:
        raise ValueError(
            f"Existing file failed verification: {path} "
            f"(size={size}, sha256={actual_sha256})"
        )
    return {
        "status": "existing_verified",
        "path": str(path),
        "size_bytes": size,
        "sha256": actual_sha256,
    }


def _open_download(url: str, timeout: int, start: int = 0):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def download_verified(
    entry: dict[str, Any], destination: Path, *, force: bool, timeout: int
) -> dict[str, Any]:
    url, expected_size, expected_sha256 = expected_file_fields(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return verify_existing(destination, expected_size, expected_sha256)

    part = destination.with_name(f".{destination.name}.part")
    if force:
        part.unlink(missing_ok=True)
    existing = part.stat().st_size if part.exists() else 0
    if existing > expected_size:
        part.unlink()
        existing = 0
    if existing == expected_size:
        actual_sha256 = sha256_file(part)
        if actual_sha256 == expected_sha256:
            os.replace(part, destination)
            return verify_existing(destination, expected_size, expected_sha256)
        part.unlink()
        existing = 0

    response = _open_download(url, timeout, start=existing)
    try:
        status = int(getattr(response, "status", response.getcode()))
        if existing and status != 206:
            response.close()
            part.unlink(missing_ok=True)
            existing = 0
            response = _open_download(url, timeout, start=0)
            status = int(getattr(response, "status", response.getcode()))
        mode = "ab" if existing else "wb"
        with part.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        resolved_url = response.geturl()
        response_headers = {
            key.lower(): value for key, value in response.headers.items()
        }
    finally:
        response.close()

    actual_size = part.stat().st_size
    actual_sha256 = sha256_file(part)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise ValueError(
            f"Downloaded file failed verification for {destination.name}: "
            f"size {actual_size} != {expected_size} or sha256 "
            f"{actual_sha256} != {expected_sha256}. Partial file retained at {part}."
        )
    os.replace(part, destination)
    return {
        "status": "downloaded",
        "path": str(destination),
        "requested_url": url,
        "resolved_url": resolved_url,
        "http_status": status,
        "headers": response_headers,
        "resumed_from_bytes": existing,
        "size_bytes": actual_size,
        "sha256": actual_sha256,
    }


def select_names(
    available: dict[str, dict[str, Any]], requested: Iterable[str] | None, all_files: bool
) -> list[str]:
    raw_names = sorted(available) if all_files else list(requested or DEFAULT_FILES)
    names = list(dict.fromkeys(raw_names))
    missing = sorted(set(names) - set(available))
    if missing:
        raise KeyError(
            f"Requested file(s) not present in dataset version {DATASET_VERSION}: {missing}; "
            f"available={sorted(available)}"
        )
    return names


def manifest_base(metadata: dict[str, Any], metadata_headers: dict[str, str]) -> dict[str, Any]:
    licence = metadata.get("data_licence") or metadata.get("licence") or {}
    return {
        "created_at": now_utc(),
        "dataset": {
            "id": metadata.get("id"),
            "version": metadata.get("version"),
            "doi": DATASET_DOI,
            "name": metadata.get("name"),
            "publish_date": metadata.get("publish_date"),
            "modified_on": metadata.get("modified_on"),
            "license": {
                "name": licence.get("short_name") or licence.get("full_name"),
                "url": licence.get("url"),
            },
            "landing_page": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
            "metadata_url": METADATA_URL,
            "metadata_response_headers": metadata_headers,
        },
        "files": [],
    }


def main() -> int:
    args = parse_args()
    metadata, metadata_headers = request_json(METADATA_URL, args.timeout)
    if str(metadata.get("id")) != DATASET_ID or int(metadata.get("version", -1)) != DATASET_VERSION:
        raise ValueError(
            f"Resolved dataset identity/version mismatch: "
            f"{metadata.get('id')} version {metadata.get('version')}"
        )
    available = repository_files(metadata)
    selected = select_names(available, args.files, args.all_files)

    print(f"Dataset: {metadata.get('name')}")
    print(f"DOI: {DATASET_DOI}")
    print(f"License: {(metadata.get('data_licence') or {}).get('short_name')}")
    print(f"Output: {args.output_dir}")
    for name in selected:
        _, size, checksum = expected_file_fields(available[name])
        print(f"- {name}: {size} bytes, sha256={checksum}")

    if not args.run:
        print("Dry run only; pass --run to write metadata and download files.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = manifest_base(metadata, metadata_headers)
    manifest_path = args.output_dir / "source_manifest.json"
    if args.metadata_only:
        manifest["available_files"] = [
            {
                "filename": name,
                "id": entry.get("id"),
                "description": entry.get("description"),
                "size": (entry.get("content_details") or {}).get("size"),
                "sha256": (entry.get("content_details") or {}).get("sha256_hash"),
            }
            for name, entry in sorted(available.items())
        ]
        atomic_write_json(manifest_path, manifest)
        print(f"Metadata manifest: {manifest_path}")
        return 0

    failures = 0
    for name in selected:
        entry = available[name]
        _, official_size, official_sha256 = expected_file_fields(entry)
        record: dict[str, Any] = {
            "filename": name,
            "repository_file_id": entry.get("id"),
            "description": entry.get("description"),
            "official_size_bytes": official_size,
            "official_sha256": official_sha256,
            "accessed_at": now_utc(),
        }
        try:
            record["download"] = download_verified(
                entry,
                args.output_dir / name,
                force=args.force,
                timeout=args.timeout,
            )
        except Exception as exc:  # preserve an audit record before returning failure
            failures += 1
            record["download"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        manifest["files"].append(record)
        atomic_write_json(manifest_path, manifest)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_write_json(LOG_ROOT / f"01_download_data_{timestamp}.json", manifest)
    if failures:
        print(f"Download completed with {failures} failure(s); see {manifest_path}", file=sys.stderr)
        return 1
    print(f"Verified source manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
