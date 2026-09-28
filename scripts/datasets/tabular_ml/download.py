"""Download raw inputs for the tabular ML dataset family.

Files are written to datasets/raw_cache/tabular_ml/<dataset_id>/ and verified
against pinned SHA-256 values (see _sources.py). For each dataset, archives
(e.g. the official UCI zip) are tried first; files still missing afterwards are
fetched from their per-file mirror URLs, in order, until one matches.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from _sources import DATASETS, ArchiveSource, DatasetSource, SourceFile


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CACHE_ROOT = REPO_ROOT / "datasets/raw_cache/tabular_ml"
CHUNK_SIZE = 1024 * 1024 * 8
TIMEOUT_SECONDS = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw files for the tabular ML datasets.")
    parser.add_argument("dataset_id", nargs="?", choices=sorted(DATASETS))
    parser.add_argument("--all", action="store_true", help="Download all supported datasets.")
    parser.add_argument("--force", action="store_true", help="Redownload files that already exist.")
    parser.add_argument("--output-root", type=Path, default=RAW_CACHE_ROOT)
    return parser.parse_args()


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(DATASETS)
    if args.dataset_id:
        return [args.dataset_id]
    raise ValueError("Provide a dataset_id or --all")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def normalize(source: SourceFile, data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n") if source.text else data


def store_verified(source: SourceFile, data: bytes, output_dir: Path, origin: str) -> bool:
    data = normalize(source, data)
    actual = sha256_bytes(data)
    if actual != source.sha256:
        print(f"    {source.file_name}: SHA-256 mismatch from {origin} ({actual})", file=sys.stderr)
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.file_name
    tmp_path = destination.with_name(destination.name + ".part")
    tmp_path.write_bytes(data)
    tmp_path.replace(destination)
    print(f"  ok: {source.file_name} ({len(data)} bytes, sha256 verified) <- {origin}")
    return True


def is_present(source: SourceFile, output_dir: Path) -> bool:
    path = output_dir / source.file_name
    if not path.exists():
        return False
    actual = sha256_file(path)
    if actual != source.sha256:
        raise RuntimeError(
            f"existing file {path} has SHA-256 {actual}, expected {source.sha256}; rerun with --force"
        )
    print(f"  ok (existing): {source.file_name}")
    return True


def try_archive(archive: ArchiveSource, wanted: list[SourceFile], output_dir: Path) -> set[str]:
    done: set[str] = set()
    print(f"  download archive: {archive.url}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "archive.zip"
            zip_path.write_bytes(fetch_bytes(archive.url))
            with zipfile.ZipFile(zip_path) as bundle:
                by_name = {PurePosixPath(name).name: name for name in bundle.namelist()}
                for source in wanted:
                    member = by_name.get(source.file_name)
                    if source.file_name not in archive.members or member is None:
                        continue
                    if store_verified(source, bundle.read(member), output_dir, f"{archive.url}!{member}"):
                        done.add(source.file_name)
    except Exception as exc:
        print(f"    archive unavailable: {exc}", file=sys.stderr)
    return done


def try_mirrors(source: SourceFile, output_dir: Path) -> bool:
    for url in source.urls:
        print(f"  download: {url}")
        try:
            data = fetch_bytes(url)
        except Exception as exc:
            print(f"    failed: {exc}", file=sys.stderr)
            continue
        if store_verified(source, data, output_dir, url):
            return True
    return False


def download_dataset(config: DatasetSource, output_dir: Path, force: bool) -> None:
    missing = [s for s in config.files if force or not is_present(s, output_dir)]
    for archive in config.archives:
        if not missing:
            break
        done = try_archive(archive, missing, output_dir)
        missing = [s for s in missing if s.file_name not in done]
    failed = [s.file_name for s in missing if not try_mirrors(s, output_dir)]
    if failed:
        raise RuntimeError("could not obtain a verified copy of: " + ", ".join(failed))


def main() -> int:
    args = parse_args()
    try:
        dataset_ids = selected_dataset_ids(args)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    failed: list[str] = []
    for dataset_id in dataset_ids:
        output_dir = args.output_root / dataset_id
        print(f"[{dataset_id}] -> {output_dir}")
        try:
            download_dataset(DATASETS[dataset_id], output_dir, args.force)
        except Exception as exc:
            failed.append(dataset_id)
            print(f"[{dataset_id}] FAIL: {exc}", file=sys.stderr)
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
