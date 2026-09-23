from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTION_ID = "e5f58829-1a66-40b5-a624-9046778e74f5"
COLLECTION_API = (
    "https://api.cellxgene.cziscience.com/curation/v1/collections/"
    f"{COLLECTION_ID}"
)
RAW_ROOT = REPO_ROOT / "datasets/raw/tabula_sapiens_v2"
H5AD_ROOT = RAW_ROOT / "h5ad"
COLLECTION_METADATA_PATH = RAW_ROOT / "collection_metadata.json"
CHUNK_SIZE = 1024 * 1024 * 8


@dataclass(frozen=True)
class H5ADAsset:
    dataset_title: str
    file_name: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Tabula Sapiens v2 H5AD files from the CELLxGENE collection API."
    )
    parser.add_argument("--force", action="store_true", help="Redownload files that already exist.")
    parser.add_argument("--list-only", action="store_true", help="List H5AD assets without downloading.")
    parser.add_argument("--limit", type=int, help="Only process the first N H5AD assets.")
    return parser.parse_args()


def safe_h5ad_file_name(title: str) -> str:
    name = title.replace("Tabula Sapiens - ", "TS2_")
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        name = "tabula_sapiens_v2_dataset"
    if not name.lower().endswith(".h5ad"):
        name = f"{name}.h5ad"
    return name


def fetch_collection_metadata() -> dict[str, Any]:
    request = urllib.request.Request(COLLECTION_API, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def write_collection_metadata(metadata: dict[str, Any]) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    COLLECTION_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def iter_h5ad_assets(metadata: dict[str, Any]) -> list[H5ADAsset]:
    assets: list[H5ADAsset] = []
    for dataset in metadata.get("datasets", []):
        title = str(dataset.get("title") or dataset.get("name") or dataset.get("id") or "dataset")
        h5ad_asset = next(
            (
                asset
                for asset in dataset.get("assets", [])
                if str(asset.get("filetype", "")).upper() == "H5AD" and asset.get("url")
            ),
            None,
        )
        if h5ad_asset is None:
            continue
        assets.append(
            H5ADAsset(
                dataset_title=title,
                file_name=safe_h5ad_file_name(title),
                url=str(h5ad_asset["url"]),
            )
        )
    return assets


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


def download_file(asset: H5ADAsset, force: bool) -> None:
    H5AD_ROOT.mkdir(parents=True, exist_ok=True)
    destination = H5AD_ROOT / asset.file_name
    if destination.exists() and not force:
        print(f"skip existing: {destination}")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    tmp_path.unlink(missing_ok=True)

    print(f"download {asset.dataset_title}: {asset.url}")
    with urllib.request.urlopen(asset.url) as response, tmp_path.open("wb") as handle:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        next_report = CHUNK_SIZE * 8
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report or (total is not None and downloaded == total):
                print(
                    f"  wrote {_format_bytes(downloaded)} / {_format_bytes(total)}",
                    flush=True,
                )
                next_report += CHUNK_SIZE * 8

    tmp_path.replace(destination)
    print(f"saved: {destination} ({_format_bytes(destination.stat().st_size)})")


def main() -> int:
    args = parse_args()
    try:
        metadata = fetch_collection_metadata()
        write_collection_metadata(metadata)
        assets = iter_h5ad_assets(metadata)
        if args.limit is not None:
            assets = assets[: args.limit]

        if args.list_only:
            for asset in assets:
                print(f"{asset.file_name}\t{asset.dataset_title}\t{asset.url}")
            print(f"Found {len(assets)} H5AD asset(s)")
            return 0

        for asset in assets:
            download_file(asset, args.force)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
