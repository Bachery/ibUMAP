from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTION_ID = "0cca8620-8dee-45d0-aef5-23f032a5cf09"
COLLECTION_API = (
    "https://api.cellxgene.cziscience.com/curation/v1/collections/"
    f"{COLLECTION_ID}"
)
RAW_ROOT = REPO_ROOT / "datasets/raw/whole_mouse_brain_merfish"
H5AD_ROOT = RAW_ROOT / "h5ad"
COLLECTION_METADATA_PATH = RAW_ROOT / "collection_metadata.json"
CHUNK_SIZE = 1024 * 1024 * 8

DATASET_TITLES = (
    "WB_MERFISH_animal1_coronal",
    "WB_MERFISH_animal2_coronal",
    "WB_MERFISH_animal3_sagittal",
    "WB_MERFISH_animal4_sagittal",
)


@dataclass(frozen=True)
class H5ADAsset:
    dataset_title: str
    file_name: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the four requested whole mouse brain MERFISH H5AD files "
            "from the CELLxGENE collection API."
        )
    )
    parser.add_argument("--force", action="store_true", help="Redownload files that already exist.")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List the selected H5AD assets without writing metadata or downloading files.",
    )
    return parser.parse_args()


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


def select_h5ad_assets(metadata: dict[str, Any]) -> list[H5ADAsset]:
    datasets_by_title: dict[str, dict[str, Any]] = {}
    for dataset in metadata.get("datasets", []):
        title = str(dataset.get("title") or "")
        if title in DATASET_TITLES:
            if title in datasets_by_title:
                raise ValueError(f"Collection contains multiple datasets titled {title!r}")
            datasets_by_title[title] = dataset

    missing = [title for title in DATASET_TITLES if title not in datasets_by_title]
    if missing:
        raise ValueError(f"Requested dataset(s) were not found in the collection: {', '.join(missing)}")

    assets: list[H5ADAsset] = []
    for title in DATASET_TITLES:
        h5ad_assets = [
            asset
            for asset in datasets_by_title[title].get("assets", [])
            if str(asset.get("filetype", "")).upper() == "H5AD" and asset.get("url")
        ]
        if len(h5ad_assets) != 1:
            raise ValueError(
                f"Expected exactly one H5AD asset for {title!r}; found {len(h5ad_assets)}"
            )
        assets.append(
            H5ADAsset(
                dataset_title=title,
                file_name=f"{title}.h5ad",
                url=str(h5ad_assets[0]["url"]),
            )
        )
    return assets


def format_bytes(value: int | None) -> str:
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
                    f"  wrote {format_bytes(downloaded)} / {format_bytes(total)}",
                    flush=True,
                )
                next_report += CHUNK_SIZE * 8

    tmp_path.replace(destination)
    print(f"saved: {destination} ({format_bytes(destination.stat().st_size)})")


def main() -> int:
    args = parse_args()
    try:
        metadata = fetch_collection_metadata()
        assets = select_h5ad_assets(metadata)

        if args.list_only:
            for asset in assets:
                print(f"{asset.file_name}\t{asset.dataset_title}\t{asset.url}")
            print(f"Found {len(assets)} requested H5AD asset(s)")
            return 0

        write_collection_metadata(metadata)
        for asset in assets:
            download_file(asset, args.force)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
