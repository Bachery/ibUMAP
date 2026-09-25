"""Build datasets/catalog.json.

The catalog lists every dataset with a reference record in ``datasets/reference/``
(the datasets of the paper), described by that record's metadata, plus any
dataset prepared under ``datasets/processed/`` that has no reference record.
``processed_path`` is the output directory of each dataset, whether or not it has
been prepared yet. The file is left untouched when its content would not change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common.metadata import read_metadata, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "datasets/processed"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "datasets/reference"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/catalog.json"


CATALOG_FIELDS = [
    "dataset_id",
    "display_name",
    "kind",
    "family",
    "processed_path",
    "raw_path",
    "raw_role",
    "source",
    "source_type",
    "source_library",
    "source_loader",
    "source_name",
    "task_type",
    "collection_id",
    "collection_api",
    "zenodo_record_id",
    "source_url",
    "source_publication",
    "obs_file",
    "identifier_column",
    "original_index_column",
    "default_color_by",
    "label_columns",
    "colorable_columns",
    "searchable_columns",
    "filterable_columns",
    "obs_schema",
    "feature_shape",
    "feature_dtype",
    "feature_source",
    "pca_components",
    "pseudotime_file",
    "pseudotime_source",
    "source_indices_file",
    "raw_paths",
    "query_shape",
    "neighbor_shape",
    "distance_shape",
    "reference_embedding_file",
    "reference_embedding_source",
    "reference_embedding_shape",
    "reference_embeddings",
    "reference_metrics",
    "extra_arrays",
    "target_file",
    "target_dtype",
    "target_source_column",
    "legacy_source_path",
    "legacy_feature_file",
    "legacy_label_file",
    "original_feature_dtype",
    "original_label_dtype",
    "label_encoding_strategy",
    "label_mapping",
    "n_classes",
    "skipped_obs_columns",
    "skipped_or_unrecognized_sources",
    "raw_h5ad_keys_summary",
    "metric",
    "subset_id",
    "subset_kind",
    "subset_filter",
    "n_cells",
    "label_field",
    "selected_labels",
    "version",
    "generated_at",
    "download_url",
    "download_urls",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build datasets/catalog.json from metadata files.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_catalog(processed_root: Path, reference_root: Path = DEFAULT_REFERENCE_ROOT) -> dict[str, Any]:
    processed_root, reference_root = Path(processed_root), Path(reference_root)
    sources = {path.parent.name: path for path in processed_root.glob("*/metadata.json")}
    # Reference records take precedence: the catalog describes the paper's datasets
    # even before (or regardless of whether) they have been prepared.
    sources.update({path.parent.name: path for path in reference_root.glob("*/metadata.json")})
    entries: list[dict[str, Any]] = []
    for name, metadata_path in sources.items():
        entry = dict(read_metadata(metadata_path))
        entry["processed_path"] = _repo_relative(processed_root / name)
        entries.append({field: entry.get(field) for field in CATALOG_FIELDS})

    entries.sort(key=lambda item: str(item.get("dataset_id", "")))
    return {
        "generated_at": utc_now_iso(),
        "datasets": entries,
    }


def main() -> int:
    args = parse_args()
    catalog = build_catalog(args.processed_root, args.reference_root)
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict) and existing.get("datasets") == catalog["datasets"]:
            print(f"{args.output} is up to date ({len(catalog['datasets'])} dataset(s))")
            return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} with {len(catalog['datasets'])} dataset(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
