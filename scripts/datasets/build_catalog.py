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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_catalog(processed_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for metadata_path in sorted(Path(processed_root).glob("*/metadata.json")):
        metadata = read_metadata(metadata_path)
        dataset_dir = metadata_path.parent
        entry = dict(metadata)
        entry["processed_path"] = _repo_relative(dataset_dir)
        entries.append({field: entry.get(field) for field in CATALOG_FIELDS})

    entries.sort(key=lambda item: str(item.get("dataset_id", "")))
    return {
        "generated_at": utc_now_iso(),
        "datasets": entries,
    }


def main() -> int:
    args = parse_args()
    catalog = build_catalog(args.processed_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} with {len(catalog['datasets'])} dataset(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
