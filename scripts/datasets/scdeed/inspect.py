from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import importlib.util
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "datasets/raw/scdeed"
DOWNLOAD_ROOT = RAW_ROOT / "downloads"
EXTRACTED_ROOT = RAW_ROOT / "extracted"
MANIFEST_PATH = RAW_ROOT / "inspect_manifest.json"
LARGE_TEXT_THRESHOLD_BYTES = 100 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect downloaded scDEED raw data files.")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract supported archives to datasets/raw/scdeed/extracted.",
    )
    parser.add_argument("--limit", type=int, help="Only inspect the first N filesystem entries.")
    parser.add_argument(
        "--count-large-text",
        action="store_true",
        help="Count rows in large CSV/TSV/TXT files instead of reading only the header.",
    )
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def file_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar.gz"
    if name.endswith(".csv.gz"):
        return "csv.gz"
    suffix = path.suffix.lower()
    if suffix in {".rda", ".rdata"}:
        return "rda"
    if suffix == ".rds":
        return "rds"
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".txt":
        return "txt"
    if suffix == ".h5ad":
        return "h5ad"
    if suffix == ".loom":
        return "loom"
    if suffix == ".zip":
        return "zip"
    if suffix == ".tar":
        return "tar"
    return "unknown"


def role_name_terms(path: Path) -> tuple[str, set[str]]:
    lower = path.name.lower()
    for suffix in [".tar.gz", ".csv.gz", ".tgz"]:
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
            break
    else:
        while Path(lower).suffix:
            lower = lower[: -len(Path(lower).suffix)]
    return lower, set(re.findall(r"[a-z0-9]+", lower))


def candidate_roles_for_name(path: Path, shape: list[int] | None = None) -> list[str]:
    stem, terms = role_name_terms(path)
    roles: set[str] = set()
    if terms & {"x", "feature", "features", "count", "counts", "input", "expression", "matrix"}:
        roles.add("features")
    if (
        terms & {"y", "obs", "meta", "metadata"}
        or any(token in stem for token in ["label", "cluster", "annotation", "cell_type"])
    ):
        roles.add("obs")
    if terms & {"umap", "tsne", "embedding", "embeddings"}:
        roles.add("reference_embedding")
    if terms & {"scdeed", "dubious", "score", "scores", "flag", "flags"}:
        roles.add("scdeed_output")
    if shape is not None and len(shape) == 2 and shape[0] > 10 and 2 < shape[1] <= 500:
        roles.add("features")
    if not roles:
        roles.add("unknown")
    return sorted(roles)


def inspect_npy(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = [int(value) for value in array.shape]
    return {
        "shape": shape,
        "dtype": str(array.dtype),
        "candidate_roles": candidate_roles_for_name(path, shape),
    }


def inspect_npz(path: Path) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as handle:
        for name in handle.files:
            array = handle[name]
            shape = [int(value) for value in array.shape]
            arrays[name] = {
                "shape": shape,
                "dtype": str(array.dtype),
                "candidate_roles": candidate_roles_for_name(Path(name), shape),
            }
    return {"arrays": arrays, "candidate_roles": ["unknown"]}


def text_delimiter(kind: str) -> str | None:
    if kind == "tsv":
        return "\t"
    return None


def inspect_text(path: Path, count_large_text: bool) -> dict[str, Any]:
    kind = file_kind(path)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    delimiter = text_delimiter(kind)
    with opener(path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=delimiter or ",")
        header = next(reader, [])
        should_count = count_large_text or path.stat().st_size <= LARGE_TEXT_THRESHOLD_BYTES
        rows = sum(1 for _ in reader) if should_count else None
    return {
        "n_rows": rows,
        "row_count_mode": "full" if rows is not None else "skipped_large_text",
        "n_columns": len(header),
        "columns_preview": header[:20],
        "candidate_roles": candidate_roles_for_name(path),
    }


def h5_dataset_summary(obj: Any) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in getattr(obj, "shape", [])],
        "dtype": str(getattr(obj, "dtype", "")),
    }


def inspect_obs_group(group: Any) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name, item in group.items():
        if hasattr(item, "keys") and {"categories", "codes"}.issubset(set(item.keys())):
            columns[name] = {
                "encoding": "categorical",
                "shape": [int(value) for value in item["codes"].shape],
                "codes_dtype": str(item["codes"].dtype),
                "n_categories": int(item["categories"].shape[0]),
            }
        elif hasattr(item, "shape"):
            columns[name] = h5_dataset_summary(item)
    return columns


def inspect_h5ad(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("h5py") is None:
        return {"status": "unsupported_without_h5py", "candidate_roles": ["unknown"]}
    import h5py

    with h5py.File(path, "r") as handle:
        output: dict[str, Any] = {
            "candidate_roles": ["features", "obs"],
            "h5ad": {},
        }
        if "X" in handle:
            output["h5ad"]["X"] = h5_dataset_summary(handle["X"])
        if "obs" in handle:
            obs_columns = inspect_obs_group(handle["obs"])
            output["h5ad"]["obs_columns"] = obs_columns
            output["h5ad"]["n_obs"] = next(
                (
                    int(value["shape"][0])
                    for value in obs_columns.values()
                    if value.get("shape")
                ),
                None,
            )
        if "obsm" in handle:
            output["h5ad"]["obsm"] = {
                name: h5_dataset_summary(item)
                for name, item in handle["obsm"].items()
                if hasattr(item, "shape")
            }
        if "var" in handle:
            output["h5ad"]["var_columns"] = inspect_obs_group(handle["var"])
        return output


def inspect_loom(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("h5py") is None:
        return {"status": "unsupported_without_h5py", "candidate_roles": ["unknown"]}
    import h5py

    with h5py.File(path, "r") as handle:
        output: dict[str, Any] = {"candidate_roles": ["features"], "loom": {}}
        if "matrix" in handle:
            output["loom"]["matrix"] = h5_dataset_summary(handle["matrix"])
        for key in ["row_attrs", "col_attrs", "layers"]:
            if key in handle:
                output["loom"][key] = sorted(handle[key].keys())
        return output


def archive_members(path: Path, extract: bool) -> dict[str, Any]:
    output: dict[str, Any] = {"members": [], "candidate_roles": ["unknown"]}
    extract_dir = EXTRACTED_ROOT / path.stem.replace(".tar", "")
    if file_kind(path) == "zip":
        with zipfile.ZipFile(path) as handle:
            output["members"] = handle.namelist()
            if extract:
                extract_dir.mkdir(parents=True, exist_ok=True)
                handle.extractall(extract_dir)
                output["extracted_to"] = repo_relative(extract_dir)
    else:
        mode = "r:gz" if file_kind(path) == "tar.gz" else "r"
        with tarfile.open(path, mode) as handle:
            output["members"] = handle.getnames()
            if extract:
                extract_dir.mkdir(parents=True, exist_ok=True)
                handle.extractall(extract_dir)
                output["extracted_to"] = repo_relative(extract_dir)
    output["members_preview"] = output["members"][:50]
    output["n_members"] = len(output["members"])
    return output


def inspect_r_file(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("pyreadr") is None:
        return {
            "status": "unsupported_without_pyreadr",
            "candidate_roles": candidate_roles_for_name(path),
        }
    import pyreadr  # type: ignore[import-not-found]

    result = pyreadr.read_r(str(path))
    objects: dict[str, Any] = {}
    for name, obj in result.items():
        objects[name] = {
            "shape": [int(value) for value in getattr(obj, "shape", [])],
            "columns_preview": [str(value) for value in getattr(obj, "columns", [])[:20]],
        }
    return {"objects": objects, "candidate_roles": candidate_roles_for_name(path)}


def inspect_file(path: Path, extract: bool, count_large_text: bool) -> dict[str, Any]:
    kind = file_kind(path)
    record: dict[str, Any] = {
        "path": repo_relative(path),
        "file_name": path.name,
        "file_type": kind,
        "size_bytes": int(path.stat().st_size),
    }
    try:
        if kind == "npy":
            record.update(inspect_npy(path))
        elif kind == "npz":
            record.update(inspect_npz(path))
        elif kind in {"csv", "csv.gz", "tsv", "txt"}:
            record.update(inspect_text(path, count_large_text))
        elif kind == "h5ad":
            record.update(inspect_h5ad(path))
        elif kind == "loom":
            record.update(inspect_loom(path))
        elif kind in {"zip", "tar", "tar.gz"}:
            record.update(archive_members(path, extract))
        elif kind in {"rda", "rds"}:
            record.update(inspect_r_file(path))
        else:
            record["candidate_roles"] = ["unknown"]
    except Exception as exc:
        record["status"] = "inspect_failed"
        record["error"] = str(exc)
        record["candidate_roles"] = ["unknown"]
    return record


def iter_files() -> list[Path]:
    roots = [DOWNLOAD_ROOT, EXTRACTED_ROOT]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(dict.fromkeys(files))


def dataset_id_from_raw_path(path: str) -> str:
    parts = Path(path).parts
    if "extracted" in parts:
        index = parts.index("extracted")
        if index + 1 < len(parts):
            return re.sub(r"[^A-Za-z0-9]+", "_", parts[index + 1]).strip("_").lower()
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(path).stem).strip("_").lower()


def build_h5ad_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    h5ad = record.get("h5ad")
    if not isinstance(h5ad, dict):
        return None
    obsm = h5ad.get("obsm") if isinstance(h5ad.get("obsm"), dict) else {}
    n_obs = h5ad.get("n_obs")
    feature_key = None
    for key in ["X_pca", "X_scvi"]:
        summary = obsm.get(key)
        if isinstance(summary, dict) and summary.get("shape"):
            feature_key = f"obsm/{key}"
            break
    if feature_key is None:
        x_summary = h5ad.get("X")
        if isinstance(x_summary, dict):
            shape = x_summary.get("shape", [])
            if len(shape) == 2 and int(shape[1]) <= 500:
                feature_key = "X"
    if feature_key is None:
        return None

    feature_summary = h5ad["X"] if feature_key == "X" else obsm[feature_key.split("/", 1)[1]]
    feature_shape = feature_summary.get("shape", [])
    if len(feature_shape) != 2 or n_obs is None or int(feature_shape[0]) != int(n_obs):
        return None

    references: dict[str, dict[str, Any]] = {}
    for key, summary in obsm.items():
        shape = summary.get("shape", [])
        if key in {"X_umap", "X_tsne"} and len(shape) == 2 and int(shape[0]) == int(n_obs):
            ref_name = key.removeprefix("X_")
            references[ref_name] = {
                "key": f"obsm/{key}",
                "shape": shape,
                "dtype": summary.get("dtype"),
            }

    return {
        "dataset_id": dataset_id_from_raw_path(str(record["path"])),
        "status": "ready",
        "confidence": "high",
        "source_format": "h5ad",
        "raw_path": record["path"],
        "feature_key": feature_key,
        "feature_shape": feature_shape,
        "feature_dtype": feature_summary.get("dtype"),
        "obs_key": "obs",
        "obs_rows": int(n_obs),
        "reference_embeddings": references,
        "notes": "H5AD candidate from downloaded and extracted Zenodo raw data.",
    }


def build_loom_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    loom = record.get("loom")
    if not isinstance(loom, dict):
        return None
    matrix = loom.get("matrix")
    if not isinstance(matrix, dict):
        return None
    shape = matrix.get("shape", [])
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        return None

    col_attrs = set(loom.get("col_attrs", []))
    references: dict[str, dict[str, Any]] = {}
    if {"cr_tSNE1", "cr_tSNE2"}.issubset(col_attrs):
        references["cr_tsne"] = {
            "columns": ["col_attrs/cr_tSNE1", "col_attrs/cr_tSNE2"],
            "shape": [int(shape[1]), 2],
        }
    if {"cr_PC1", "cr_PC2"}.issubset(col_attrs):
        references["cr_pca"] = {
            "columns": ["col_attrs/cr_PC1", "col_attrs/cr_PC2"],
            "shape": [int(shape[1]), 2],
        }

    return {
        "dataset_id": dataset_id_from_raw_path(str(record["path"])),
        "status": "ready",
        "confidence": "medium",
        "source_format": "loom",
        "raw_path": record["path"],
        "feature_key": "matrix",
        "feature_shape": [int(shape[1]), int(shape[0])],
        "feature_dtype": matrix.get("dtype"),
        "feature_orientation": "source loom matrix is genes by cells; processed features are cells by genes",
        "obs_key": "col_attrs",
        "obs_rows": int(shape[1]),
        "reference_embeddings": references,
        "notes": "Loom candidate from downloaded and extracted Zenodo raw data.",
    }


def object_summary(record: dict[str, Any], object_name: str) -> dict[str, Any] | None:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return None
    summary = objects.get(object_name)
    if summary is None and object_name == "null":
        summary = objects.get(None)
    return summary if isinstance(summary, dict) else None


def find_record(records_by_name: dict[str, dict[str, Any]], file_name: str) -> dict[str, Any] | None:
    return records_by_name.get(file_name)


def r_object_reference(record: dict[str, Any], object_name: str) -> dict[str, Any]:
    return {
        "path": record["path"],
        "object": object_name,
    }


def build_samusik_candidate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    samusik_records = [
        record
        for record in records
        if "/Samusik/" in str(record.get("path", "")) and record.get("status", "ok") == "ok"
    ]
    records_by_name = {str(record.get("file_name")): record for record in samusik_records}
    feature_record = find_record(records_by_name, "xp.RData")
    label_record = find_record(records_by_name, "label.RData")
    if feature_record is None or label_record is None:
        return None
    feature_summary = object_summary(feature_record, "xp")
    label_summary = object_summary(label_record, "cluster_label")
    if feature_summary is None or label_summary is None:
        return None
    feature_shape = feature_summary.get("shape", [])
    label_shape = label_summary.get("shape", [])
    if (
        len(feature_shape) != 2
        or len(label_shape) != 2
        or int(feature_shape[0]) != int(label_shape[0])
        or int(label_shape[1]) != 1
    ):
        return None

    obs_extra = []
    populations_record = find_record(records_by_name, "populations_manual.RData")
    if populations_record is not None and object_summary(populations_record, "populations"):
        obs_extra.append(
            {
                "column": "population_numeric",
                **r_object_reference(populations_record, "populations"),
            }
        )

    references: dict[str, dict[str, Any]] = {}
    for name, file_name, object_name in [
        ("umap", "umap.RData", "umap"),
        ("umap_mindist0_05", "loadings_mindist0.05.Rds", "null"),
        ("umap_n160_m0_7", "loadings_n160_m0.7.Rds", "null"),
        ("umap_nneighbors160", "loadings_nneighbors160.Rds", "null"),
    ]:
        record = find_record(records_by_name, file_name)
        summary = object_summary(record, object_name) if record is not None else None
        shape = summary.get("shape", []) if summary is not None else []
        if len(shape) == 2 and int(shape[0]) == int(feature_shape[0]) and int(shape[1]) == 2:
            references[name] = {
                **r_object_reference(record, object_name),
                "shape": shape,
            }

    metrics: dict[str, dict[str, Any]] = {}
    for name, file_name, object_name in [
        ("dubious_joint_umap", "dubiousNo_joint_UMAP.RData", "dubious_number"),
        ("dubious_mindist_umap", "dubiousNo_mindist_UMAP.RData", "dubious_number_UMAP"),
        ("dubious_nneighbors_umap", "dubiousNo_nNeighbors_UMAP.RData", "dubious_number_nneighbors"),
    ]:
        record = find_record(records_by_name, file_name)
        summary = object_summary(record, object_name) if record is not None else None
        if summary is not None:
            metrics[name] = {
                **r_object_reference(record, object_name),
                "shape": summary.get("shape", []),
            }

    return {
        "dataset_id": "samusik",
        "status": "ready",
        "confidence": "high",
        "source_format": "r_table",
        "feature_path": feature_record["path"],
        "feature_object": "xp",
        "feature_shape": feature_shape,
        "feature_columns_preview": feature_summary.get("columns_preview", []),
        "obs_path": label_record["path"],
        "obs_object": "cluster_label",
        "obs_column": "cluster_label",
        "obs_rows": int(feature_shape[0]),
        "obs_extra": obs_extra,
        "reference_embeddings": references,
        "reference_metrics": metrics,
        "notes": "Samusik feature and label tables recognized from downloaded and extracted Zenodo raw RData/Rds files.",
    }


def build_cart_candidate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    cart_records = [
        record
        for record in records
        if "/CART/" in str(record.get("path", "")) and record.get("status", "ok") == "ok"
    ]
    records_by_name = {str(record.get("file_name")): record for record in cart_records}
    feature_record = find_record(records_by_name, "normalised_counts.tsv")
    metadata_record = find_record(records_by_name, "metadata_grey.tsv")
    full_metadata_record = find_record(records_by_name, "meta-data.txt")
    if feature_record is None or metadata_record is None:
        return None
    n_columns = feature_record.get("n_columns")
    if not isinstance(n_columns, int) or n_columns <= 2:
        return None

    metrics: dict[str, dict[str, Any]] = {}
    dubious_record = find_record(records_by_name, "dubiousNo_tSNE.RData")
    if dubious_record is not None:
        summary = object_summary(dubious_record, "dubious_number_tSNE")
        if summary is not None:
            metrics["dubious_tsne"] = {
                **r_object_reference(dubious_record, "dubious_number_tSNE"),
                "shape": summary.get("shape", []),
            }

    return {
        "dataset_id": "cart",
        "status": "ready",
        "confidence": "high",
        "source_format": "cart_tsv",
        "feature_path": feature_record["path"],
        "obs_path": metadata_record["path"],
        "full_metadata_path": full_metadata_record["path"] if full_metadata_record else None,
        "feature_shape": [int(n_columns) - 1, None],
        "obs_rows": metadata_record.get("n_rows"),
        "reference_embeddings": {"tsne": {"source": "meta-data.txt:tSNE_1,tSNE_2"}}
        if full_metadata_record
        else {},
        "reference_metrics": metrics,
        "notes": "CART candidate from normalized counts TSV plus downloaded raw metadata tables.",
    }


def build_prepare_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        if record.get("file_type") == "h5ad":
            candidate = build_h5ad_candidate(record)
            if candidate is not None:
                candidates.append(candidate)
        if record.get("file_type") == "loom":
            candidate = build_loom_candidate(record)
            if candidate is not None:
                candidates.append(candidate)
    samusik = build_samusik_candidate(records)
    if samusik is not None:
        candidates.append(samusik)
    cart = build_cart_candidate(records)
    if cart is not None:
        candidates.append(cart)
    return candidates


def main() -> int:
    args = parse_args()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    files = iter_files()
    if args.limit is not None:
        files = files[: args.limit]

    records = [
        inspect_file(path, args.extract, args.count_large_text)
        for path in files
    ]
    candidates = build_prepare_candidates(records)

    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "raw_roots": {
            "downloads": repo_relative(DOWNLOAD_ROOT),
            "extracted": repo_relative(EXTRACTED_ROOT),
        },
        "pyreadr_available": importlib.util.find_spec("pyreadr") is not None,
        "h5py_available": importlib.util.find_spec("h5py") is not None,
        "files": records,
        "prepare_candidates": candidates,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {MANIFEST_PATH} with {len(records)} file record(s)")
    if candidates:
        print("Prepare candidates: " + ", ".join(str(item["dataset_id"]) for item in candidates))
    else:
        print("No prepare candidates recognized from downloaded raw data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
