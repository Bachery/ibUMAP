#!/usr/bin/env python3
"""Validate and canonicalize BRAQUE MILAN CSV data for later LNS processing.

This script implements the contract observed in the public Mendeley archive:
remove a verified CSV-export row index, normalize marker names to the token
after the final underscore, keep the first duplicate marker, and preserve every
remaining column as a raw marker intensity.  The public files do not contain
the x/y/area columns assumed by the authors' paper-specific legacy loader, so
this script never invents spatial coordinates by relabelling marker columns.
It intentionally does not run Lognormal Shrinkage; that expensive stochastic
stage must be implemented and frozen separately.  The script is a dry run
unless --run is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd


CASE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = CASE_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
LOG_ROOT = CASE_ROOT / "logs"
DEFAULT_ARCHIVE = RAW_ROOT / "BRAQUE-RawCSVdata.zip"
DEFAULT_ANTIBODIES = RAW_ROOT / "ListOfPrimaryAntibodies.xlsx"
DEFAULT_SAMPLES = ("L2",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--antibodies", type=Path, default=DEFAULT_ANTIBODIES)
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        help="Sample token such as L2, T1, or lymphnode; may be repeated. Default: L2.",
    )
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_ROOT)
    parser.add_argument(
        "--allow-critical-issues",
        action="store_true",
        help="Write processed arrays even when a critical quality gate fails.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true")
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


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.gz")
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_csv_members(archive: zipfile.ZipFile) -> list[str]:
    members: list[str] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or path.suffix.lower() != ".csv":
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {info.filename}")
        # Finder-created ZIP archives contain AppleDouble resource-fork files
        # such as __MACOSX/._L2.csv.  They are metadata, not data samples.
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        members.append(info.filename)
    if not members:
        raise ValueError("Archive contains no CSV files")
    return sorted(members)


def normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def member_sample_name(member: str) -> str:
    return PurePosixPath(member).stem


def resolve_samples(
    members: list[str], requested: Iterable[str] | None, all_samples: bool
) -> list[tuple[str, str]]:
    if all_samples:
        return [(member_sample_name(member), member) for member in members]
    selected: list[tuple[str, str]] = []
    for token in requested or DEFAULT_SAMPLES:
        normalized = normalized_token(token)
        exact = [m for m in members if normalized_token(member_sample_name(m)) == normalized]
        candidates = exact or [
            m for m in members if normalized in normalized_token(member_sample_name(m))
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Sample token {token!r} matched {len(candidates)} CSV members: {candidates}. "
                f"Available: {[member_sample_name(m) for m in members]}"
            )
        selected.append((str(token), candidates[0]))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for label, member in selected:
        if member not in seen:
            seen.add(member)
            unique.append((label, member))
    return unique


def read_member(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member, "r") as handle:
        return pd.read_csv(handle, low_memory=False)


def legacy_short_name(column: Any) -> str:
    value = str(column).strip()
    return value.split("_")[-1].strip()


def exported_index_columns(frame: pd.DataFrame) -> list[str]:
    """Return only unnamed columns that exactly encode the zero-based row index."""
    expected = np.arange(len(frame), dtype=np.int64)
    detected: list[str] = []
    for column in frame.columns:
        name = str(column).strip()
        if not re.fullmatch(r"Unnamed:\s*\d+", name, flags=re.IGNORECASE):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        if values.shape == expected.shape and np.array_equal(values, expected):
            detected.append(name)
    return detected


def numeric_frame(frame: pd.DataFrame, context: str) -> tuple[pd.DataFrame, dict[str, int]]:
    converted = frame.apply(pd.to_numeric, errors="coerce")
    source_missing = frame.isna()
    parse_failures = converted.isna() & ~source_missing
    counts = {
        str(column): int(parse_failures[column].sum())
        for column in converted.columns
        if int(parse_failures[column].sum()) > 0
    }
    if counts:
        print(f"WARNING: numeric parse failures in {context}: {counts}", file=sys.stderr)
    return converted, counts


def read_antibodies(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row: int | None = None
    for index, row in raw.iterrows():
        values = {str(value).strip() for value in row.tolist() if pd.notna(value)}
        if "Name" in values and "Significance" in values:
            header_row = int(index)
            break
    if header_row is None:
        raise ValueError("Could not find Name/Significance header row in antibody workbook")
    frame = pd.read_excel(path, sheet_name=0, header=header_row)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all").reset_index(drop=True)
    if "Name" not in frame or "Significance" not in frame:
        raise ValueError("Antibody workbook lacks required Name/Significance columns")
    frame = frame[frame["Name"].notna()].copy()
    frame["Name"] = frame["Name"].astype(str).str.strip()
    return frame


def antibody_match_report(feature_names: list[str], antibodies: pd.DataFrame) -> dict[str, Any]:
    aliases: dict[str, list[str]] = {}
    for _, row in antibodies.iterrows():
        values = [row.get("Name"), row.get("HGNC name")]
        canonical_name = str(row.get("Name", "")).strip()
        for value in values:
            if pd.isna(value):
                continue
            key = normalized_token(str(value))
            if key:
                aliases.setdefault(key, []).append(canonical_name)
    matched: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for feature in feature_names:
        candidates = sorted(set(aliases.get(normalized_token(feature), [])))
        if candidates:
            matched[feature] = candidates
        else:
            unmatched.append(feature)
    return {
        "feature_count": len(feature_names),
        "matched_feature_count": len(matched),
        "match_rate": len(matched) / len(feature_names) if feature_names else 0.0,
        "matches": matched,
        "unmatched_features": unmatched,
    }


def profile_features(matrix: np.ndarray, feature_names: list[str]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        values = matrix[:, index]
        finite = values[np.isfinite(values)]
        profiles.append(
            {
                "feature_index": index,
                "feature_name": name,
                "finite_count": int(finite.size),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "negative_count": int((finite < 0).sum()),
                "zero_rate": float((finite == 0).mean()) if finite.size else None,
                "min": float(finite.min()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "constant": bool(finite.size and np.ptp(finite) == 0),
            }
        )
    return profiles


def canonicalize_sample(
    frame: pd.DataFrame,
    *,
    sample_name: str,
    archive_path: Path,
    archive_member: str,
    antibodies: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if frame.shape[1] < 1:
        raise ValueError(f"{sample_name}: expected at least one marker column, got {frame.shape}")
    source_columns = [str(column) for column in frame.columns]
    index_columns = exported_index_columns(frame)
    without_index = frame.drop(columns=index_columns).copy()
    marker_source_columns = [str(column) for column in without_index.columns]
    short_names = [legacy_short_name(column) for column in marker_source_columns]
    if any(not name for name in short_names):
        raise ValueError(f"{sample_name}: empty column name after legacy normalization")

    first_index: dict[str, int] = {}
    duplicate_records: list[dict[str, Any]] = []
    keep_indices: list[int] = []
    for index, (source, short) in enumerate(zip(marker_source_columns, short_names)):
        if short in first_index:
            duplicate_records.append(
                {
                    "normalized_name": short,
                    "kept_source_column": marker_source_columns[first_index[short]],
                    "dropped_source_column": source,
                    "dropped_column_index": index,
                }
            )
        else:
            first_index[short] = index
            keep_indices.append(index)
    canonical = without_index.iloc[:, keep_indices].copy()
    canonical_source_columns = [marker_source_columns[index] for index in keep_indices]
    canonical.columns = [short_names[index] for index in keep_indices]
    if canonical.shape[1] < 1:
        raise ValueError(f"{sample_name}: index and duplicate removal left no marker columns")

    marker_numeric, marker_parse_failures = numeric_frame(canonical, "markers")
    marker_matrix = np.asarray(marker_numeric.to_numpy(dtype=np.float32), dtype=np.float32, order="C")
    feature_names = [str(column) for column in marker_numeric.columns]

    cells = pd.DataFrame(
        {
            "cell_id": [f"{sample_name}:{index}" for index in range(len(canonical))],
            "source_row": np.arange(len(canonical), dtype=np.int64),
        }
    )
    features = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names), dtype=np.int64),
            "feature_name": feature_names,
            "source_column": canonical_source_columns,
        }
    )

    finite_markers = np.isfinite(marker_matrix)
    negative_marker_count = int((marker_matrix[finite_markers] < 0).sum())
    all_zero_rows = (
        np.all(marker_matrix == 0, axis=1) if finite_markers.all() else np.zeros(len(marker_matrix), dtype=bool)
    )
    feature_profiles = profile_features(marker_matrix, feature_names)
    critical: list[str] = []
    warnings: list[str] = []
    if marker_matrix.shape[0] == 0 or marker_matrix.shape[1] == 0:
        critical.append("empty marker matrix")
    if not finite_markers.all():
        critical.append(
            f"marker matrix contains {int(np.isnan(marker_matrix).sum())} NaN and "
            f"{int(np.isinf(marker_matrix).sum())} Inf values"
        )
    if negative_marker_count:
        critical.append(f"marker matrix contains {negative_marker_count} negative values")
    if cells["cell_id"].duplicated().any():
        critical.append("generated cell IDs are not unique")
    constant_features = [p["feature_name"] for p in feature_profiles if p["constant"]]
    if constant_features:
        warnings.append(f"constant markers: {constant_features}")
    if int(all_zero_rows.sum()):
        warnings.append(f"all-zero cells: {int(all_zero_rows.sum())}")
    if duplicate_records:
        warnings.append(
            f"dropped {len(duplicate_records)} duplicate marker/column names using the author's keep-first rule"
        )

    antibody_report = antibody_match_report(feature_names, antibodies)
    if antibody_report["match_rate"] < 0.5:
        warnings.append(
            f"antibody-table feature match rate is low: {antibody_report['match_rate']:.3f}"
        )
    quality = {
        "created_at": now_utc(),
        "sample": sample_name,
        "grain": "one row per segmented cell; one feature column per retained marker",
        "source": {
            "archive_path": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "archive_member": archive_member,
            "source_shape": list(frame.shape),
            "source_columns": source_columns,
        },
        "public_archive_column_contract": {
            "rule": "drop verified CSV-export row index; normalize each marker to final underscore token; keep first duplicate; retain every remaining column as a marker",
            "retained_shape": list(canonical.shape),
            "export_index_columns_dropped": index_columns,
            "duplicate_columns_dropped": duplicate_records,
        },
        "matrix": {
            "shape": list(marker_matrix.shape),
            "dtype": str(marker_matrix.dtype),
            "nan_count": int(np.isnan(marker_matrix).sum()),
            "inf_count": int(np.isinf(marker_matrix).sum()),
            "negative_count": negative_marker_count,
            "all_zero_cell_count": int(all_zero_rows.sum()),
            "constant_feature_count": len(constant_features),
        },
        "spatial": {
            "available": False,
            "reason": "The public Mendeley CSV contains an export index and marker intensities only; no x/y/area columns are present.",
        },
        "numeric_parse_failures": {
            "markers": marker_parse_failures,
        },
        "feature_profiles": feature_profiles,
        "antibody_matching": antibody_report,
        "quality_gate": {
            "critical_issues": critical,
            "warnings": warnings,
            "passed": not critical,
        },
    }
    return marker_matrix, cells, features, quality


def output_paths(root: Path) -> dict[str, Path]:
    return {
        "markers": root / "markers_raw.npy",
        "cells": root / "cells.csv.gz",
        "features": root / "features.csv.gz",
        "antibodies": root / "antibodies.csv.gz",
        "metadata": root / "metadata.json",
        "quality": root / "quality_report.json",
        "checksums": root / "SHA256SUMS",
    }


def write_sample_outputs(
    root: Path,
    marker_matrix: np.ndarray,
    cells: pd.DataFrame,
    features: pd.DataFrame,
    antibodies: pd.DataFrame,
    quality: dict[str, Any],
    *,
    overwrite: bool,
    allow_critical: bool = False,
) -> dict[str, Any]:
    paths = output_paths(root)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs already exist; pass --overwrite: {existing}")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["quality"], quality)
    if quality["quality_gate"]["critical_issues"] and not allow_critical:
        raise ValueError(
            "Critical data-quality issues: "
            + "; ".join(quality["quality_gate"]["critical_issues"])
        )
    atomic_save_npy(paths["markers"], marker_matrix)
    atomic_write_csv_gzip(paths["cells"], cells)
    atomic_write_csv_gzip(paths["features"], features)
    atomic_write_csv_gzip(paths["antibodies"], antibodies)
    reloaded = np.load(paths["markers"], allow_pickle=False)
    if reloaded.shape != marker_matrix.shape or reloaded.dtype != marker_matrix.dtype:
        raise ValueError("Reloaded markers_raw.npy does not match the written data contract")
    metadata = {
        "created_at": now_utc(),
        "sample": quality["sample"],
        "stage": "canonical raw MILAN data; feature selection and LNS not yet applied",
        "cell_count": int(marker_matrix.shape[0]),
        "feature_count": int(marker_matrix.shape[1]),
        "dtype": str(marker_matrix.dtype),
        "cell_id_column": "cell_id",
        "feature_name_column": "feature_name",
        "quality_gate_passed": bool(quality["quality_gate"]["passed"]),
        "quality_override_used": bool(quality["quality_gate"].get("override_used", False)),
        "files": {},
    }
    for name in ("markers", "cells", "features", "antibodies", "quality"):
        path = paths[name]
        metadata["files"][path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_write_json(paths["metadata"], metadata)
    checksum_lines = []
    for name in ("markers", "cells", "features", "antibodies", "quality", "metadata"):
        path = paths[name]
        checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    atomic_write_text(paths["checksums"], "\n".join(checksum_lines) + "\n")
    return metadata


def write_with_optional_override(
    root: Path,
    marker_matrix: np.ndarray,
    cells: pd.DataFrame,
    features: pd.DataFrame,
    antibodies: pd.DataFrame,
    quality: dict[str, Any],
    *,
    overwrite: bool,
    allow_critical: bool,
) -> dict[str, Any]:
    if not allow_critical or not quality["quality_gate"]["critical_issues"]:
        return write_sample_outputs(
            root,
            marker_matrix,
            cells,
            features,
            antibodies,
            quality,
            overwrite=overwrite,
            allow_critical=False,
        )
    quality = dict(quality)
    quality["quality_gate"] = dict(quality["quality_gate"])
    quality["quality_gate"]["override_used"] = True
    quality["quality_gate"]["warnings"] = list(quality["quality_gate"]["warnings"]) + [
        "Critical-quality override used; outputs are not approved for formal experiments."
    ]
    return write_sample_outputs(
        root,
        marker_matrix,
        cells,
        features,
        antibodies,
        quality,
        overwrite=overwrite,
        allow_critical=True,
    )


def main() -> int:
    args = parse_args()
    print(f"Archive: {args.archive}")
    print(f"Antibodies: {args.antibodies}")
    print(f"Output root: {args.output_dir}")
    print(f"Samples: {'all archive CSVs' if args.all_samples else args.samples or list(DEFAULT_SAMPLES)}")
    if not args.run:
        print("Dry run only; pass --run after downloading the source files.")
        return 0
    if not args.archive.is_file():
        raise FileNotFoundError(f"Archive not found: {args.archive}")
    if not args.antibodies.is_file():
        raise FileNotFoundError(f"Antibody workbook not found: {args.antibodies}")
    if not zipfile.is_zipfile(args.archive):
        raise ValueError(f"Not a valid ZIP archive: {args.archive}")

    antibodies = read_antibodies(args.antibodies)
    run_report: dict[str, Any] = {
        "created_at": now_utc(),
        "archive": {
            "path": str(args.archive),
            "size_bytes": args.archive.stat().st_size,
            "sha256": sha256_file(args.archive),
        },
        "antibodies": {
            "path": str(args.antibodies),
            "size_bytes": args.antibodies.stat().st_size,
            "sha256": sha256_file(args.antibodies),
            "rows": int(len(antibodies)),
        },
        "samples": [],
    }
    failures = 0
    with zipfile.ZipFile(args.archive) as archive:
        members = safe_csv_members(archive)
        selected = resolve_samples(members, args.samples, args.all_samples)
        run_report["archive_csv_members"] = members
        for requested_name, member in selected:
            sample_name = member_sample_name(member)
            record: dict[str, Any] = {
                "requested_sample": requested_name,
                "sample": sample_name,
                "archive_member": member,
                "started_at": now_utc(),
            }
            try:
                frame = read_member(archive, member)
                marker_matrix, cells, features, quality = canonicalize_sample(
                    frame,
                    sample_name=sample_name,
                    archive_path=args.archive,
                    archive_member=member,
                    antibodies=antibodies,
                )
                destination = args.output_dir / sample_name
                record["quality_gate"] = quality["quality_gate"]
                record["output"] = write_with_optional_override(
                    destination,
                    marker_matrix,
                    cells,
                    features,
                    antibodies,
                    quality,
                    overwrite=args.overwrite,
                    allow_critical=args.allow_critical_issues,
                )
                record["status"] = "prepared"
            except Exception as exc:
                failures += 1
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            record["finished_at"] = now_utc()
            run_report["samples"].append(record)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_ROOT / f"02_check_and_prepare_data_{timestamp}.json"
    atomic_write_json(log_path, run_report)
    if failures:
        print(f"Preparation completed with {failures} failure(s); see {log_path}", file=sys.stderr)
        return 1
    print(f"Preparation log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
