#!/usr/bin/env python3
"""Select BRAQUE markers using a transparent public-antibody-table proxy.

The paper-specific reference.csv (including ``Ab`` and ``Lineage Defining``)
is not public.  This stage therefore reproduces only the author's documented
drop rules that the Mendeley antibody workbook can support: exact normalized
matching against Name/HGNC name, non-missing Significance, and exclusion of
Significance values containing IGNORE.  Every decision is written to a report.
The script is a dry run unless ``--run`` is supplied.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CASE_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = CASE_ROOT / "data" / "processed"
POLICY_ID = "public_antibody_proxy_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="L2")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict-paper-reference", action="store_true")
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


def normalized_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.gz")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as zipped:
                frame.to_csv(zipped, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_stage_directory(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {Path("/"), Path.home().resolve(), CASE_ROOT.resolve(), (CASE_ROOT / "data").resolve()}
    if resolved in forbidden or len(resolved.parts) < 4:
        raise ValueError(f"Refusing unsafe recursive overwrite target: {resolved}")
    shutil.rmtree(resolved)


def verify_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing input checksum ledger: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Input checksum mismatch: {path}")


def alias_index(antibodies: pd.DataFrame) -> dict[str, list[int]]:
    aliases: dict[str, list[int]] = {}
    for index, row in antibodies.iterrows():
        for column in ("Name", "HGNC name"):
            value = row.get(column)
            if pd.isna(value):
                continue
            token = normalized_token(value)
            if token:
                aliases.setdefault(token, []).append(int(index))
    return {token: sorted(set(indices)) for token, indices in aliases.items()}


def select_features(features: pd.DataFrame, antibodies: pd.DataFrame) -> pd.DataFrame:
    aliases = alias_index(antibodies)
    records: list[dict[str, Any]] = []
    for row in features.itertuples(index=False):
        feature = str(row.feature_name)
        matches = aliases.get(normalized_token(feature), [])
        status = "selected"
        reason = "exact normalized match with non-missing, non-IGNORE Significance"
        reference_name: str | None = None
        significance: str | None = None
        if len(matches) == 0:
            status, reason = "excluded", "no exact Name/HGNC-name match in public antibody workbook"
        elif len(matches) > 1:
            status, reason = "excluded", "ambiguous public antibody workbook match"
        else:
            reference = antibodies.iloc[matches[0]]
            reference_name = str(reference.get("Name", "")).strip() or None
            raw_significance = reference.get("Significance")
            if pd.isna(raw_significance) or not str(raw_significance).strip():
                status, reason = "excluded", "missing Significance in public antibody workbook"
            else:
                significance = str(raw_significance).strip()
                if "IGNORE" in significance.upper():
                    status, reason = "excluded", "Significance contains IGNORE"
        records.append(
            {
                "source_feature_index": int(row.feature_index),
                "feature_name": feature,
                "source_column": str(row.source_column),
                "status": status,
                "reason": reason,
                "reference_name": reference_name,
                "significance": significance,
                "match_count": len(matches),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir or PROCESSED_ROOT / args.sample
    output_dir = args.output_dir or input_dir / "features_selected"
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Policy: {POLICY_ID}")
    if not args.run:
        print("Dry run only; pass --run to write selected-feature artifacts.")
        return 0

    verify_checksums(input_dir)
    required = [input_dir / name for name in ("markers_raw.npy", "features.csv.gz", "antibodies.csv.gz")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared inputs: {missing}")

    features = pd.read_csv(input_dir / "features.csv.gz")
    antibodies = pd.read_csv(input_dir / "antibodies.csv.gz")
    matrix = np.load(input_dir / "markers_raw.npy", allow_pickle=False)
    if matrix.shape[1] != len(features):
        raise ValueError(f"Feature table/matrix mismatch: {len(features)} vs {matrix.shape}")

    paper_columns = {"Ab", "Name", "Significance", "Lineage Defining"}
    missing_paper_columns = sorted(paper_columns - set(antibodies.columns))
    if args.strict_paper_reference and missing_paper_columns:
        raise ValueError(
            "Strict paper-reference mode requested, but public workbook lacks: "
            + ", ".join(missing_paper_columns)
        )

    decisions = select_features(features, antibodies)
    selected = decisions[decisions["status"] == "selected"].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("Proxy feature policy selected no markers")
    indices = selected["source_feature_index"].to_numpy(dtype=np.int64)
    selected_matrix = np.asarray(matrix[:, indices], dtype=np.float32, order="C")
    selected.insert(0, "selected_feature_index", np.arange(len(selected), dtype=np.int64))

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        remove_stage_directory(output_dir)
    output_dir.mkdir(parents=True)
    atomic_npy(output_dir / "markers_selected.npy", selected_matrix)
    atomic_csv_gzip(output_dir / "selected_features.csv.gz", selected)
    atomic_csv_gzip(output_dir / "selection_decisions.csv.gz", decisions)

    report = {
        "created_at": now_utc(),
        "sample": args.sample,
        "policy_id": POLICY_ID,
        "fidelity": "public-workbook proxy; not the unpublished paper reference.csv",
        "missing_paper_reference_columns": missing_paper_columns,
        "input": {
            "directory": str(input_dir),
            "markers_raw_sha256": sha256_file(input_dir / "markers_raw.npy"),
            "feature_count": int(matrix.shape[1]),
            "cell_count": int(matrix.shape[0]),
        },
        "selection": {
            "selected_count": int(len(selected)),
            "excluded_count": int(len(decisions) - len(selected)),
            "excluded_by_reason": {
                str(key): int(value)
                for key, value in decisions.loc[decisions.status == "excluded", "reason"].value_counts().items()
            },
        },
    }
    atomic_json(output_dir / "selection_report.json", report)
    files = ["markers_selected.npy", "selected_features.csv.gz", "selection_decisions.csv.gz", "selection_report.json"]
    ledger = "".join(f"{sha256_file(output_dir / name)}  {name}\n" for name in files)
    (output_dir / "SHA256SUMS").write_text(ledger, encoding="utf-8")
    print(json.dumps(report["selection"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
