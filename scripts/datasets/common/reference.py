"""Reference records of the processed datasets used in the paper.

``datasets/reference/<dataset_id>/`` holds, for every benchmark dataset, the
``metadata.json`` written when the paper's copy was prepared and a
``checksums.txt`` with the SHA-256 of its data files. For ``*.gz`` files the
hash covers the decompressed content, because gzip headers carry a timestamp.
``datasets/processed/`` itself only holds prepared output and is not tracked.

:func:`compare_to_reference` checks a prepared dataset against its record:
every data file must be present with the recorded content hash, and the
structural fields of ``metadata.json`` (shapes, dtypes, file names) must agree.
Other metadata fields, such as ``generated_at``, may differ and are reported as
information only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .checksums import sha256_file
from .validation import ValidationResult


REFERENCE_HEADER = (
    "# SHA-256 of the data files of the processed dataset used in the paper.\n"
    "# For *.gz files the hash is of the decompressed content (gzip headers carry a\n"
    "# timestamp). metadata.json is compared field by field, not byte for byte.\n"
)
UNCHECKED_FILES = {"metadata.json", "checksums.txt"}
STRUCTURAL_KEYS = (
    "dataset_id",
    "family",
    "kind",
    "primary_array",
    "feature_shape",
    "feature_dtype",
    "identifier_column",
    "obs_file",
    "target_file",
    "target_dtype",
    "label_columns",
    "metric",
)


def content_sha256(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    """SHA-256 of a file, or of its decompressed content for ``*.gz`` files."""
    path = Path(path)
    if not path.name.endswith(".gz"):
        return sha256_file(path, chunk_size=chunk_size)
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference_checksums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, rel_path = stripped.split(maxsplit=1)
        records[rel_path.strip()] = digest
    return records


def write_reference_checksums(path: Path, records: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{records[rel]}  {rel}\n" for rel in sorted(records)]
    path.write_text(REFERENCE_HEADER + "".join(lines), encoding="utf-8")


def _data_files(dataset_dir: Path) -> Iterable[str]:
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name not in UNCHECKED_FILES and not path.name.startswith("."):
            yield path.relative_to(dataset_dir).as_posix()


def compare_to_reference(dataset_dir: Path, reference_dir: Path) -> ValidationResult:
    dataset_dir, reference_dir = Path(dataset_dir), Path(reference_dir)
    result = ValidationResult(scope="reference", path=dataset_dir)
    checksum_path = reference_dir / "checksums.txt"
    if not checksum_path.exists():
        result.errors.append(f"No reference record: {checksum_path}")
        return result

    records = read_reference_checksums(checksum_path)
    matched = 0
    for rel_path, expected in records.items():
        file_path = dataset_dir / rel_path
        if not file_path.is_file():
            result.errors.append(f"Missing file listed in the reference: {rel_path}")
            continue
        actual = content_sha256(file_path)
        if actual != expected:
            result.errors.append(f"Content differs from the reference: {rel_path}")
        else:
            matched += 1
    extra = [rel for rel in _data_files(dataset_dir) if rel not in records]
    if extra:
        result.warnings.append("Files not in the reference: " + ", ".join(extra))

    reference_meta_path = reference_dir / "metadata.json"
    meta_path = dataset_dir / "metadata.json"
    if reference_meta_path.exists() and meta_path.exists():
        reference_meta = json.loads(reference_meta_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in STRUCTURAL_KEYS:
            if key in reference_meta and meta.get(key) != reference_meta[key]:
                result.errors.append(
                    f"metadata.json field {key!r} differs: {meta.get(key)!r} != reference {reference_meta[key]!r}"
                )
        other = sorted(
            key for key in set(meta) | set(reference_meta)
            if key not in STRUCTURAL_KEYS and meta.get(key) != reference_meta.get(key)
        )
        if other:
            result.info.append("metadata.json fields that differ from the reference (not checked): " + ", ".join(other))
    elif not meta_path.exists():
        result.errors.append("metadata.json is missing")

    result.info.append(f"{matched}/{len(records)} data files match the reference")
    return result
