# Purpose of this module:
# - Validate processed dataset directories against expected file and metadata contracts.
# - Check metadata.json, obs.csv.gz, array files, and declared schema fields for consistency.
# - Return actionable validation errors so dataset preparation problems can be fixed early.

from __future__ import annotations

import csv
import json
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common.checksums import sha256_file
from common.metadata import read_metadata


@dataclass
class ValidationResult:
    scope: str
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)


def _result(scope: str, path: Path) -> ValidationResult:
    return ValidationResult(scope=scope, path=Path(path))


def _repo_root_from_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    parts = dataset_dir.parts
    if len(parts) >= 3 and parts[-3:-1] == ("datasets", "processed"):
        return dataset_dir.parents[2]
    return Path.cwd().resolve()


def _resolve_repo_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _read_obs_info(path: Path, required_columns: set[str]) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        fieldname_set = set(fieldnames)
        missing_columns = sorted(required_columns - fieldname_set)
        if missing_columns:
            errors.append(f"{path.name} missing columns: {', '.join(missing_columns)}")
        rows = sum(1 for _ in reader)
    return rows, fieldnames, errors


def _read_obs_column(path: Path, column: str) -> list[str]:
    values: list[str] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(row.get(column, ""))
    return values


def _load_npy_metadata(dataset_dir: Path, rel_path: str) -> tuple[list[int], str] | tuple[None, None]:
    array = np.load(dataset_dir / rel_path, mmap_mode="r")
    return [int(value) for value in array.shape], str(array.dtype)


def _validate_checksums_file(base_dir: Path, checksum_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not checksum_path.exists():
        return errors, warnings

    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [f"Could not read checksum file {checksum_path}: {exc}"], warnings

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"{checksum_path} line {line_number} is not '<sha256> <path>'")
            continue
        expected_hash, rel_path = parts
        rel_path = rel_path.strip()
        if rel_path.startswith("*"):
            rel_path = rel_path[1:]
        if len(expected_hash) != 64:
            errors.append(f"{checksum_path} line {line_number} has invalid sha256: {expected_hash}")
            continue
        if Path(rel_path).is_absolute():
            warnings.append(f"Skipping absolute checksum path in {checksum_path}: {rel_path}")
            continue
        file_path = base_dir / rel_path
        if not file_path.exists():
            errors.append(f"Checksum file entry is missing: {file_path}")
            continue
        if not file_path.is_file():
            warnings.append(f"Skipping checksum entry that is not a file: {file_path}")
            continue
        try:
            actual_hash = sha256_file(file_path)
        except Exception as exc:
            errors.append(f"Could not checksum {file_path}: {exc}")
            continue
        if actual_hash != expected_hash:
            errors.append(
                f"Checksum mismatch for {file_path}: expected={expected_hash}, actual={actual_hash}"
            )

    return errors, warnings


def _extra_array_records(extra_arrays: Any) -> list[tuple[str, dict[str, Any]]]:
    if not extra_arrays:
        return []
    if isinstance(extra_arrays, dict):
        records: list[tuple[str, dict[str, Any]]] = []
        for key, value in extra_arrays.items():
            if isinstance(value, dict):
                file_name = str(value.get("file") or key)
                records.append((file_name, value))
            else:
                records.append((str(key), {"file": str(key)}))
        return records
    if isinstance(extra_arrays, list):
        records = []
        for value in extra_arrays:
            if isinstance(value, dict) and isinstance(value.get("file"), str):
                records.append((str(value["file"]), value))
            elif isinstance(value, str):
                records.append((value, {"file": value}))
        return records
    return []


def _validate_obs_metadata(metadata: dict[str, Any], obs_columns: list[str]) -> list[str]:
    errors: list[str] = []
    obs_schema = metadata.get("obs_schema")
    if not isinstance(obs_schema, dict):
        return ["metadata.json field obs_schema must be a dict"]

    schema_columns = set(str(column) for column in obs_schema)
    obs_column_set = set(obs_columns)
    missing_in_obs = sorted(schema_columns - obs_column_set)
    if missing_in_obs:
        errors.append(f"obs_schema columns missing from obs.csv.gz: {', '.join(missing_in_obs)}")
    missing_in_schema = sorted(obs_column_set - schema_columns)
    if missing_in_schema:
        errors.append(f"obs.csv.gz columns missing from obs_schema: {', '.join(missing_in_schema)}")

    for column, config in obs_schema.items():
        if not isinstance(config, dict):
            errors.append(f"obs_schema.{column} must be a dict")
            continue
        if not isinstance(config.get("dtype"), str):
            errors.append(f"obs_schema.{column}.dtype must be a string")
        if not isinstance(config.get("role"), str):
            errors.append(f"obs_schema.{column}.role must be a string")
        if not isinstance(config.get("description"), str):
            errors.append(f"obs_schema.{column}.description must be a string")

    identifier_column = metadata.get("identifier_column")
    if not isinstance(identifier_column, str):
        errors.append("metadata.json field identifier_column must be a string")
    elif identifier_column not in schema_columns or identifier_column not in obs_column_set:
        errors.append(
            "identifier_column must exist in obs_schema and obs.csv.gz: "
            f"{identifier_column}"
        )

    list_fields = [
        "label_columns",
        "colorable_columns",
        "searchable_columns",
        "filterable_columns",
    ]
    for field in list_fields:
        value = metadata.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"metadata.json field {field} must be a list of strings")
            continue
        missing = sorted(set(value) - schema_columns)
        if missing:
            errors.append(f"{field} references columns missing from obs_schema: {', '.join(missing)}")

    default_color_by = metadata.get("default_color_by")
    if default_color_by is not None:
        if not isinstance(default_color_by, str):
            errors.append("default_color_by must be a string or null")
        elif default_color_by not in schema_columns:
            errors.append(f"default_color_by missing from obs_schema: {default_color_by}")
        elif default_color_by not in metadata.get("colorable_columns", []):
            errors.append(f"default_color_by must be included in colorable_columns: {default_color_by}")

    return errors


def _validate_target_file(
    metadata: dict[str, Any],
    dataset_dir: Path,
    feature_rows: int,
    obs_columns: list[str],
    obs_path: Path,
) -> list[str]:
    errors: list[str] = []
    target_file = metadata.get("target_file")
    if target_file is None:
        return errors
    if not isinstance(target_file, str):
        return ["metadata.json field target_file must be a string or null"]

    target_path = dataset_dir / target_file
    if not target_path.exists():
        return [f"Missing required file: {target_path}"]

    try:
        target = np.load(target_path, mmap_mode="r")
    except Exception as exc:
        return [f"Could not read {target_file}: {exc}"]

    expected_dtype = metadata.get("target_dtype")
    if not isinstance(expected_dtype, str):
        errors.append("metadata.json field target_dtype must be a string when target_file is set")
    else:
        actual_dtype = str(target.dtype)
        if actual_dtype != expected_dtype:
            errors.append(
                f"target_dtype mismatch: metadata={expected_dtype}, actual={actual_dtype}"
            )

    if len(target.shape) < 1:
        errors.append(f"{target_file} must have at least one dimension")
    elif int(target.shape[0]) != feature_rows:
        errors.append(
            f"{target_file} row count mismatch: target={int(target.shape[0])}, "
            f"features rows={feature_rows}"
        )

    target_source_column = metadata.get("target_source_column")
    if target_source_column is not None:
        if not isinstance(target_source_column, str):
            errors.append("metadata.json field target_source_column must be a string or null")
        elif target_source_column not in obs_columns:
            errors.append(
                f"target_source_column missing from obs.csv.gz: {target_source_column}"
            )
        elif len(target.shape) == 1:
            try:
                source_values = np.asarray(
                    [int(value) for value in _read_obs_column(obs_path, target_source_column)],
                    dtype=target.dtype,
                )
            except ValueError as exc:
                errors.append(
                    f"target_source_column contains non-integer values: {target_source_column}: {exc}"
                )
            else:
                if source_values.shape != target.shape:
                    errors.append(
                        f"target_source_column row count mismatch: "
                        f"obs={source_values.shape[0]}, target={target.shape[0]}"
                    )
                elif not np.array_equal(np.asarray(target), source_values):
                    errors.append(
                        f"{target_file} contents do not match obs.csv.gz column: "
                        f"{target_source_column}"
                    )

    return errors


def _validate_image_obs_values(obs_path: Path, obs_columns: list[str]) -> list[str]:
    errors: list[str] = []
    required_columns = {"sample_id", "split", "label", "label_name"}
    missing = sorted(required_columns - set(obs_columns))
    if missing:
        return [f"image_benchmark obs.csv.gz missing columns: {', '.join(missing)}"]

    invalid_splits: set[str] = set()
    empty_label_name_rows = 0
    with gzip.open(obs_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row.get("split", "")
            if split not in {"train", "test"}:
                invalid_splits.add(split)
            if not row.get("label_name", "").strip():
                empty_label_name_rows += 1

    if invalid_splits:
        rendered = ", ".join(repr(value) for value in sorted(invalid_splits))
        errors.append(f"split column contains values outside train/test: {rendered}")
    if empty_label_name_rows:
        errors.append(f"label_name contains empty values: {empty_label_name_rows} row(s)")

    return errors


def _validate_label_obs_values(obs_path: Path, obs_columns: list[str], kind: str) -> list[str]:
    errors: list[str] = []
    required_columns = {"sample_id", "label", "label_name"}
    missing = sorted(required_columns - set(obs_columns))
    if missing:
        return [f"{kind} obs.csv.gz missing columns: {', '.join(missing)}"]

    empty_label_name_rows = 0
    with gzip.open(obs_path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("label_name", "").strip():
                empty_label_name_rows += 1

    if empty_label_name_rows:
        errors.append(f"label_name contains empty values: {empty_label_name_rows} row(s)")
    return errors


def _validate_label_mapping(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label_mapping = metadata.get("label_mapping")
    if not isinstance(label_mapping, list) or not label_mapping:
        return ["metadata.json field label_mapping must be a non-empty list"]

    labels: list[int] = []
    label_names: list[str] = []
    for index, item in enumerate(label_mapping):
        if not isinstance(item, dict):
            errors.append(f"label_mapping[{index}] must be a dict")
            continue
        label = item.get("label")
        label_name = item.get("label_name")
        if not isinstance(label, int):
            errors.append(f"label_mapping[{index}].label must be an integer")
        else:
            labels.append(label)
        if not isinstance(label_name, str) or not label_name.strip():
            errors.append(f"label_mapping[{index}].label_name must be a non-empty string")
        else:
            label_names.append(label_name)

    if labels and sorted(labels) != list(range(len(labels))):
        errors.append("label_mapping labels must be contiguous integers starting at 0")
    if len(set(label_names)) != len(label_names):
        errors.append("label_mapping label_name values must be unique")

    n_classes = metadata.get("n_classes")
    if n_classes is not None:
        if not isinstance(n_classes, int):
            errors.append("metadata.json field n_classes must be an integer")
        elif n_classes != len(label_mapping):
            errors.append(
                f"n_classes mismatch: metadata={n_classes}, label_mapping={len(label_mapping)}"
            )

    return errors


def _validate_reference_embedding(
    metadata: dict[str, Any],
    dataset_dir: Path,
    feature_rows: int,
) -> list[str]:
    errors: list[str] = []
    reference_file = metadata.get("reference_embedding_file")
    if reference_file is None:
        return errors
    if not isinstance(reference_file, str):
        return ["metadata.json field reference_embedding_file must be a string or null"]

    reference_path = dataset_dir / reference_file
    if not reference_path.exists():
        return [f"Missing required file: {reference_path}"]

    try:
        reference = np.load(reference_path, mmap_mode="r")
    except Exception as exc:
        return [f"Could not read {reference_file}: {exc}"]

    reference_shape = [int(value) for value in reference.shape]
    expected_shape = metadata.get("reference_embedding_shape")
    if expected_shape is not None and expected_shape != reference_shape:
        errors.append(
            f"reference_embedding_shape mismatch: metadata={expected_shape}, actual={reference_shape}"
        )
    if len(reference.shape) < 1:
        errors.append(f"{reference_file} must have at least one dimension")
    elif int(reference.shape[0]) != feature_rows:
        errors.append(
            f"{reference_file} row count mismatch: reference={int(reference.shape[0])}, "
            f"features rows={feature_rows}"
        )
    return errors


def _validate_reference_embeddings(
    metadata: dict[str, Any],
    dataset_dir: Path,
    feature_rows: int,
) -> list[str]:
    errors: list[str] = []
    references = metadata.get("reference_embeddings")
    if not references:
        return errors
    if not isinstance(references, dict):
        return ["metadata.json field reference_embeddings must be a dict when present"]

    for name, record in references.items():
        if not isinstance(record, dict):
            errors.append(f"reference_embeddings.{name} must be a dict")
            continue
        rel_path = record.get("file")
        if not isinstance(rel_path, str):
            errors.append(f"reference_embeddings.{name}.file must be a string")
            continue
        path = dataset_dir / rel_path
        if not path.exists():
            errors.append(f"Missing reference embedding file: {path}")
            continue
        try:
            array = np.load(path, mmap_mode="r")
        except Exception as exc:
            errors.append(f"Could not read reference embedding {rel_path}: {exc}")
            continue
        shape = [int(value) for value in array.shape]
        expected_shape = record.get("shape")
        if expected_shape is not None and expected_shape != shape:
            errors.append(
                f"reference_embeddings.{name}.shape mismatch: "
                f"metadata={expected_shape}, actual={shape}"
            )
        if len(array.shape) < 1 or int(array.shape[0]) != feature_rows:
            errors.append(
                f"reference embedding {rel_path} row count mismatch: "
                f"reference={shape[0] if shape else None}, features rows={feature_rows}"
            )
    return errors


def validate_processed_dataset(dataset_dir: Path) -> ValidationResult:
    dataset_dir = Path(dataset_dir)
    result = _result("processed", dataset_dir)
    errors = result.errors
    warnings = result.warnings

    if not dataset_dir.exists():
        errors.append(f"Dataset directory does not exist: {dataset_dir}")
        return result
    if not dataset_dir.is_dir():
        errors.append(f"Dataset path is not a directory: {dataset_dir}")
        return result

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        errors.append(f"Missing required file: {metadata_path}")
        return result

    try:
        metadata: dict[str, Any] = read_metadata(metadata_path)
    except Exception as exc:
        errors.append(f"Could not read metadata.json: {exc}")
        return result

    primary_array = metadata.get("primary_array")
    obs_file = metadata.get("obs_file")
    required: list[str] = []
    if isinstance(primary_array, str):
        required.append(primary_array)
    else:
        errors.append("metadata.json field primary_array must be a string")
    if isinstance(obs_file, str):
        required.append(obs_file)
    elif obs_file is not None:
        errors.append("metadata.json field obs_file must be a string")

    extra_records = _extra_array_records(metadata.get("extra_arrays"))
    if metadata.get("extra_arrays") and not extra_records:
        errors.append("metadata.json field extra_arrays must be a dict or list")
    for rel_path, _record in extra_records:
        required.append(rel_path)

    target_file = metadata.get("target_file")
    if target_file is not None:
        if isinstance(target_file, str):
            required.append(target_file)
        else:
            errors.append("metadata.json field target_file must be a string or null")

    reference_file = metadata.get("reference_embedding_file")
    if reference_file is not None:
        if isinstance(reference_file, str):
            required.append(reference_file)
        else:
            errors.append("metadata.json field reference_embedding_file must be a string or null")

    pseudotime_file = metadata.get("pseudotime_file")
    if pseudotime_file is not None:
        if isinstance(pseudotime_file, str):
            required.append(pseudotime_file)
        else:
            errors.append("metadata.json field pseudotime_file must be a string or null")

    source_indices_file = metadata.get("source_indices_file")
    if source_indices_file is not None:
        if isinstance(source_indices_file, str):
            required.append(source_indices_file)
        else:
            errors.append("metadata.json field source_indices_file must be a string or null")

    summary_file = metadata.get("summary_file")
    if summary_file is not None:
        if isinstance(summary_file, str):
            required.append(summary_file)
        else:
            errors.append("metadata.json field summary_file must be a string or null")

    for rel_path in required:
        path = dataset_dir / rel_path
        if not path.exists():
            errors.append(f"Missing required file: {path}")

    checksum_path = dataset_dir / "checksums.txt"
    if checksum_path.exists():
        checksum_errors, checksum_warnings = _validate_checksums_file(dataset_dir, checksum_path)
        errors.extend(checksum_errors)
        warnings.extend(checksum_warnings)

    if errors:
        return result

    feature_path = dataset_dir / str(primary_array)
    try:
        features = np.load(feature_path, mmap_mode="r")
    except Exception as exc:
        errors.append(f"Could not read {primary_array}: {exc}")
        return result

    expected_shape = metadata.get("feature_shape")
    actual_shape = list(int(v) for v in features.shape)
    if expected_shape != actual_shape:
        errors.append(
            f"feature_shape mismatch: metadata={expected_shape}, actual={actual_shape}"
        )

    expected_dtype = metadata.get("feature_dtype")
    actual_dtype = str(features.dtype)
    if expected_dtype != actual_dtype:
        errors.append(
            f"feature_dtype mismatch: metadata={expected_dtype}, actual={actual_dtype}"
        )

    obs_columns: list[str] = []
    if isinstance(obs_file, str):
        obs_path = dataset_dir / obs_file
        try:
            required_columns = {"sample_id"}
            if metadata.get("kind") == "word_embedding":
                required_columns.add("token")
            if metadata.get("kind") == "image_benchmark":
                required_columns.update({"split", "label", "label_name"})
            if metadata.get("kind") == "tabular_ml":
                required_columns.update({"label", "label_name"})
            obs_rows, obs_columns, obs_errors = _read_obs_info(obs_path, required_columns)
            errors.extend(obs_errors)
            errors.extend(_validate_obs_metadata(metadata, obs_columns))
            if metadata.get("kind") == "image_benchmark":
                errors.extend(_validate_image_obs_values(obs_path, obs_columns))
            if metadata.get("kind") == "tabular_ml":
                errors.extend(_validate_label_obs_values(obs_path, obs_columns, "tabular_ml"))
                errors.extend(_validate_label_mapping(metadata))
            if len(actual_shape) >= 1 and obs_rows != actual_shape[0]:
                errors.append(
                    f"obs row count mismatch: obs={obs_rows}, features rows={actual_shape[0]}"
                )
            if len(actual_shape) >= 1:
                errors.extend(
                    _validate_target_file(
                        metadata,
                        dataset_dir,
                        int(actual_shape[0]),
                        obs_columns,
                        obs_path,
                    )
                )
        except Exception as exc:
            errors.append(f"Could not validate {obs_file}: {exc}")
    elif metadata.get("target_file") is not None:
        errors.append("metadata.json field obs_file is required when target_file is set")

    if len(actual_shape) >= 1:
        errors.extend(_validate_reference_embedding(metadata, dataset_dir, int(actual_shape[0])))
        errors.extend(_validate_reference_embeddings(metadata, dataset_dir, int(actual_shape[0])))

    if len(actual_shape) >= 1 and isinstance(pseudotime_file, str):
        try:
            pseudotime = np.load(dataset_dir / pseudotime_file, mmap_mode="r")
        except Exception as exc:
            errors.append(f"Could not read {pseudotime_file}: {exc}")
        else:
            if len(pseudotime.shape) != 1:
                errors.append(f"{pseudotime_file} must be a 1D array")
            elif int(pseudotime.shape[0]) != int(actual_shape[0]):
                errors.append(
                    f"{pseudotime_file} row count mismatch: "
                    f"pseudotime={int(pseudotime.shape[0])}, features rows={actual_shape[0]}"
                )

    if len(actual_shape) >= 1 and isinstance(source_indices_file, str):
        try:
            source_indices = np.load(dataset_dir / source_indices_file, mmap_mode="r")
        except Exception as exc:
            errors.append(f"Could not read {source_indices_file}: {exc}")
        else:
            if len(source_indices.shape) != 1:
                errors.append(f"{source_indices_file} must be a 1D array")
            elif int(source_indices.shape[0]) != int(actual_shape[0]):
                errors.append(
                    f"{source_indices_file} row count mismatch: "
                    f"source_indices={int(source_indices.shape[0])}, "
                    f"features rows={actual_shape[0]}"
                )
            expected_dtype = metadata.get("source_indices_dtype", "int64")
            if str(source_indices.dtype) != str(expected_dtype):
                errors.append(
                    f"{source_indices_file} dtype mismatch: "
                    f"metadata/default={expected_dtype}, actual={source_indices.dtype}"
                )

    if isinstance(summary_file, str):
        try:
            json.loads((dataset_dir / summary_file).read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Could not parse {summary_file}: {exc}")

    for rel_path, record in extra_records:
        try:
            array_shape, array_dtype = _load_npy_metadata(dataset_dir, rel_path)
        except Exception as exc:
            errors.append(f"Could not read extra array {rel_path}: {exc}")
            continue

        expected_extra_shape = record.get("shape")
        if expected_extra_shape is not None and expected_extra_shape != array_shape:
            errors.append(
                f"{rel_path} shape mismatch: metadata={expected_extra_shape}, actual={array_shape}"
            )

        expected_extra_dtype = record.get("dtype")
        if expected_extra_dtype is not None and expected_extra_dtype != array_dtype:
            errors.append(
                f"{rel_path} dtype mismatch: metadata={expected_extra_dtype}, actual={array_dtype}"
            )

        top_level_shape_fields = {
            "queries.npy": "query_shape",
            "neighbors.npy": "neighbor_shape",
            "distances.npy": "distance_shape",
        }
        shape_field = top_level_shape_fields.get(rel_path)
        if shape_field and metadata.get(shape_field) != array_shape:
            errors.append(
                f"{shape_field} mismatch for {rel_path}: "
                f"metadata={metadata.get(shape_field)}, actual={array_shape}"
            )

    raw_path = metadata.get("raw_path")
    if raw_path is not None and not isinstance(raw_path, str):
        warnings.append("metadata.json field raw_path is not a string; skipped in processed scope")
    raw_paths = metadata.get("raw_paths")
    if raw_paths is not None:
        if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
            warnings.append("metadata.json field raw_paths is not a list of strings; skipped in processed scope")

    dataset_id = metadata.get("dataset_id")
    if dataset_id != dataset_dir.name:
        errors.append(
            f"dataset_id mismatch: metadata={dataset_id}, directory={dataset_dir.name}"
        )

    return result


def _summarize_raw_directory(path: Path) -> tuple[int, int]:
    file_count = 0
    total_size = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            try:
                total_size += item.stat().st_size
            except OSError:
                pass
    return file_count, total_size


def validate_raw_path(path: Path) -> ValidationResult:
    path = Path(path)
    result = _result("raw", path)
    errors = result.errors
    warnings = result.warnings
    info = result.info

    if not path.exists():
        errors.append(f"Raw path does not exist: {path}")
        return result

    if path.is_file():
        info.append(f"raw file exists: {path} ({path.stat().st_size} bytes)")
        return result

    if not path.is_dir():
        errors.append(f"Raw path is neither a file nor a directory: {path}")
        return result

    metadata_names = {
        "zenodo_metadata.json",
        "collection_metadata.json",
        "inspect_manifest.json",
        "download_manifest.json",
        "manifest.json",
        "metadata.json",
    }
    found_metadata = [item for item in sorted(path.iterdir()) if item.name in metadata_names]
    for metadata_path in found_metadata:
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            errors.append(f"Could not parse raw metadata JSON {metadata_path}: {exc}")

    checksum_path = path / "checksums.txt"
    if checksum_path.exists():
        checksum_errors, checksum_warnings = _validate_checksums_file(path, checksum_path)
        errors.extend(checksum_errors)
        warnings.extend(checksum_warnings)

    file_count, total_size = _summarize_raw_directory(path)
    metadata_label = ", ".join(item.name for item in found_metadata) if found_metadata else "none"
    info.append(
        f"raw summary: path exists; files={file_count}; total_size={total_size} bytes; "
        f"metadata={metadata_label}"
    )
    if not found_metadata and not checksum_path.exists():
        warnings.append("raw path has no known metadata or checksums.txt; basic existence summary only")

    return result


def _raw_paths_from_metadata(metadata: dict[str, Any], repo_root: Path) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    errors: list[str] = []

    raw_path = metadata.get("raw_path")
    raw_paths = metadata.get("raw_paths")
    if isinstance(raw_path, str):
        paths.append(_resolve_repo_path(raw_path, repo_root))
    elif raw_path is not None:
        errors.append("metadata.json field raw_path must be a string or null for full validation")

    if raw_paths is not None:
        if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
            errors.append("metadata.json field raw_paths must be a list of strings for full validation")
        else:
            paths.extend(_resolve_repo_path(item, repo_root) for item in raw_paths)

    return paths, errors


def validate_full_dataset(dataset_dir: Path) -> ValidationResult:
    dataset_dir = Path(dataset_dir)
    result = _result("full", dataset_dir)
    processed_result = validate_processed_dataset(dataset_dir)
    result.errors.extend(processed_result.errors)
    result.warnings.extend(processed_result.warnings)
    if processed_result.errors:
        return result

    metadata_path = dataset_dir / "metadata.json"
    try:
        metadata: dict[str, Any] = read_metadata(metadata_path)
    except Exception as exc:
        result.errors.append(f"Could not read metadata.json for full validation: {exc}")
        return result

    repo_root = _repo_root_from_dataset_dir(dataset_dir)
    raw_paths, raw_path_errors = _raw_paths_from_metadata(metadata, repo_root)
    result.errors.extend(raw_path_errors)
    if not raw_paths:
        result.errors.append("full validation requires metadata raw_path or raw_paths")
        return result

    seen: set[Path] = set()
    for raw_path in raw_paths:
        if raw_path in seen:
            continue
        seen.add(raw_path)
        raw_result = validate_raw_path(raw_path)
        for error in raw_result.errors:
            result.errors.append(f"raw provenance failed for {raw_path}: {error}")
        for warning in raw_result.warnings:
            result.warnings.append(f"raw provenance for {raw_path}: {warning}")
        for message in raw_result.info:
            result.info.append(f"raw provenance for {raw_path}: {message}")

    return result
