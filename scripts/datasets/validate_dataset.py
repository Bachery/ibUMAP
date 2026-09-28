"""
Validate standardized dataset assets.

By default this script validates processed datasets only, so it can run on
benchmark machines that do not keep large raw data files. Raw-source validation
and end-to-end provenance validation are available only through explicit
validation scopes.

``--reference`` additionally compares a processed dataset with the record of
the copy used in the paper, ``datasets/reference/<dataset_id>/`` (data-file
content hashes and the structural fields of metadata.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common.validation import (
    ValidationResult,
    validate_full_dataset,
    validate_processed_dataset,
    validate_raw_path,
)
from common.reference import compare_to_reference


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "datasets/processed"
DEFAULT_RAW_ROOT = REPO_ROOT / "datasets/raw"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "datasets/reference"
VALID_SCOPES = ("processed", "raw", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset directories by scope.")
    parser.add_argument("dataset_dir", nargs="?", type=Path)
    parser.add_argument(
        "--scope",
        choices=VALID_SCOPES,
        default="processed",
        help="Validation scope. Defaults to processed.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all processed datasets by default, or all raw children with --scope raw.",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Also compare each processed dataset with its record in datasets/reference/.",
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--all-processed",
        action="store_true",
        help="Alias for --scope processed --all.",
    )
    parser.add_argument(
        "--all-raw",
        action="store_true",
        help="Alias for --scope raw --all.",
    )
    return parser.parse_args()


def _iter_all_dataset_dirs() -> list[Path]:
    if not DEFAULT_PROCESSED_ROOT.exists():
        return []
    return sorted(path.parent for path in DEFAULT_PROCESSED_ROOT.glob("*/metadata.json"))


def _iter_all_raw_paths() -> list[Path]:
    if not DEFAULT_RAW_ROOT.exists():
        return []
    return sorted(DEFAULT_RAW_ROOT.iterdir())


def _validate_one(path: Path, scope: str) -> ValidationResult:
    if scope == "processed":
        return validate_processed_dataset(path)
    if scope == "raw":
        return validate_raw_path(path)
    if scope == "full":
        return validate_full_dataset(path)
    raise ValueError(f"Unsupported validation scope: {scope}")


def _print_result(result: ValidationResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status} [{result.scope}] {result.path}")
    for message in result.info:
        print(f"INFO [{result.scope}] {message}")
    for warning in result.warnings:
        print(f"WARNING [{result.scope}] {warning}")
    for error in result.errors:
        print(f"ERROR [{result.scope}] {error}")


def main() -> int:
    args = parse_args()
    if args.all_processed:
        args.scope = "processed"
        args.all = True
    if args.all_raw:
        args.scope = "raw"
        args.all = True

    if args.all:
        if args.scope == "raw":
            dataset_dirs = _iter_all_raw_paths()
            search_root = DEFAULT_RAW_ROOT
        else:
            dataset_dirs = _iter_all_dataset_dirs()
            search_root = DEFAULT_PROCESSED_ROOT
        if not dataset_dirs:
            print(f"FAIL [{args.scope}] no datasets found under {search_root}")
            return 1
    elif args.dataset_dir is not None:
        dataset_dirs = [args.dataset_dir]
    else:
        print("FAIL provide a dataset directory or --all")
        return 2

    if args.reference and args.scope == "raw":
        print("FAIL --reference applies to processed datasets, not to --scope raw")
        return 2

    results = []
    for dataset_dir in dataset_dirs:
        result = _validate_one(dataset_dir, args.scope)
        _print_result(result)
        results.append(result)
        if args.reference:
            reference = compare_to_reference(dataset_dir, args.reference_root / Path(dataset_dir).resolve().name)
            _print_result(reference)
            results.append(reference)

    if args.all:
        passed = sum(1 for result in results if result.ok)
        failed = sum(1 for result in results if not result.ok)
        warnings = sum(len(result.warnings) for result in results)
        print(f"SUMMARY [{args.scope}{'+reference' if args.reference else ''}] "
              f"passed={passed} failed={failed} warnings={warnings}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
