from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
H5AD_ROOT = REPO_ROOT / "datasets/raw/whole_mouse_brain_merfish/h5ad"
RAW_FILE_NAMES = (
    "WB_MERFISH_animal1_coronal.h5ad",
    "WB_MERFISH_animal2_coronal.h5ad",
    "WB_MERFISH_animal3_sagittal.h5ad",
    "WB_MERFISH_animal4_sagittal.h5ad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the structural metadata of whole mouse brain MERFISH H5AD files."
    )
    parser.add_argument("--all", action="store_true", help="Inspect all downloaded H5AD files.")
    parser.add_argument("--list", action="store_true", help="List expected raw H5AD files.")
    parser.add_argument("--dataset", help="Inspect one H5AD file by name or stem.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to write instead of stdout.")
    return parser.parse_args()


def decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def list_h5ad_files() -> list[Path]:
    return [H5AD_ROOT / name for name in RAW_FILE_NAMES if (H5AD_ROOT / name).exists()]


def select_h5ad_files(args: argparse.Namespace) -> list[Path]:
    files = list_h5ad_files()
    if args.list:
        return files
    if args.all:
        return files
    if args.dataset:
        query = args.dataset
        query_stem = Path(query).stem
        selected = [path for path in files if path.name == query or path.stem == query_stem]
        if not selected:
            raise FileNotFoundError(f"No downloaded H5AD matched {query!r}. Use --list to inspect names.")
        return selected
    raise ValueError("Provide --all, --list, or --dataset")


def summary_for(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        x_group = handle["X"]
        shape = [int(value) for value in x_group.attrs["shape"]]
        nnz = int(x_group["data"].shape[0])
        root_keys = sorted(str(key) for key in handle.keys())
        x_encoding = str(decode_scalar(x_group.attrs.get("encoding-type", "")))
        x_data_dtype = str(x_group["data"].dtype)
        obs_group = handle["obs"]
        obs_index = str(decode_scalar(obs_group.attrs.get("_index", "")))
        obs_columns: dict[str, dict[str, Any]] = {}
        for name, obj in obs_group.items():
            if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(obj.keys()):
                obs_columns[name] = {
                    "encoding": "categorical",
                    "n_categories": int(obj["categories"].shape[0]),
                    "codes_dtype": str(obj["codes"].dtype),
                }
            elif isinstance(obj, h5py.Dataset):
                obs_columns[name] = {
                    "encoding": "dataset",
                    "shape": [int(value) for value in obj.shape],
                    "dtype": str(obj.dtype),
                }
            else:
                obs_columns[name] = {"encoding": type(obj).__name__}

        embeddings: dict[str, dict[str, Any]] = {}
        for name, obj in handle["obsm"].items():
            if isinstance(obj, h5py.Dataset):
                embeddings[name] = {
                    "shape": [int(value) for value in obj.shape],
                    "dtype": str(obj.dtype),
                }
        raw_x = handle["raw/X"]
        raw_x_encoding = str(decode_scalar(raw_x.attrs.get("encoding-type", "")))
        raw_x_shape = [int(value) for value in raw_x.attrs["shape"]]

    return {
        "file": path.name,
        "size_bytes": int(path.stat().st_size),
        "root_keys": root_keys,
        "X": {
            "encoding_type": x_encoding,
            "shape": shape,
            "nnz": nnz,
            "density": float(nnz / (shape[0] * shape[1])),
            "data_dtype": x_data_dtype,
        },
        "obs_index": obs_index,
        "obs_columns": obs_columns,
        "obsm": embeddings,
        "raw_X": {
            "encoding_type": raw_x_encoding,
            "shape": raw_x_shape,
        },
    }


def main() -> int:
    args = parse_args()
    try:
        selected = select_h5ad_files(args)
        if args.list:
            for path in selected:
                print(path.name)
            return 0
        if not selected:
            raise FileNotFoundError(f"No expected H5AD files found under {H5AD_ROOT}")
        payload = {"datasets": [summary_for(path) for path in selected]}
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(f"Wrote {args.output}")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
