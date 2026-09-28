from __future__ import annotations

import json
import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_obs_schema(columns: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name, config in columns.items():
        if "dtype" not in config:
            raise ValueError(f"obs schema column {name!r} is missing dtype")
        if "role" not in config:
            raise ValueError(f"obs schema column {name!r} is missing role")
        schema[name] = {
            "dtype": str(config["dtype"]),
            "role": str(config["role"]),
            "description": str(config.get("description", "")),
        }
    return schema


def obs_metadata_fields(
    *,
    obs_schema: Mapping[str, Mapping[str, Any]],
    identifier_column: str,
    default_color_by: str | None = None,
    label_columns: list[str] | None = None,
    colorable_columns: list[str] | None = None,
    searchable_columns: list[str] | None = None,
    filterable_columns: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "obs_schema": build_obs_schema(obs_schema),
        "identifier_column": identifier_column,
        "default_color_by": default_color_by,
        "label_columns": label_columns or [],
        "colorable_columns": colorable_columns or [],
        "searchable_columns": searchable_columns or [],
        "filterable_columns": filterable_columns or [],
    }


def infer_obs_schema(path: Path) -> dict[str, dict[str, Any]]:
    opener = gzip.open if Path(path).name.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
    return build_obs_schema(
        {
            column: {
                "dtype": "string",
                "role": "metadata",
                "description": f"Column {column} from the observation table.",
            }
            for column in columns
        }
    )
