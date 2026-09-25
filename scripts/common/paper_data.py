"""Write experiment summaries in the layout read by ``scripts/paper`` (``paper/data/<group>/``).

Each experiment's export step writes one group directory (``e2e_benchmark``,
``mechanism``, ``psweep``, ``braque``, ``safeguards``). A full rerun collects the
five groups in one directory, by default ``paper/rerun/``, and then runs

    python scripts/paper/make_all.py --data-dir paper/rerun

CSV files are gzip-compressed deterministically (no file name, mtime 0), so an
unchanged summary always produces identical bytes.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from common.paths import REPO_ROOT

DEFAULT_RERUN_ROOT = REPO_ROOT / "paper" / "rerun"


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Rows as strings (``.csv`` or ``.csv.gz``), so values can be copied without reformatting."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def gzip_bytes(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as handle:
        handle.write(payload)
    return buffer.getvalue()


def write_csv_gz(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    """Write rows (missing keys empty, extra keys rejected); return the SHA-256 of the file."""
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    data = gzip_bytes(text.getvalue().encode("utf-8"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_json(path: str | Path, payload: Any) -> str:
    data = (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_npz(path: str | Path, arrays: Mapping[str, Any]) -> str:
    """Write arrays as a deterministic ``.npz`` (sorted members, fixed timestamps); return its SHA-256.

    ``numpy.savez`` stamps each member with the current time, so two exports of the
    same arrays would differ. The file loads with ``numpy.load`` as usual.
    """
    import zipfile

    import numpy as np

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    data = buffer.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
