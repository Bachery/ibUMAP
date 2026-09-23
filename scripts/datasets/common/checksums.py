from __future__ import annotations

from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(
    output_path: Path,
    files: Iterable[Path],
    base_dir: Path | None = None,
) -> dict[str, str]:
    output_path = Path(output_path)
    base_dir = Path(base_dir) if base_dir is not None else output_path.parent

    records: dict[str, str] = {}
    for file_path in files:
        file_path = Path(file_path)
        rel_path = file_path.relative_to(base_dir).as_posix()
        records[rel_path] = sha256_file(file_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rel_path in sorted(records):
            handle.write(f"{records[rel_path]}  {rel_path}\n")

    return records
