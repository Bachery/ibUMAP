"""JSON I/O and run provenance (package versions, git commit, timestamps)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

from common.paths import REPO_ROOT

DEFAULT_PACKAGES = ("ibumap", "numpy", "scipy", "numba", "umap-learn", "pynndescent", "pyFFTW", "scikit-learn")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object; a missing, unreadable or non-object file gives ``{}``."""
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write sorted, indented JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
                             encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def package_versions(distributions: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    return {distribution: package_version(distribution) for distribution in distributions}


def git_commit(repo_root: str | Path | None = None) -> str | None:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                                   text=True)
    except Exception:
        return None
    return completed.stdout.strip() or None


def runtime_metadata(
    *,
    repo_root: str | Path | None = None,
    packages: Iterable[str] = DEFAULT_PACKAGES,
) -> dict[str, Any]:
    return {
        "created_at": now_utc(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": git_commit(repo_root),
        "package_versions": package_versions(packages),
    }
