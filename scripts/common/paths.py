"""Repository paths shared by the experiment scripts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the ibUMAP repository root (the directory with pyproject.toml and src/ibumap)."""
    current = Path(start if start is not None else Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "ibumap").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find the ibUMAP repository root from {current}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SCRIPTS_ROOT = REPO_ROOT / "scripts"
DATASETS_ROOT = REPO_ROOT / "datasets"
PROCESSED_ROOT = DATASETS_ROOT / "processed"
RAW_ROOT = DATASETS_ROOT / "raw"
RAW_CACHE_ROOT = DATASETS_ROOT / "raw_cache"
DATASET_CACHE_ROOT = DATASETS_ROOT / "cache"
CATALOG_PATH = DATASETS_ROOT / "catalog.json"
IBUMAP_SRC = REPO_ROOT / "src"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def repo_relative(path: str | Path, repo_root: str | Path | None = None) -> str:
    """Path relative to the repository root when possible (keeps outputs machine independent)."""
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def resolve_repo_path(path: str | Path | None, *, repo_root: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    if value.is_absolute():
        return value
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    return (root / value).resolve()


def ensure_ibumap_importable() -> Path | None:
    """Make ``import ibumap`` work.

    An installed package (``pip install -e .``, see environments/README.md) is
    used as is. Otherwise ``src/`` is put on ``sys.path``; the optional compiled
    extensions are then unavailable and the CPU path uses its pure-Python fallback.
    """
    if "ibumap" in sys.modules or importlib.util.find_spec("ibumap") is not None:
        return None
    if str(IBUMAP_SRC) not in sys.path:
        sys.path.insert(0, str(IBUMAP_SRC))
    return IBUMAP_SRC


def add_scripts_root_to_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    scripts = root / "scripts"
    if scripts.exists() and str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return scripts
