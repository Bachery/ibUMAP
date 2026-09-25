"""Shared helpers of the BRAQUE repeatability case study (paper Section 6).

Four CPU profiles (umap-learn and ibUMAP, each seeded and unseeded) are run five
times in fresh worker processes on one frozen BRAQUE input; HDBSCAN clusters each
embedding. Every batch lives in results/repeatability/<run-id>/ and is sealed
with SHA256SUMS ledgers that later stages verify.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
REPO = CASE.parents[1]
SRC = REPO / "src"
OUTPUT = CASE / "results" / "repeatability"
PROTOCOL = "braque-repeatability-v1"
# SHA-256 of data/processed/L2/lns/umap_input.npy (56,962 cells x 62 features) used for the paper.
PAPER_INPUT_SHA256 = "8e3698f8c350dac7489edf2be8257c6d4c1a499d1ba6897339f23c92bf84abb8"
PROFILES = ("umap_unseeded", "umap_seeded", "ibumap_unseeded", "ibumap_seeded")
DISPLAY = dict(zip(PROFILES, ("UMAP unseeded", "UMAP seeded", "ibUMAP unseeded", "ibUMAP seeded")))
LIMITATIONS = [
    "Public BRAQUE L2 data with public-workbook proxy feature selection; not an exact paper reproduction.",
    "No verified spatial coordinates or expert phenotype ground truth; changes concern computational partitions.",
    "Same-device, fixed-environment repeatability does not establish cross-seed robustness or biological correctness.",
    "Unseeded reruns do not isolate thread races from random graph construction, initialization, or optimization.",
    "Matched assignment disagreement includes cluster splits/merges; it is not a misclassification rate.",
    "Pairwise comparisons share runs and are not independent experimental replicates.",
    "Seeded cost ratios include graph/thread execution policies; they do not isolate the seed alone.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("IDs must be 1–96 letters/digits, underscores, dots or hyphens, starting with a letter/digit")
    return value


def run_root(run_id):
    root = OUTPUT / safe_name(run_id)
    if root.is_symlink() or root.resolve().parent != OUTPUT.resolve():
        raise ValueError("Run directory must be inside results/repeatability")
    return root


def seal(directory, names):
    (directory / "SHA256SUMS").write_text("".join(f"{sha(directory / n)}  {n}\n" for n in names))


def verify(directory):
    directory = Path(directory)
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Invalid checksum path: {name}")
        if sha(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    packages = {}
    for name in ("ibumap", "numpy", "scipy", "scikit-learn", "umap-learn", "pynndescent", "numba", "llvmlite", "hdbscan", "pyfftw", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu, memory = platform.processor(), None
    if Path("/proc/cpuinfo").exists():
        cpu = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")), cpu)
        memory = next((x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemTotal:")), None)
    elif platform.system() == "Darwin":
        cpu = command(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = command(["sysctl", "-n", "hw.memsize"])
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu_model": cpu, "logical_cpus": os.cpu_count(), "host_memory": memory,
            "python": platform.python_version(), "packages": packages}


def source_record():
    paths = sorted((SRC / "ibumap").rglob("*.py"))
    paths += [Path(__file__), CASE / "scripts" / "05_run_repeatability.py"]
    files = {str(p.relative_to(REPO)): sha(p) for p in paths}
    return {"sha256": digest(files), "files": files,
            "git_commit": command(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "git_status": command(["git", "-C", str(REPO), "status", "--short"])}


def load_experiment(root):
    d = read_json(root / "experiment.json")
    if d["protocol"] != PROTOCOL or digest(d["contract"]) != d["contract_sha256"]:
        raise ValueError("Unknown protocol or modified experiment contract")
    verify(root / "inputs")
    for name, expected in d["input_sha256s"].items():
        if sha(root / "inputs" / name) != expected:
            raise ValueError(f"Frozen experiment input changed: {name}")
    return d


def completed(root, profile, repeat, contract_sha):
    base = root / "runs" / profile / f"repeat_{repeat:03d}"
    successes = []
    for path in sorted(base.glob("attempt_*/run.json")):
        d = read_json(path)
        if d.get("status") == "completed":
            # A worker killed between writing metadata and sealing its files is an
            # interrupted attempt, not a reusable success. Preserve it and retry.
            if not (path.parent / "SHA256SUMS").is_file():
                continue
            if d["contract_sha256"] != contract_sha or d["profile"] != profile or d["repeat"] != repeat:
                raise ValueError(f"Run identity mismatch: {path}")
            verify(path.parent)
            successes.append((path.parent, d))
    if len(successes) > 1:
        raise ValueError(f"Multiple successful attempts: {base}")
    return successes[0] if successes else None


def match_labels(reference, candidate):
    """Maximum-overlap one-to-one matching; noise stays noise, unmatched get new IDs."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    refs = np.unique(reference[reference >= 0])
    cands = np.unique(candidate[candidate >= 0])
    mapping = {-1: -1}
    if len(refs) and len(cands):
        mask = (reference >= 0) & (candidate >= 0)
        counts = np.zeros((len(refs), len(cands)), dtype=np.int64)
        np.add.at(counts, (np.searchsorted(refs, reference[mask]), np.searchsorted(cands, candidate[mask])), 1)
        rows, cols = linear_sum_assignment(-counts)
        mapping.update({int(cands[c]): int(refs[r]) for r, c in zip(rows, cols) if counts[r, c] > 0})
    next_id = int(refs.max()) + 1 if len(refs) else 0
    for c in cands:
        if int(c) not in mapping:
            mapping[int(c)] = next_id
            next_id += 1
    aligned = np.array([mapping[int(c)] for c in candidate], dtype=np.int64)
    return aligned, mapping


def rigid_align(reference, candidate):
    """Display only: translation + orthogonal rotation/reflection, NO scaling."""
    import numpy as np
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    u, _, vt = np.linalg.svd((b - b.mean(0)).T @ (a - a.mean(0)))
    rotation = u @ vt
    translation = a.mean(0) - b.mean(0) @ rotation
    return b @ rotation + translation, rotation, translation


def label_metrics(a, b):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    aligned, _ = match_labels(a, b)
    both = (a != -1) & (b != -1)
    noise = (a == -1) != (b == -1)
    return {"ari_including_noise": float(adjusted_rand_score(a, b)),
            "ari_both_nonnoise": float(adjusted_rand_score(a[both], b[both])) if both.sum() >= 2 else None,
            "both_nonnoise_count": int(both.sum()), "ami_including_noise": float(adjusted_mutual_info_score(a, b)),
            "assignment_disagreement": float(np.mean(a != aligned)),
            "cluster_to_cluster_disagreement": float(np.mean(both & (a != aligned))),
            "noise_status_disagreement": float(noise.mean()),
            "cluster_to_noise_count": int(np.sum((a != -1) & (b == -1))),
            "noise_to_cluster_count": int(np.sum((a == -1) & (b != -1)))}
