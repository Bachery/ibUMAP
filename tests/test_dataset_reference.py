"""Reference records of the paper's processed datasets (datasets/reference/)."""
import gzip
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
DATASET_SCRIPTS = ROOT / "scripts" / "datasets"


def test_every_catalog_dataset_has_a_reference_record():
    catalog = json.loads((DATASETS / "catalog.json").read_text())
    ids = {entry["dataset_id"] for entry in catalog["datasets"]}
    assert ids == {path.name for path in (DATASETS / "reference").iterdir() if path.is_dir()}
    for entry in catalog["datasets"]:
        reference = DATASETS / "reference" / entry["dataset_id"]
        metadata = json.loads((reference / "metadata.json").read_text())
        assert metadata["dataset_id"] == entry["dataset_id"]
        assert metadata["feature_shape"] == entry["feature_shape"]
        records = {}
        for line in (reference / "checksums.txt").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                digest, rel = line.split(maxsplit=1)
                assert len(digest) == 64 and not rel.startswith("/")
                records[rel] = digest
        assert metadata.get("primary_array", "features.npy") in records
        assert "metadata.json" not in records and "checksums.txt" not in records
    # Only the placeholder is tracked under datasets/processed/.
    tracked = subprocess.run(["git", "ls-files", "datasets/processed"], cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode == 0:
        assert tracked.stdout.split() == ["datasets/processed/.gitkeep"]


def _compare(dataset_dir, reference_dir):
    # Run in a subprocess: scripts/datasets/common and scripts/common are both named "common".
    code = textwrap.dedent(f"""
        import json
        from pathlib import Path
        from common.reference import compare_to_reference
        r = compare_to_reference(Path({str(dataset_dir)!r}), Path({str(reference_dir)!r}))
        print(json.dumps({{"ok": r.ok, "errors": r.errors, "warnings": r.warnings, "info": r.info}}))
    """)
    out = subprocess.run([sys.executable, "-c", code], cwd=DATASET_SCRIPTS, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _write_dataset(directory, *, mtime, generated_at, shape=(3, 2)):
    directory.mkdir(parents=True)
    np.save(directory / "features.npy", np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    with gzip.GzipFile(directory / "obs.csv.gz", "wb", mtime=mtime) as handle:
        handle.write(b"sample_id\n0\n1\n2\n")
    (directory / "metadata.json").write_text(json.dumps(
        {"dataset_id": "toy", "feature_shape": list(shape), "feature_dtype": "float32", "generated_at": generated_at}))


def test_compare_uses_content_hashes_and_structural_metadata(tmp_path):
    reference, prepared = tmp_path / "reference", tmp_path / "prepared"
    _write_dataset(reference, mtime=1, generated_at="2026-01-01")
    code = textwrap.dedent(f"""
        from pathlib import Path
        from common.reference import content_sha256, write_reference_checksums
        d = Path({str(reference)!r})
        write_reference_checksums(d / "checksums.txt", {{n: content_sha256(d / n) for n in ("features.npy", "obs.csv.gz")}})
    """)
    subprocess.run([sys.executable, "-c", code], cwd=DATASET_SCRIPTS, check=True)
    # A different gzip timestamp and generation time still match.
    _write_dataset(prepared, mtime=2, generated_at="2026-09-25")
    assert (prepared / "obs.csv.gz").read_bytes() != (reference / "obs.csv.gz").read_bytes()
    result = _compare(prepared, reference)
    assert result["ok"], result
    assert any("generated_at" in message for message in result["info"])
    # A changed array or a changed shape does not.
    np.save(prepared / "features.npy", np.zeros((3, 2), dtype=np.float32))
    assert any("features.npy" in message for message in _compare(prepared, reference)["errors"])
    other = tmp_path / "other"
    _write_dataset(other, mtime=1, generated_at="2026-01-01", shape=(2, 3))
    errors = _compare(other, reference)["errors"]
    assert any("feature_shape" in message for message in errors)
