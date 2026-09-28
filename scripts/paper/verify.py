#!/usr/bin/env python3
"""Check paper/build against the values in the submitted manuscript.

  * every LaTeX table-row file and plot-data CSV must match the manuscript version
    exactly (after dropping '%' comment lines and normalizing line endings);
  * every manuscript figure must exist as a PDF;
  * selected numbers quoted in the text must round to the printed value.

Expected hashes and quoted values are listed in paper/expected_outputs.json.
PDFs are not compared byte for byte: glyph shapes depend on the installed fonts
(Times New Roman in the manuscript, STIXGeneral as fallback). With --compare-pdf DIR
and poppler's pdftoppm on PATH, each PDF is rasterized next to DIR/<name>.pdf and the
mean absolute pixel difference is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import DEFAULT_BUILD, REPO

EXPECTED = REPO / "paper" / "expected_outputs.json"


def normalized_sha256(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = [line for line in text.split("\n") if not line.lstrip().startswith("%")]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def lookup(obj, keys):
    for key in keys:
        obj = obj[key]
    return obj


def raster_difference(a: Path, b: Path) -> float | None:
    if shutil.which("pdftoppm") is None:
        return None
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        images = []
        for i, pdf in enumerate((a, b)):
            stem = Path(tmp) / str(i)
            subprocess.run(["pdftoppm", "-r", "110", "-png", "-singlefile", str(pdf), str(stem)], check=True)
            images.append(np.asarray(Image.open(f"{stem}.png").convert("RGB"), float))
    if images[0].shape != images[1].shape:
        return float("inf")
    return float(np.abs(images[0] - images[1]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--compare-pdf", type=Path, help="directory with the manuscript PDFs")
    args = parser.parse_args()
    expected = json.loads(EXPECTED.read_text())
    build = args.build_dir.resolve()
    failures = []

    for name, digest in expected["files"].items():
        path = build / name
        if not path.exists():
            failures.append(f"missing {name}")
        elif normalized_sha256(path) != digest:
            failures.append(f"differs from the manuscript: {name}")
        else:
            print(f"[ok] {name}")
    for name in expected["figures"]:
        path = build / "figures" / name
        if not path.exists():
            failures.append(f"missing figures/{name}")
            continue
        note = ""
        if args.compare_pdf:
            diff = raster_difference(path, args.compare_pdf / name)
            note = "" if diff is None else f"  (mean abs pixel difference {diff:.3f})"
        print(f"[ok] figures/{name}{note}")
    for item in expected["quoted_values"]:
        path = build / item["file"]
        try:
            value = item["format"].format(lookup(json.loads(path.read_text()), item["path"]))
        except (OSError, KeyError, IndexError) as exc:
            failures.append(f"cannot read {item['file']}:{'/'.join(map(str, item['path']))} ({exc})")
            continue
        if value != item["expected"]:
            failures.append(f"{item['where']}: expected {item['expected']}, got {value}")
        else:
            print(f"[ok] {item['where']}: {value}")

    total = len(expected["files"]) + len(expected["figures"]) + len(expected["quoted_values"])
    if failures:
        print(f"\n{len(failures)} of {total} checks failed:", *failures, sep="\n  ")
        return 1
    print(f"\nAll {total} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
