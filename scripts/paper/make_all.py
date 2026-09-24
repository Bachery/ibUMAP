#!/usr/bin/env python3
"""Rebuild every figure and table of the paper from the frozen results, then verify.

    python scripts/paper/make_all.py                     # paper/data -> paper/build
    python scripts/paper/make_all.py --data-dir DIR      # results of a fresh (L2) run with the same layout
    python scripts/paper/make_all.py --formats pdf png   # additional formats

Needs only numpy, scipy, pandas, matplotlib and scikit-learn (pip install -e ".[paper]"),
no GPU and no ibUMAP installation. Figure 1 (method overview) is a drawing, not data.
Verification against the manuscript is skipped when --data-dir is given.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _common import DEFAULT_BUILD, DEFAULT_DATA

HERE = Path(__file__).resolve().parent
# Order matters: the figure builders cross-check against tables written earlier.
BUILDERS = [
    ("ch4_tables.py", False),          # tab:mechanism-quality, -effects, -family
    ("ch4_effect_figure.py", True),    # fig:mechanism-effects
    ("ch4_psweep_table.py", False),    # tab:mechanism-psweep
    ("ch4_calendar_table.py", False),  # tab:calendar-audit
    ("ch5_tables.py", False),          # tab:e2e-fidelity-common and Appendix B tables
    ("ch5_main_figure.py", True),      # fig:e2e-results
    ("ch5_profile_figures.py", True),  # fig:appendix-speedups, fig:appendix-runtime-profiles
    ("ch5_seed_cost_figure.py", True),  # fig:appendix-cpu-seed-cost
    ("ch5_fidelity_figure.py", True),  # fig:e2e-fidelity
    ("ch5_stage_share_figures.py", True),  # fig:appendix-stage-share
    ("ch6_panels.py", True),           # fig:braque-repeatability
    ("ch6_reuse_table.py", False),     # tab:braque-reuse
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--formats", nargs="+", default=["pdf"], choices=("pdf", "svg", "png"))
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    common = ["--data-dir", str(args.data_dir or DEFAULT_DATA), "--build-dir", str(args.build_dir)]
    for script, is_figure in BUILDERS:
        start = time.perf_counter()
        command = [sys.executable, str(HERE / script), *common] + (["--formats", *args.formats] if is_figure else [])
        result = subprocess.run(command)
        if result.returncode:
            print(f"FAILED: {script}")
            return result.returncode
        print(f"  ({time.perf_counter() - start:.1f}s)")
    if args.no_verify or args.data_dir is not None:
        return 0
    return subprocess.run([sys.executable, str(HERE / "verify.py"), "--build-dir", str(args.build_dir)]).returncode


if __name__ == "__main__":
    sys.exit(main())
