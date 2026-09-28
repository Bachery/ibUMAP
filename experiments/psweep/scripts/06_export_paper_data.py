#!/usr/bin/env python3
"""Convert results/04_summaries to the paper-data layout (``<output>/psweep/``).

    python scripts/06_export_paper_data.py                 # -> paper/rerun/psweep/
    python scripts/06_export_paper_data.py --output DIR    # -> DIR/psweep/

Writes the two files that ``scripts/paper/ch4_psweep_table.py`` reads:

  quality.csv.gz   quality_scores.csv without the local path column
  runs.csv.gz      runs.csv without the local path columns

Values are copied as strings. The export refuses summaries with issues or
incomplete tasks (run ``04_summarize_results.py --strict`` first).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import EXPERIMENT_ROOT, experiment_paths, load_configs, read_json
from common.paper_data import DEFAULT_RERUN_ROOT, read_csv, write_csv_gz

GROUP = "psweep"
QUALITY_DROP = ("evaluation_path",)
RUN_DROP = ("embedding_path", "record_path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--output", type=Path, default=DEFAULT_RERUN_ROOT,
                        help="paper-data root; the group is written to OUTPUT/psweep (default: paper/rerun)")
    args = parser.parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    summaries = experiment_paths(configs, config_dir).summaries_root
    manifest = read_json(summaries / "manifest.json")
    if manifest.get("configuration_hash") is None or manifest.get("issues"):
        raise SystemExit(f"Summary manifest has issues; rerun 04_summarize_results.py --strict: {manifest.get('issues')}")
    counts = manifest.get("status_counts") or {}
    tasks = int(manifest.get("task_count", 0))
    if counts.get("run_ok") != tasks or counts.get("evaluation_ok") != tasks:
        raise SystemExit(f"Incomplete summary ({counts}, {tasks} tasks); finish runs and evaluations first.")

    q_fields, quality = read_csv(summaries / "quality_scores.csv")
    r_fields, runs = read_csv(summaries / "runs.csv")
    q_fields = [field for field in q_fields if field not in QUALITY_DROP]
    r_fields = [field for field in r_fields if field not in RUN_DROP]
    out = args.output.resolve() / GROUP
    files = {
        "quality.csv.gz": write_csv_gz(out / "quality.csv.gz", q_fields,
                                       [{field: row[field] for field in q_fields} for row in quality]),
        "runs.csv.gz": write_csv_gz(out / "runs.csv.gz", r_fields,
                                    [{field: row[field] for field in r_fields} for row in runs]),
    }
    print(f"Exported {len(runs)} runs and {len(quality)} quality rows -> {out}")
    for name, digest in files.items():
        print(f"  {name}  {digest}")


if __name__ == "__main__":
    main()
