"""Paper-specific checks of scripts/paper are hard errors for the frozen data only."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

PAPER_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "paper"


def _run(tmp_path, frozen):
    data = tmp_path / ("frozen" if frozen else "rerun")
    data.mkdir()
    if frozen:
        (data / "MANIFEST.json").write_text("{}")
    # Subprocess: several script directories have a module named _common.
    code = textwrap.dedent(f"""
        import json
        import _style as style
        from _common import Paths, expect_paper
        paths = Paths({str(data)!r}, {str(tmp_path / "build")!r})
        out = {{"frozen": paths.frozen}}
        for name, call in [
            ("expect", lambda: expect_paper(paths, False, "sign flipped")),
            ("limits", lambda: style.fit_log_limits(paths, [{{"x": 500.0, "dataset": "d"}}], "x", (1, 100), "X")),
        ]:
            try:
                out[name] = ["ok", call()]
            except (AssertionError, ValueError) as exc:
                out[name] = ["error", str(exc)]
        print(json.dumps(out))
    """)
    run = subprocess.run([sys.executable, "-c", code], cwd=PAPER_SCRIPTS, capture_output=True, text=True, check=True)
    return json.loads(run.stdout.splitlines()[-1]), run.stderr


def test_frozen_data_must_match_the_paper(tmp_path):
    out, _ = _run(tmp_path, frozen=True)
    assert out["frozen"] and out["expect"][0] == "error" and out["limits"][0] == "error"


def test_new_results_are_reported_and_accepted(tmp_path):
    out, stderr = _run(tmp_path, frozen=False)
    assert not out["frozen"]
    assert out["expect"] == ["ok", None] and "sign flipped" in stderr
    status, (low, high) = out["limits"]
    assert status == "ok" and low == 1 and high > 500
