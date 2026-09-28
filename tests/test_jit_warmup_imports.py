from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_fresh_python(source: str, *, timeout: float = 60.0):
    environment = os.environ.copy()
    environment.pop("IBUMAP_DISABLE_JIT_WARMUP", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(SRC), environment.get("PYTHONPATH")) if value
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_import_does_not_start_background_warmup_or_deadlock():
    completed = _run_fresh_python(
        """
        import json
        import threading

        import ibumap
        try:
            from ibumap.kernels.cpu import ibfft
            import_result = "ok"
        except ModuleNotFoundError as exc:
            import_result = f"optional_dependency_missing:{exc.name}"

        print(json.dumps({
            "import_result": import_result,
            "threads": [thread.name for thread in threading.enumerate()],
            "status": ibumap.get_jit_warmup_status(),
        }))
        """
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["import_result"] == "ok" or payload["import_result"].startswith(
        "optional_dependency_missing:"
    )
    assert "ibumap-jit-warmup" not in payload["threads"]
    assert payload["status"]["state"] == "not_started"


def test_explicit_warmup_is_synchronous_and_idempotent():
    completed = _run_fresh_python(
        """
        import json
        import threading

        import ibumap

        first = ibumap.warmup_jit()
        second = ibumap.warmup_jit()
        print(json.dumps({
            "first": first,
            "second": second,
            "threads": [thread.name for thread in threading.enumerate()],
        }))
        """,
        timeout=180.0,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["first"]["state"] in {"complete", "failed"}
    assert payload["second"] == payload["first"]
    assert "ibumap-jit-warmup" not in payload["threads"]
