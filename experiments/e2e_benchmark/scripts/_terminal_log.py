"""Capture one script invocation's complete terminal output in a log file."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


_ACTIVE_SCRIPT_ENV = "IBUMAP_TERMINAL_LOG_ACTIVE_SCRIPT"


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _write_chunk(chunk: bytes, log_handle: Any) -> None:
    log_handle.write(chunk)
    log_handle.flush()
    stream = sys.stdout
    binary = getattr(stream, "buffer", None)
    if binary is not None:
        binary.write(chunk)
        binary.flush()
        return
    stream.write(chunk.decode("utf-8", errors="replace"))
    stream.flush()


def reexec_with_terminal_log(script_file: str | Path) -> None:
    """Re-run an entry-point script while teeing its stdout and stderr.

    The parent process starts before the script's heavy imports. The child
    inherits stdin, while stdout and stderr are merged into one pipe so Python
    warnings, tracebacks, native-library diagnostics, and subprocess output
    are captured in the same order seen in the terminal.
    """

    script_path = Path(script_file).resolve()
    active_script = os.environ.get(_ACTIVE_SCRIPT_ENV)
    if active_script == str(script_path):
        os.environ.pop(_ACTIVE_SCRIPT_ENV, None)
        return

    experiment_root = script_path.parents[1]
    logs_root = experiment_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"{script_path.stem}_{_timestamp()}_pid{os.getpid()}.log"
    command = [sys.executable, "-u", str(script_path), *sys.argv[1:]]
    environment = os.environ.copy()
    environment[_ACTIVE_SCRIPT_ENV] = str(script_path)
    environment["PYTHONUNBUFFERED"] = "1"

    started_at = datetime.now().astimezone().isoformat()
    header = (
        "=== terminal capture start ===\n"
        f"started_at: {started_at}\n"
        f"script: {script_path}\n"
        f"command: {shlex.join(command)}\n"
        f"log_path: {log_path}\n"
        "=== output ===\n"
    ).encode("utf-8")

    process: subprocess.Popen[bytes] | None = None
    return_code = 1
    with log_path.open("wb") as log_handle:
        _write_chunk(header, log_handle)
        try:
            process = subprocess.Popen(
                command,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                bufsize=0,
            )
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                _write_chunk(chunk, log_handle)
            return_code = int(process.wait())
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            return_code = 130
        except BaseException:
            _write_chunk(traceback.format_exc().encode("utf-8"), log_handle)
            return_code = 1
        finally:
            if return_code < 0:
                return_code = 128 + abs(return_code)
            footer = (
                "\n=== terminal capture end ===\n"
                f"finished_at: {datetime.now().astimezone().isoformat()}\n"
                f"exit_code: {return_code}\n"
            ).encode("utf-8")
            _write_chunk(footer, log_handle)

    raise SystemExit(return_code)


def setup_terminal_logger(name: str) -> logging.Logger:
    """Create a stream logger; the terminal capture owns file persistence."""

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger
