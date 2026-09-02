"""Opt-in, notebook-level progress logs for long simulations.

The messages emitted here always appear in the notebook.  After
``configure_run_logging(...)`` is enabled, the same messages are also appended
to one immediately-flushed text file for the lifetime of the Python kernel.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import re
import threading
import time

from .paths import RUN_LOGS_ROOT


_log_path = None
_write_lock = threading.Lock()


def _safe_notebook_name(notebook_name):
    name = Path(str(notebook_name)).stem.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "Notebook"


def configure_run_logging(enabled=True, notebook_name="Notebook", log_dir=None):
    """Enable or disable the text progress log for this notebook kernel.

    Typical notebook usage::

        from md_Helpers.run_logs import configure_run_logging
        configure_run_logging(True, notebook_name="Cavitation_EOS_Sweep_Driver")

    Put this call at the top of a simulation cell. Call
    ``configure_run_logging(False)`` before a small run that should only print
    progress in the cell. Enabling logging again starts a new log file. Any
    source thermalization started by a cavitation or excitation call uses the
    same active log automatically.

    The returned value is the active ``Path``, or ``None`` when disabled.
    """
    global _log_path

    if not enabled:
        _log_path = None
        return None

    folder = Path(RUN_LOGS_ROOT if log_dir is None else log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    _log_path = folder / f"{_safe_notebook_name(notebook_name)}_{started}.log"
    _log_path.touch(exist_ok=False)
    return _log_path


def current_run_log():
    """Return the active progress-log path, or ``None`` if logging is off."""
    return _log_path


def progress_print(message):
    """Print one progress line and mirror it to the active log, if any."""
    message = str(message)
    print(message, flush=True)

    path = _log_path
    if path is None:
        return

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with _write_lock:
        with path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(f"[{timestamp}] {message}\n")
            stream.flush()


@contextmanager
def simulation_progress(simulation_name, **parameters):
    """Print/log a simulation's start, successful finish, or failure."""
    arguments = ", ".join(
        f"{name} = {value}" for name, value in parameters.items()
    )
    progress_print(f"Starting {simulation_name} Evolution ({arguments})")
    started = time.perf_counter()

    try:
        yield
    except BaseException as error:
        elapsed = time.perf_counter() - started
        progress_print(
            f"Simulation failed after {elapsed:.2f} seconds: "
            f"{type(error).__name__}: {error}"
        )
        raise
    else:
        elapsed = time.perf_counter() - started
        progress_print(f"simulation done, simulation time = {elapsed:.2f} seconds")
