"""Guarded operations that coordinate run files with database records."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from .database import SQLiteRunDatabase
from .paths import ProjectPaths


_RUN_ID_PATTERN = re.compile(r"^[0-9]{14}$")
_ACTIVE_STATUSES = {"Initializing", "Running"}


def _validated_run_directory(
    run_id: str,
    master: dict[str, Any],
    thermalization: dict[str, Any] | None,
    project_paths: ProjectPaths,
) -> Path:
    sim_type = master.get("Sim_Type")
    if sim_type != "Thermalization":
        raise NotImplementedError(
            "delete_run currently supports Thermalization runs only"
        )

    top_directory = project_paths.top_directory.resolve()
    expected = project_paths.for_run(sim_type, run_id).directory.resolve()
    stored_location = (
        thermalization.get("File_Location") if thermalization else None
    )
    if stored_location:
        stored_path = Path(stored_location).expanduser()
        candidate = (
            stored_path if stored_path.is_absolute()
            else top_directory / stored_path
        ).resolve()
    else:
        candidate = expected

    if candidate != expected:
        raise RuntimeError(
            "Refusing deletion because File_Location does not match the "
            f"canonical run directory: {candidate} != {expected}"
        )
    if not candidate.is_relative_to(top_directory):
        raise RuntimeError(
            f"Refusing to delete a directory outside {top_directory}"
        )
    if candidate == top_directory or candidate.name != run_id:
        raise RuntimeError(f"Unsafe run directory resolved for {run_id}")
    return candidate


def delete_run(
    run_id: str,
    *,
    dry_run: bool = True,
    confirm_run_id: str | None = None,
    force: bool = False,
    project_paths: ProjectPaths | None = None,
    database: SQLiteRunDatabase | None = None,
) -> dict[str, Any]:
    """Delete one run directory and its Thermalization and Master rows.

    ``dry_run=True`` only reports the exact targets. Permanent deletion
    requires ``confirm_run_id`` to exactly match ``run_id``. Initializing and
    running simulations are rejected unless ``force=True``.
    """

    run_id = str(run_id)
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain exactly 14 decimal digits")

    project_paths = project_paths or ProjectPaths()
    database = database or SQLiteRunDatabase(project_paths.database)
    database.initialize()

    master = database.get_run(run_id)
    if master is None:
        raise KeyError(f"Run_ID was not found: {run_id}")
    thermalization = database.get_thermalization(run_id)
    run_directory = _validated_run_directory(
        run_id,
        master,
        thermalization,
        project_paths,
    )

    preview: dict[str, Any] = {
        "run_id": run_id,
        "sim_type": master.get("Sim_Type"),
        "status": master.get("Status"),
        "run_directory": str(run_directory),
        "trajectory_gsd": str(run_directory / "trajectory.gsd"),
        "trajectory_exists": (run_directory / "trajectory.gsd").is_file(),
        "run_hdf5": str(run_directory / "run.hdf5"),
        "hdf5_exists": (run_directory / "run.hdf5").is_file(),
        "master_rows": 1,
        "thermalization_rows": int(thermalization is not None),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return preview

    if confirm_run_id != run_id:
        raise ValueError(
            "Permanent deletion requires confirm_run_id to exactly match run_id"
        )
    if master.get("Status") in _ACTIVE_STATUSES and not force:
        raise RuntimeError(
            f"Refusing to delete active run {run_id} with "
            f"Status={master.get('Status')!r}; pass force=True only after "
            "confirming that the simulation process has stopped"
        )
    if not run_directory.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist; no SQL rows were deleted: "
            f"{run_directory}"
        )

    staging_root = project_paths.top_directory / ".deletion_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_directory = staging_root / f"{run_id}-{uuid4().hex}"
    run_directory.rename(staged_directory)

    try:
        deleted = database.delete_run_records(
            run_id,
            allow_active=force,
        )
    except Exception:
        staged_directory.rename(run_directory)
        try:
            staging_root.rmdir()
        except OSError:
            pass
        raise

    try:
        shutil.rmtree(staged_directory)
        try:
            staging_root.rmdir()
        except OSError:
            pass
    except Exception as error:
        raise RuntimeError(
            "SQL rows were deleted, but the run files could not be "
            f"permanently removed. They remain at {staged_directory}"
        ) from error

    return {
        **preview,
        **deleted,
        "directory_deleted": True,
        "dry_run": False,
    }
