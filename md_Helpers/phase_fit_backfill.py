"""Resumable migration to the default skew-liquid phase-fit model."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analysis import voxel_bins_for_ncells
from .database import SQLiteRunDatabase, utc_now
from .paths import ProjectPaths
from .storage import replace_versioned_phase_fit_metadata
from .voxel_fit import (
    PHASE_FIT_METHOD,
    PHASE_FIT_METHOD_VERSION,
    fit_trajectory_voxel_skew_mixture,
    phase_fit_sql_values,
)


def _trajectory_and_hdf5(row, paths: ProjectPaths) -> tuple[Path, Path]:
    location = Path(str(row["File_Location"])).expanduser()
    if not location.is_absolute():
        location = paths.top_directory / location
    return location / "trajectory.gsd", location / "run.hdf5"


def _saved_phase_frames(hdf5_path: Path) -> list[int] | None:
    if not hdf5_path.exists():
        return None
    import h5py

    with h5py.File(hdf5_path, mode="r") as hdf5:
        group_path = "mdsims/output"
        key = "Phase_Average_Trajectory_Frame_IDs"
        if group_path not in hdf5 or key not in hdf5[group_path].attrs:
            return None
        return [int(value) for value in np.atleast_1d(hdf5[group_path].attrs[key])]


def _validate_fit(fit: dict[str, Any], n_cells: int) -> None:
    if not bool(fit.get("success")):
        raise RuntimeError(f"optimizer failed: {fit.get('message')}")
    expected_nbins = voxel_bins_for_ncells(n_cells)
    if int(fit["voxel_nbins"]) != expected_nbins:
        raise ValueError(
            f"nbins changed: expected {expected_nbins}, got {fit['voxel_nbins']}"
        )
    finite_fields = (
        "rho_liquid",
        "rho_gas",
        "V_liquid",
        "V_gas",
        "liquid_scale_density",
        "liquid_shape_alpha",
        "gas_weight",
        "liquid_weight",
        "interface_weight",
        "log_likelihood",
        "AIC",
        "BIC",
    )
    invalid = [field for field in finite_fields if not np.isfinite(fit[field])]
    if invalid:
        raise ValueError(f"non-finite fitted values: {invalid}")
    if not float(fit["rho_gas"]) < float(fit["rho_liquid"]):
        raise ValueError("fitted vapor density is not below liquid density")
    weights = np.array(
        [fit["gas_weight"], fit["liquid_weight"], fit["interface_weight"]],
        dtype=float,
    )
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"invalid mixture weights: {weights.tolist()}")
    alpha = float(fit["liquid_shape_alpha"])
    if np.isclose(alpha, -10.0, atol=1e-6) or np.isclose(alpha, 10.0, atol=1e-6):
        raise ValueError(f"liquid skewness hit its bound: alpha={alpha}")
    normal_log_likelihood = (10.0 - float(fit["normal_model_AIC"])) / 2.0
    if float(fit["log_likelihood"]) + 1e-6 < normal_log_likelihood:
        raise ValueError("skew model likelihood is below its nested normal model")
    if not np.isclose(
        float(fit["V_liquid"]) + float(fit["V_gas"]),
        float(fit["box_volume"]),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("fitted phase volumes do not sum to the box volume")


def phase_fit_history_values(
    fit: dict[str, Any],
    *,
    attempt_count: int,
    started_at: str,
) -> dict[str, Any]:
    return {
        "Method": PHASE_FIT_METHOD,
        "Method_Version": PHASE_FIT_METHOD_VERSION,
        "Status": "Complete",
        "Attempt_Count": int(attempt_count),
        "Started_At": started_at,
        "Completed_At": utc_now(),
        "Error_Message": None,
        "N_Bins": int(fit["voxel_nbins"]),
        "Frame_Indices": json.dumps([int(value) for value in fit["frame_indices"]]),
        "Frames_Used": int(fit["frames_used"]),
        "Interface_Points": int(fit["interface_points"]),
        "Interface_Void_Fraction": float(fit["interface_void_fraction"]),
        "Alpha_Lower_Bound": float(fit["alpha_lower_bound"]),
        "Alpha_Upper_Bound": float(fit["alpha_upper_bound"]),
        "rho_liquid": float(fit["rho_liquid"]),
        "rho_liquid_unc": float(fit["rho_liquid_unc"]),
        "rho_gas": float(fit["rho_gas"]),
        "rho_gas_unc": float(fit["rho_gas_unc"]),
        "V_liquid": float(fit["V_liquid"]),
        "V_liquid_unc": float(fit["V_liquid_unc"]),
        "V_gas": float(fit["V_gas"]),
        "V_gas_unc": float(fit["V_gas_unc"]),
        "Liquid_Scale_Density": float(fit["liquid_scale_density"]),
        "Liquid_Shape_Alpha": float(fit["liquid_shape_alpha"]),
        "Liquid_Shape_Alpha_Unc": float(fit["liquid_shape_alpha_unc"]),
        "Gas_Weight": float(fit["gas_weight"]),
        "Liquid_Weight": float(fit["liquid_weight"]),
        "Interface_Weight": float(fit["interface_weight"]),
        "Log_Likelihood": float(fit["log_likelihood"]),
        "AIC": float(fit["AIC"]),
        "BIC": float(fit["BIC"]),
    }


def backfill_skew_phase_fits(
    database: SQLiteRunDatabase,
    *,
    project_paths: ProjectPaths | None = None,
    run_ids: Iterable[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    retry_failed: bool = False,
    max_attempts: int = 3,
    fit_options: dict[str, Any] | None = None,
):
    """Refit separated runs safely, continuing past per-run failures.

    A run is fitted only once even when it appears in several SQL result
    tables. A validated result updates all of those tables atomically, while
    each table receives its own versioned history record.
    """

    import pandas as pd

    database.initialize()
    paths = project_paths or ProjectPaths()
    fit_options = dict(fit_options or {})
    forbidden_options = {"nbins", "frame_indices", "num_frames"} & set(fit_options)
    if forbidden_options:
        raise ValueError(
            "Backfill preserves the original nbins/frame policy; remove options: "
            f"{sorted(forbidden_options)}"
        )
    requested = None if run_ids is None else [str(value) for value in run_ids]
    targets = database.phase_fit_targets(run_ids=requested, separated_only=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        grouped[str(target["Run_ID"])].append(target)
    run_id_order = sorted(grouped)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive or None")

    results = []
    attempted = 0
    for run_id in run_id_order:
        rows = grouped[run_id]
        tables = sorted({str(row["Sim_Table"]) for row in rows})
        history = database.query_phase_fit_history(
            run_id=run_id,
            method_version=PHASE_FIT_METHOD_VERSION,
        )
        history_by_table = {str(row["Sim_Table"]): row for row in history}
        if all(
            history_by_table.get(table, {}).get("Status") == "Complete"
            for table in tables
        ):
            results.append({
                "Run_ID": run_id,
                "Tables": ",".join(tables),
                "Status": "Already_Complete",
                "Message": None,
            })
            continue
        if not retry_failed and any(
            history_by_table.get(table, {}).get("Status") == "Failed"
            for table in tables
        ):
            results.append({
                "Run_ID": run_id,
                "Tables": ",".join(tables),
                "Status": "Failed_Not_Retried",
                "Message": "Pass retry_failed=True to retry this run.",
            })
            continue
        attempt = 1 + max(
            [int(row.get("Attempt_Count") or 0) for row in history] or [0]
        )
        if attempt > int(max_attempts):
            results.append({
                "Run_ID": run_id,
                "Tables": ",".join(tables),
                "Status": "Max_Attempts",
                "Message": f"maximum attempts reached ({max_attempts})",
            })
            continue
        if limit is not None and attempted >= int(limit):
            break
        attempted += 1
        n_cells_values = {int(row["N_Cells"]) for row in rows}
        locations = {
            str(_trajectory_and_hdf5(row, paths)[0].parent.resolve())
            for row in rows
        }
        if len(n_cells_values) != 1 or len(locations) != 1:
            error = (
                f"inconsistent duplicate SQL rows: N_Cells={sorted(n_cells_values)}, "
                f"File_Location={sorted(locations)}"
            )
            for table in tables:
                database.upsert_phase_fit_history(
                    Run_ID=run_id,
                    Sim_Table=table,
                    Method=PHASE_FIT_METHOD,
                    Method_Version=PHASE_FIT_METHOD_VERSION,
                    Status="Failed",
                    Attempt_Count=attempt,
                    Completed_At=utc_now(),
                    Error_Message=error,
                )
            results.append({
                "Run_ID": run_id,
                "Tables": ",".join(tables),
                "Status": "Failed",
                "Message": error,
            })
            continue
        if dry_run:
            results.append({
                "Run_ID": run_id,
                "Tables": ",".join(tables),
                "Status": "Planned",
                "Message": f"nbins={voxel_bins_for_ncells(next(iter(n_cells_values)))}",
            })
            continue

        started_at = utc_now()
        for table in tables:
            database.upsert_phase_fit_history(
                Run_ID=run_id,
                Sim_Table=table,
                Method=PHASE_FIT_METHOD,
                Method_Version=PHASE_FIT_METHOD_VERSION,
                Status="Running",
                Attempt_Count=attempt,
                Started_At=started_at,
            )
        try:
            canonical = rows[0]
            trajectory_path, hdf5_path = _trajectory_and_hdf5(canonical, paths)
            if not trajectory_path.exists():
                raise FileNotFoundError(trajectory_path)
            frame_indices = _saved_phase_frames(hdf5_path)
            fit = fit_trajectory_voxel_skew_mixture(
                trajectory_path,
                next(iter(n_cells_values)),
                frame_indices=frame_indices,
                **fit_options,
            )
            _validate_fit(fit, next(iter(n_cells_values)))
            fit = {"status": "Complete", **fit, "backfilled_at": utc_now()}
            replace_versioned_phase_fit_metadata(hdf5_path, fit)
            active_values = phase_fit_sql_values(fit)
            updated_tables = database.apply_phase_fit_everywhere(
                run_id,
                active_values=active_values,
                history_values=phase_fit_history_values(
                    fit,
                    attempt_count=attempt,
                    started_at=started_at,
                ),
            )
            results.append({
                "Run_ID": run_id,
                "Tables": ",".join(updated_tables),
                "Status": "Complete",
                "Message": None,
            })
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            for table in tables:
                database.upsert_phase_fit_history(
                    Run_ID=run_id,
                    Sim_Table=table,
                    Method=PHASE_FIT_METHOD,
                    Method_Version=PHASE_FIT_METHOD_VERSION,
                    Status="Failed",
                    Attempt_Count=attempt,
                    Started_At=started_at,
                    Completed_At=utc_now(),
                    Error_Message=message,
                )
            results.append({
                "Run_ID": run_id,
                "Tables": ",".join(tables),
                "Status": "Failed",
                "Message": message,
            })
    return pd.DataFrame.from_records(
        results,
        columns=["Run_ID", "Tables", "Status", "Message"],
    )
