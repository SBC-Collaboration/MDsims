from pathlib import Path

import numpy as np
import pandas as pd

from .paths import MASTER_CSVS_V3_ROOT, THERMALIZED_STATES_V3_ROOT


THERMO_BASE = "hoomd-data/md/compute/ThermodynamicQuantities"
TIMESTEP_PATH = "hoomd-data/Simulation/timestep"


def _clean_value(value):
    if isinstance(value, bytes):
        return value.decode()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()

    return value


def _read_attrs(hdf, group_path):
    if group_path not in hdf:
        return {}

    return {
        key: _clean_value(value)
        for key, value in hdf[group_path].attrs.items()
    }


def _last_window_stats(values, n_last):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan, np.nan

    window = values[-min(int(n_last), values.size):]
    std = float(np.std(window, ddof=1)) if window.size > 1 else 0.0

    return float(np.mean(window)), std


def _read_dataset(hdf, dataset_path):
    if dataset_path not in hdf:
        return None

    return np.asarray(hdf[dataset_path])


def summarize_thermalization_log(log_path, n_last=100):
    """
    Summarize one V3 thermalization HDF5 log as one master-CSV row.

    The output includes metadata plus final and last-window statistics for
    pressure, potential energy, kinetic energy, and measured kinetic
    temperature.
    """

    log_path = Path(log_path)
    import h5py

    row = {
        "log_path": str(log_path),
        "status": "completed",
    }

    try:
        with h5py.File(log_path, mode="r") as hdf:
            state = _read_attrs(hdf, "metadata/state")
            run = _read_attrs(hdf, "metadata/run")
            lj = _read_attrs(hdf, "metadata/lj")
            paths = _read_attrs(hdf, "metadata/paths")
            source = _read_attrs(hdf, "metadata/source")
            phase = _read_attrs(
                hdf,
                "metadata/classification/phase_separation",
            )

            row.update(state)
            row.update(run)
            row.update(lj)
            row.update(source)
            row.update(paths)

            if "log_path" not in row or not row["log_path"]:
                row["log_path"] = str(log_path)

            for key, value in phase.items():
                row[key] = value

            timestep = _read_dataset(hdf, TIMESTEP_PATH)
            if timestep is not None and timestep.size > 0:
                row["n_log_rows"] = int(timestep.size)
                row["first_logged_timestep"] = int(timestep[0])
                row["last_logged_timestep"] = int(timestep[-1])
            else:
                row["n_log_rows"] = 0

            quantity_specs = [
                (
                    "pressure",
                    f"{THERMO_BASE}/pressure",
                    "pressure",
                    False,
                ),
                (
                    "potential_energy",
                    f"{THERMO_BASE}/potential_energy",
                    "PE",
                    True,
                ),
                (
                    "kinetic_energy",
                    f"{THERMO_BASE}/kinetic_energy",
                    "KE",
                    True,
                ),
                (
                    "kinetic_temperature",
                    f"{THERMO_BASE}/kinetic_temperature",
                    "kinetic_temperature",
                    False,
                ),
            ]

            n_particles = row.get("N")

            for quantity, dataset_path, label, per_particle in quantity_specs:
                values = _read_dataset(hdf, dataset_path)
                if values is None or values.size == 0:
                    continue

                values = np.asarray(values, dtype=np.float64)
                mean, std = _last_window_stats(values, n_last)
                row[f"{label}_final"] = float(values[-1])
                row[f"{label}_mean_last{int(n_last)}"] = mean
                row[f"{label}_std_last{int(n_last)}"] = std

                if per_particle and n_particles:
                    per_particle_values = values / int(n_particles)
                    mean_pp, std_pp = _last_window_stats(
                        per_particle_values,
                        n_last,
                    )
                    row[f"{label}_final_per_particle"] = float(
                        per_particle_values[-1]
                    )
                    row[f"{label}_per_particle_mean_last{int(n_last)}"] = (
                        mean_pp
                    )
                    row[f"{label}_per_particle_std_last{int(n_last)}"] = (
                        std_pp
                    )

    except Exception as error:
        row["status"] = "summary_failed"
        row["summary_error"] = repr(error)

    if "actual_rho" not in row or pd.isna(row.get("actual_rho")):
        if row.get("N") is not None and row.get("volume") is not None:
            row["actual_rho"] = float(row["N"]) / float(row["volume"])
        elif row.get("N") is not None and row.get("BoxLength") is not None:
            row["actual_rho"] = (
                float(row["N"]) / float(row["BoxLength"]) ** 3
            )
        elif row.get("target_rho") is not None:
            row["actual_rho"] = row["target_rho"]

    if "phase_separated" not in row:
        row["phase_separated"] = False

    return row


def _default_master_csv_path(output_path=None, output_name=None):
    if output_path is not None:
        return Path(output_path)

    if output_name is None:
        output_name = "thermalization_master.csv"

    return Path(MASTER_CSVS_V3_ROOT) / output_name


def build_thermalization_master_csv(
    root=THERMALIZED_STATES_V3_ROOT,
    output_path=None,
    output_name=None,
    n_last=100,
    n_fcc_cells=None,
    preserve_extra_columns=True,
):
    """
    Recreate/update a master CSV for all V3 thermalization logs found so far.

    Parameters
    ----------
    root:
        Root folder to scan. Defaults to ``THERMALIZED_STATES_V3_ROOT``.
    output_path:
        CSV path to write. Defaults to
        ``MASTER_CSVS_V3_ROOT / "thermalization_master.csv"``.
    output_name:
        Filename to use inside ``MASTER_CSVS_V3_ROOT`` when ``output_path`` is
        not supplied. For example, ``"thermalization_master_ncells_30.csv"``.
    n_last:
        Number of final log rows used for mean/std summary columns.
    n_fcc_cells:
        Optional filter, useful for recreating only the ncells=30 table.
    preserve_extra_columns:
        If the CSV already exists and has hand-added columns, keep those
        columns by merging them back on ``log_path``.
    """

    root = Path(root)
    output_path = _default_master_csv_path(output_path, output_name)

    if not root.exists():
        raise FileNotFoundError(f"Thermalization root does not exist: {root}")

    rows = []

    for log_path in sorted(root.rglob("*_log.hdf5")):
        row = summarize_thermalization_log(log_path, n_last=n_last)

        if n_fcc_cells is not None:
            if int(row.get("n_fcc_cells", -1)) != int(n_fcc_cells):
                continue

        rows.append(row)

    table = pd.DataFrame(rows)

    if not table.empty:
        table = table.sort_values(
            [
                col for col in [
                    "n_fcc_cells",
                    "target_rho",
                    "kT",
                    "nsteps",
                    "seed",
                    "log_path",
                ]
                if col in table.columns
            ]
        ).reset_index(drop=True)

    if (
        preserve_extra_columns
        and output_path.exists()
        and "log_path" in table.columns
    ):
        old = pd.read_csv(output_path)
        if "log_path" in old.columns:
            extra_cols = [
                col for col in old.columns
                if col not in table.columns and col != "log_path"
            ]
            if extra_cols:
                table = table.merge(
                    old[["log_path", *extra_cols]],
                    on="log_path",
                    how="left",
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    return table
