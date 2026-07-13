from pathlib import Path

import numpy as np
import pandas as pd

from .paths import (
    CAVITATION_EVOLVED_V3_ROOT,
    CAVITATION_STATES_V3_ROOT,
    EXCITATION_EVOLVED_V3_ROOT,
    EXCITATION_STATES_V3_ROOT,
    MASTER_CSVS_V3_ROOT,
    SIMPLE_LATTICES_V3_ROOT,
    THERMALIZED_STATES_V3_ROOT,
)


THERMO_BASE = "hoomd-data/md/compute/ThermodynamicQuantities"
TIMESTEP_PATH = "hoomd-data/Simulation/timestep"

RESULT_FILE_SUFFIXES = {
    ".csv",
    ".gsd",
    ".h5",
    ".hdf5",
    ".parquet",
}

DEFAULT_RESULTS_INVENTORY_ROOTS = {
    "simple_lattice": SIMPLE_LATTICES_V3_ROOT,
    "thermalized": THERMALIZED_STATES_V3_ROOT,
    "cavitation_initial": CAVITATION_STATES_V3_ROOT,
    "cavitation_evolved": CAVITATION_EVOLVED_V3_ROOT,
    "excitation_initial": EXCITATION_STATES_V3_ROOT,
    "excitation_evolved": EXCITATION_EVOLVED_V3_ROOT,
    "master_csv": MASTER_CSVS_V3_ROOT,
}


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


def _relative_path(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _result_file_role(path):
    name = path.name

    if name.endswith("_log.hdf5") or name in {
        "cavitation_log.hdf5",
        "excitation_log.hdf5",
        "randomization_log.hdf5",
    }:
        return "log"

    if name.endswith("_metadata.hdf5") or name.endswith("_creation.hdf5"):
        return "metadata"

    if name.endswith("_trajectory.gsd"):
        return "trajectory"

    if name.endswith("_final.gsd"):
        return "final_state"

    if name.endswith(".gsd"):
        return "state"

    if name.endswith(".csv"):
        return "summary_csv"

    if name.endswith(".parquet"):
        return "index"

    return "result_file"


def _file_stem_key(path):
    stem = path.stem
    for suffix in [
        "_trajectory",
        "_final",
        "_initial",
        "_creation",
        "_metadata",
        "_log",
    ]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def inventory_result_files(
    roots=None,
    suffixes=RESULT_FILE_SUFFIXES,
    include_missing_roots=True,
):
    """
    Return one row per discovered result file under the configured V3 roots.

    The inventory is intentionally shallow: it records what files exist,
    where they live, their size/mtime, and simple role labels inferred from
    filename conventions. It does not open GSD/HDF5 contents, so it can be run
    quickly while compute services are unavailable.
    """

    roots = roots or DEFAULT_RESULTS_INVENTORY_ROOTS
    suffixes = {str(suffix).lower() for suffix in suffixes}
    rows = []

    for result_family, root in roots.items():
        root = Path(root)

        if not root.exists():
            if include_missing_roots:
                rows.append({
                    "result_family": result_family,
                    "root": str(root),
                    "relative_path": "",
                    "path": "",
                    "filename": "",
                    "suffix": "",
                    "file_role": "missing_root",
                    "stem_key": "",
                    "parent": "",
                    "exists": False,
                    "size_bytes": np.nan,
                    "mtime": "",
                })
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in suffixes:
                continue

            stat = path.stat()
            rows.append({
                "result_family": result_family,
                "root": str(root),
                "relative_path": _relative_path(path, root),
                "path": str(path),
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "file_role": _result_file_role(path),
                "stem_key": _file_stem_key(path),
                "parent": str(path.parent),
                "exists": True,
                "size_bytes": int(stat.st_size),
                "mtime": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    sort_cols = [
        col for col in [
            "exists",
            "result_family",
            "parent",
            "stem_key",
            "file_role",
            "filename",
        ]
        if col in table.columns
    ]
    return table.sort_values(sort_cols).reset_index(drop=True)


def summarize_results_inventory(inventory):
    """Summarize an inventory table by result family and file role."""

    if inventory.empty:
        return pd.DataFrame(columns=[
            "result_family",
            "file_role",
            "n_files",
            "total_size_bytes",
        ])

    existing = inventory[inventory["exists"]].copy()
    if existing.empty:
        return pd.DataFrame(columns=[
            "result_family",
            "file_role",
            "n_files",
            "total_size_bytes",
        ])

    return (
        existing.groupby(["result_family", "file_role"], dropna=False)
        .agg(
            n_files=("path", "count"),
            total_size_bytes=("size_bytes", "sum"),
        )
        .reset_index()
        .sort_values(["result_family", "file_role"])
        .reset_index(drop=True)
    )


def build_results_inventory_csv(
    roots=None,
    output_path=None,
    output_name="results_inventory.csv",
    suffixes=RESULT_FILE_SUFFIXES,
    include_missing_roots=True,
):
    """
    Scan V3 result folders and write a lightweight file inventory CSV.

    Returns the detailed inventory table. Use
    :func:`summarize_results_inventory` on the returned table for a compact
    count/size summary by result family and file role.
    """

    output_path = _default_master_csv_path(output_path, output_name)
    table = inventory_result_files(
        roots=roots,
        suffixes=suffixes,
        include_missing_roots=include_missing_roots,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    return table


SEITZ_MASTER_COLUMNS = [
    "n_cells",
    "N_source",
    "rho_source",
    "source_kT",
    "source_nsteps",
    "source_seed",
    "source_radius",
    "bubble_seed",
    "evolve_nsteps",
    "evolve_seed",
    "final_phase_separated",
    "radius_outcome",
    "status",
    "reason",
    "N_cav",
    "u_c",
    "rho_liquid",
    "rho_cav",
    "P_EOS",
    "u_EOS",
    "Q",
    "cavitation_log_path",
    "trajectory_path",
]


def _seitz_nan_terms():
    return {
        "N_cav": np.nan,
        "u_c": np.nan,
        "rho_liquid": np.nan,
        "rho_cav": np.nan,
        "P_EOS": np.nan,
        "u_EOS": np.nan,
        "Q": np.nan,
    }


def _first_nonmissing(*values):
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return np.nan


def _summarize_cavitation_radius(log_path, trajectory_path):
    from . import cavitation_analysis
    from .cavitation_sweep import summarize_bubble_survival

    if not trajectory_path or not Path(trajectory_path).exists():
        return {}

    measurements = cavitation_analysis.measure_cavitation_trajectory(
        trajectory_path=trajectory_path,
        log_path=log_path,
    )
    return summarize_bubble_survival(measurements)


def summarize_seitz_cavitation_log(
    log_path,
    eos_table=None,
    n_last=100,
    classification_kwargs=None,
    seitz_kwargs=None,
):
    """
    Summarize one completed cavitation evolution for the barebones Seitz CSV.

    Seitz terms are only computed when the final voxel classifier reports
    phase separation. Rethermalized rows keep all Seitz-specific columns as
    ``NaN``.
    """

    import h5py
    from . import classification
    from . import seitz

    log_path = Path(log_path)
    classification_kwargs = dict(classification_kwargs or {})
    seitz_kwargs = dict(seitz_kwargs or {})

    row = {
        "cavitation_log_path": str(log_path),
        **_seitz_nan_terms(),
    }

    try:
        with h5py.File(log_path, mode="r") as hdf:
            state = _read_attrs(hdf, "metadata/state")
            run = _read_attrs(hdf, "metadata/run")
            source = _read_attrs(hdf, "metadata/source")
            creation = _read_attrs(hdf, "metadata/creation")
            paths = _read_attrs(hdf, "metadata/paths")

        n_cells = _first_nonmissing(
            state.get("n_fcc_cells"),
            source.get("n_fcc_cells"),
        )
        row.update({
            "n_cells": n_cells,
            "N_source": _first_nonmissing(
                source.get("source_N"),
                4 * int(n_cells) ** 3 if not pd.isna(n_cells) else np.nan,
            ),
            "rho_source": _first_nonmissing(
                source.get("source_rho"),
                state.get("source_rho"),
            ),
            "source_kT": _first_nonmissing(
                source.get("source_kT"),
                state.get("kT"),
            ),
            "source_nsteps": source.get("source_nsteps", np.nan),
            "source_seed": source.get("source_seed", np.nan),
            "source_radius": _first_nonmissing(
                creation.get("radius"),
                creation.get("bubble_radius"),
            ),
            "bubble_seed": creation.get("bubble_seed", np.nan),
            "evolve_nsteps": run.get("nsteps", np.nan),
            "evolve_seed": run.get("seed", np.nan),
            "trajectory_path": str(paths.get("trajectory_path", "")),
        })

        final_state_path = paths.get("final_state_path")
        voxel, _ = classification.read_phase_method_attrs(log_path, "voxel")
        if "phase_separated" not in voxel:
            voxel = classification.write_voxel_phase_separation_metadata(
                log_path=log_path,
                state_path=final_state_path,
                **classification_kwargs,
            )

        row["final_phase_separated"] = voxel.get("phase_separated")

        try:
            survival = _summarize_cavitation_radius(
                log_path=log_path,
                trajectory_path=row["trajectory_path"],
            )
            row["radius_outcome"] = survival.get("radius_outcome", np.nan)
        except Exception as error:
            row["radius_outcome"] = np.nan
            row["radius_summary_error"] = repr(error)

        if not bool(row["final_phase_separated"]):
            row["status"] = "rethermalized"
            row["reason"] = "final voxel classifier did not phase separate"
            return row

        terms = seitz.extract_bubble_state_terms(
            metadata_path=log_path,
            eos_table=eos_table,
            n_last=n_last,
            **seitz_kwargs,
        )
        row.update({
            "status": "seitz_computed",
            "reason": "",
            "N_cav": terms.get("Nc", np.nan),
            "u_c": terms.get("uc", np.nan),
            "rho_liquid": terms.get("rho_0", np.nan),
            "rho_cav": terms.get("rho_c", np.nan),
            "P_EOS": terms.get("P0", terms.get("p0", np.nan)),
            "u_EOS": terms.get("u0", np.nan),
            "Q": terms.get("Q", terms.get("q_seitz", np.nan)),
        })
        return row
    except Exception as error:
        row["status"] = "summary_failed"
        row["reason"] = repr(error)
        return row


def build_seitz_master_csv(
    root=CAVITATION_EVOLVED_V3_ROOT,
    output_path=None,
    output_name="seitz_master.csv",
    eos_table=None,
    n_last=100,
    classification_kwargs=None,
    seitz_kwargs=None,
    preserve_extra_columns=True,
):
    """Build/update the barebones Seitz master CSV for found cavitation runs."""

    root = Path(root)
    output_path = _default_master_csv_path(output_path, output_name)

    rows = []
    if root.exists():
        for log_path in sorted(root.rglob("cavitation_log.hdf5")):
            rows.append(summarize_seitz_cavitation_log(
                log_path=log_path,
                eos_table=eos_table,
                n_last=n_last,
                classification_kwargs=classification_kwargs,
                seitz_kwargs=seitz_kwargs,
            ))

    table = pd.DataFrame(rows)
    for column in SEITZ_MASTER_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan

    extra_columns = [
        column for column in table.columns
        if column not in SEITZ_MASTER_COLUMNS
    ]
    table = table[[*SEITZ_MASTER_COLUMNS, *extra_columns]]

    if not table.empty:
        table = table.sort_values([
            column for column in [
                "n_cells",
                "rho_source",
                "source_kT",
                "source_nsteps",
                "source_seed",
                "source_radius",
                "bubble_seed",
                "evolve_nsteps",
                "evolve_seed",
                "cavitation_log_path",
            ]
            if column in table.columns
        ]).reset_index(drop=True)

    if (
        preserve_extra_columns
        and output_path.exists()
        and "cavitation_log_path" in table.columns
    ):
        old = pd.read_csv(output_path)
        if "cavitation_log_path" in old.columns:
            old_extra_cols = [
                col for col in old.columns
                if col not in table.columns and col != "cavitation_log_path"
            ]
            if old_extra_cols:
                table = table.merge(
                    old[["cavitation_log_path", *old_extra_cols]],
                    on="cavitation_log_path",
                    how="left",
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    return table


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
