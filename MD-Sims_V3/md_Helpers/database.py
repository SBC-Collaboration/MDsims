# database.py
# ============================================================

from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import h5py
import shutil


from . import runs as lh
from . import classification as ps
from .paths import THERMALIZED_STATES_V2_ROOT


# ============================================================
# Build scan arrays
# ============================================================

def make_scan_values(
    min_value,
    max_value,
    step,
    decimals=2,
):
    """
    Build a rounded scan array including the endpoint.

    Example:
        make_scan_values(0.70, 1.00, 0.02)
    """

    values = np.round(
        np.arange(
            min_value,
            max_value + 0.5 * step,
            step,
        ),
        decimals,
    )

    return values


# ============================================================
# Sweep key
# ============================================================

def make_v2_sweep_key(
    n_fcc_cells,
    target_rho,
    kT,
    nsteps,
    log_period,
    seed,
    phase_name,
):
    """
    Make a stable string key identifying one V2 sweep point.
    """

    return (
        f"n_cells_{int(n_fcc_cells)}"
        f"_rho_{float(target_rho):.3f}"
        f"_kT_{float(kT):.3f}"
        f"_nsteps_{int(nsteps)}"
        f"_logperiod_{int(log_period)}"
        f"_seed_{int(seed)}"
        f"_phase_{phase_name}"
    )


# ============================================================
# Summary CSV path
# ============================================================

def get_v2_summary_path(
    n_fcc_cells,
    kT_min,
    kT_max,
    rho_min,
    rho_max,
    nsteps,
    base_folder=THERMALIZED_STATES_V2_ROOT,
):
    """
    Build the standard V2 sweep-summary CSV path.
    """

    summary_folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / "Sweep_Summaries"
    )

    summary_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        summary_folder
        / (
            f"sweep_summary_ncells_{int(n_fcc_cells)}"
            f"_kT_{float(kT_min):.2f}_{float(kT_max):.2f}"
            f"_rho_{float(rho_min):.2f}_{float(rho_max):.2f}"
            f"_nsteps_{int(nsteps)}.csv"
        )
    )

    return summary_path


# ============================================================
# Load existing summary CSV
# ============================================================

def load_summary_csv(
    summary_path,
):
    """
    Load a summary CSV if it exists. Otherwise return an empty DataFrame.
    """

    summary_path = Path(summary_path)

    if summary_path.exists():
        return pd.read_csv(summary_path)

    return pd.DataFrame()


# ============================================================
# Update one row in summary CSV dataframe
# ============================================================

def update_summary_row(
    df,
    row,
    key_column="sweep_key",
):
    """
    Replace any existing row with the same sweep key, then append row.
    """

    row_df = pd.DataFrame([row])

    if df.empty or key_column not in df.columns:
        return row_df

    df = df[df[key_column] != row[key_column]].copy()

    df = pd.concat(
        [df, row_df],
        ignore_index=True,
    )

    sort_cols = []

    for col in ["kT", "target_rho", "nsteps", "seed"]:
        if col in df.columns:
            sort_cols.append(col)

    if len(sort_cols) > 0:
        df = df.sort_values(
            sort_cols,
            ignore_index=True,
        )

    return df


# ============================================================
# Check whether row has useful completed values
# ============================================================

def completed_row_has_values(
    row,
):
    """
    Return True only if the CSV row is completed and has usable values.

    This intentionally does NOT just check whether a log file exists,
    because failed simulations can leave behind empty or partial log files.
    """

    if row.get("status", None) != "completed":
        return False

    required_columns = [
        "N",
        "actual_rho",
        "BoxLength",
        "volume",
        "fcc_cell_size",
        "phase_separated",

        "pressure_mean_last100",
        "pressure_std_last100",

        "PE_per_particle_mean_last100",
        "PE_per_particle_std_last100",

        "KE_per_particle_mean_last100",
        "KE_per_particle_std_last100",

        "kinetic_temperature_mean_last100",
        "kinetic_temperature_std_last100",

        "n_values_used",
        "first_timestep_used",
        "last_timestep_used",

        "state_path",
        "log_path",
    ]

    for col in required_columns:
        if col not in row.index:
            return False

        if pd.isna(row[col]):
            return False

    if int(row["n_values_used"]) <= 0:
        return False

    return True


def already_completed_with_values(
    df,
    sweep_key,
):
    """
    Check whether this sweep point already has a completed CSV row
    with usable values.
    """

    if df.empty:
        return False

    if "sweep_key" not in df.columns:
        return False

    matches = df[df["sweep_key"] == sweep_key]

    if len(matches) == 0:
        return False

    row = matches.iloc[-1]

    return completed_row_has_values(row)


# ============================================================
# HDF5 log tail statistics
# ============================================================

def get_log_tail_stats(
    log,
    quantity,
    n_last=100,
):
    """
    Compute mean/std from the last n_last logged values.

    Supported quantities:
    - pressure
    - PE_per_particle
    - KE_per_particle
    - kinetic_temperature
    """

    metadata = log["metadata"]["attrs"]

    N = int(metadata["N"])

    timestep = np.asarray(
        log["hoomd-data"]["Simulation"]["timestep"],
        dtype=int,
    )

    thermo = (
        log["hoomd-data"]["md"]
           ["compute"]
           ["ThermodynamicQuantities"]
    )

    if quantity == "pressure":
        values = np.asarray(
            thermo["pressure"],
            dtype=float,
        )

    elif quantity == "PE_per_particle":
        values = (
            np.asarray(thermo["potential_energy"], dtype=float) / N
        )

    elif quantity == "KE_per_particle":
        values = (
            np.asarray(thermo["kinetic_energy"], dtype=float) / N
        )

    elif quantity == "kinetic_temperature":
        values = np.asarray(
            thermo["kinetic_temperature"],
            dtype=float,
        )

    else:
        raise ValueError(
            "quantity must be one of: "
            "'pressure', 'PE_per_particle', "
            "'KE_per_particle', 'kinetic_temperature'"
        )

    if len(values) == 0:
        raise ValueError(f"No logged values found for quantity: {quantity}")

    values_tail = values[-n_last:]
    timestep_tail = timestep[-n_last:]

    if len(values_tail) > 1:
        std = float(np.std(values_tail, ddof=1))
    else:
        std = 0.0

    stats = {
        "quantity": quantity,
        "mean": float(np.mean(values_tail)),
        "std": std,
        "n_values_used": int(len(values_tail)),
        "first_timestep_used": int(timestep_tail[0]),
        "last_timestep_used": int(timestep_tail[-1]),
        "N": N,
    }

    return stats


# ============================================================
# Summarize a completed run
# ============================================================

def summarize_completed_result(
    result,
    run_time_seconds,
    n_last=100,
):
    """
    Given the result dictionary from sh.get_or_make_thermalized_state(...),
    read the HDF5 log and return a row-update dictionary for the sweep CSV.
    """

    paths = result["paths"]

    log_path = paths["log_path"]
    state_path = paths["state_path"]

    log = lh.read_hdf5_log(log_path)

    metadata = log["metadata"]["attrs"]

    pressure_stats = get_log_tail_stats(
        log=log,
        quantity="pressure",
        n_last=n_last,
    )

    pe_per_particle_stats = get_log_tail_stats(
        log=log,
        quantity="PE_per_particle",
        n_last=n_last,
    )

    ke_per_particle_stats = get_log_tail_stats(
        log=log,
        quantity="KE_per_particle",
        n_last=n_last,
    )

    kinetic_temperature_stats = get_log_tail_stats(
        log=log,
        quantity="kinetic_temperature",
        n_last=n_last,
    )

    phase_separated = metadata.get(
        "phase_separated",
        None,
    )

    if phase_separated is not None:
        phase_separated = bool(phase_separated)

    row_update = {
        "status": "completed",
        "created_new": bool(result["created_new"]),

        "state_path": str(state_path),
        "log_path": str(log_path),

        "phase_name": metadata.get("phase_name", ""),
        "lattice_type": metadata.get("lattice_type", "fcc"),
        "density_mode": metadata.get(
            "density_mode",
            "fixed_N_variable_L",
        ),

        "n_fcc_cells": int(metadata["n_fcc_cells"]),
        "N": int(metadata["N"]),

        "target_rho": float(metadata["target_rho"]),
        "actual_rho": float(metadata["actual_rho"]),

        "BoxLength": float(metadata["BoxLength"]),
        "volume": float(metadata["volume"]),
        "fcc_cell_size": float(
            metadata.get(
                "fcc_cell_size",
                float(metadata["BoxLength"])
                / int(metadata["n_fcc_cells"]),
            )
        ),

        "kT": float(metadata["kT"]),
        "nsteps": int(metadata["nsteps"]),
        "log_period": int(metadata["log_period"]),
        "seed": int(metadata["seed"]),

        "dt": float(metadata["dt"]),
        "epsilon_LJ": float(metadata["epsilon_LJ"]),
        "sigma_LJ": float(metadata["sigma_LJ"]),
        "r_cut_LJ": float(metadata["r_cut_LJ"]),
        "r_on_LJ": float(metadata["r_on_LJ"]),
        "buffer_LJ": float(metadata["buffer_LJ"]),
        "lj_mode": metadata["lj_mode"],

        "phase_separated": phase_separated,

        "pressure_mean_last100": pressure_stats["mean"],
        "pressure_std_last100": pressure_stats["std"],

        "PE_per_particle_mean_last100": pe_per_particle_stats["mean"],
        "PE_per_particle_std_last100": pe_per_particle_stats["std"],

        "KE_per_particle_mean_last100": ke_per_particle_stats["mean"],
        "KE_per_particle_std_last100": ke_per_particle_stats["std"],

        "kinetic_temperature_mean_last100": kinetic_temperature_stats["mean"],
        "kinetic_temperature_std_last100": kinetic_temperature_stats["std"],

        "n_values_used": pressure_stats["n_values_used"],
        "first_timestep_used": pressure_stats["first_timestep_used"],
        "last_timestep_used": pressure_stats["last_timestep_used"],

        "final_timestep": int(
            metadata.get("final_timestep", metadata["nsteps"])
        ),

        "starting_state_path": metadata.get(
            "starting_state_path",
            "",
        ),

        "run_time_seconds": float(run_time_seconds),
        "error": "",
        "traceback": "",
    }

    return row_update


# ============================================================
# Make failed row update
# ============================================================

def make_failed_row_update(
    error,
    run_time_seconds,
):
    """
    Build the failed-run part of a sweep CSV row.
    """

    row_update = {
        "status": "failed",
        "created_new": np.nan,

        "actual_rho": np.nan,
        "BoxLength": np.nan,
        "volume": np.nan,
        "fcc_cell_size": np.nan,

        "phase_separated": np.nan,

        "pressure_mean_last100": np.nan,
        "pressure_std_last100": np.nan,

        "PE_per_particle_mean_last100": np.nan,
        "PE_per_particle_std_last100": np.nan,

        "KE_per_particle_mean_last100": np.nan,
        "KE_per_particle_std_last100": np.nan,

        "kinetic_temperature_mean_last100": np.nan,
        "kinetic_temperature_std_last100": np.nan,

        "n_values_used": np.nan,
        "first_timestep_used": np.nan,
        "last_timestep_used": np.nan,

        "run_time_seconds": float(run_time_seconds),
        "error": str(error),
        "traceback": traceback.format_exc(),
    }

    return row_update


# ============================================================
# Build full expected grid for plotting
# ============================================================

def build_full_grid(
    kT_values,
    rho_values,
):
    """
    Build full temperature-density grid for plotting,
    including not-yet-run points.
    """

    full_grid = pd.MultiIndex.from_product(
        [kT_values, rho_values],
        names=["kT", "target_rho"],
    ).to_frame(index=False)

    return full_grid


# ============================================================
# Clean boolean values from CSV
# ============================================================

def clean_bool_value(
    x,
):
    """
    Convert CSV values like True, False, 'True', 'False', 1, 0, nan
    into actual Python True/False/None.
    """

    if pd.isna(x):
        return None

    if isinstance(x, bool):
        return x

    x_str = str(x).strip().lower()

    if x_str in ["true", "1", "yes"]:
        return True

    if x_str in ["false", "0", "no"]:
        return False

    return None


# ============================================================
# Read phase-separation metadata for CSV
# ============================================================

def read_phase_separation_metadata_for_csv(
    log_path,
):
    return ps.read_phase_separation_metadata_for_csv(
        log_path=log_path,
    )


def read_phase_separation_metadata_flexible_for_csv(
    log_path,
):
    return ps.read_phase_separation_metadata_for_csv(
        log_path=log_path,
    )


    

# ============================================================
# Update sweep CSV phase-separation columns from HDF5 metadata
# ============================================================

def _clean_hdf5_attr_value(
    value,
):
    """
    Convert HDF5 attribute values into normal Python values.
    """

    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def update_one_summary_csv_from_metadata(
    summary_path,
    dry_run=True,
    backup=True,
    show_only_changed=True,
):
    """
    Update one sweep-summary CSV using HDF5 metadata.

    This does not recompute logs or rerun simulations.
    It only updates:
    - phase_separated
    - phase_sep_method
    - phase_sep_nbins
    - phase_sep_density_threshold
    - phase_sep_voxel_fraction_threshold
    - phase_sep_low_density_fraction
    - phase_sep_updated_from_saved_gsd
    """

    summary_path = Path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV does not exist: {summary_path}")

    df = pd.read_csv(summary_path)

    if "log_path" not in df.columns:
        raise KeyError(
            f"CSV does not have a log_path column: {summary_path}"
        )

    # ============================================================
    # Make sure phase-separation columns exist with correct dtypes
    # ============================================================

    if "phase_separated" not in df.columns:
        df["phase_separated"] = pd.Series(
            [None] * len(df),
            dtype="object",
        )
    else:
        df["phase_separated"] = df["phase_separated"].astype("object")

    # String / object columns
    object_phase_columns = [
        "phase_sep_method",
        "phase_sep_updated_from_saved_gsd",
    ]

    for col in object_phase_columns:
        if col not in df.columns:
            df[col] = pd.Series(
                [None] * len(df),
                dtype="object",
            )
        else:
            df[col] = df[col].astype("object")

    # Numeric columns
    numeric_phase_columns = [
        "phase_sep_nbins",
        "phase_sep_density_threshold",
        "phase_sep_voxel_fraction_threshold",
        "phase_sep_low_density_fraction",
    ]

    for col in numeric_phase_columns:
        if col not in df.columns:
            df[col] = np.nan

    report_rows = []

    # ============================================================
    # Loop through completed rows and patch phase metadata
    # ============================================================

    for idx, row in df.iterrows():
        try:
            if "status" in df.columns:
                if row.get("status", None) != "completed":
                    continue

            log_path = row.get("log_path", "")

            if pd.isna(log_path) or str(log_path).strip() == "":
                continue

            phase_metadata = read_phase_separation_metadata_for_csv(
                log_path=log_path,
            )

            old_phase_separated = clean_bool_value(
                row.get("phase_separated", np.nan)
            )

            new_phase_separated = phase_metadata["phase_separated"]

            changed = (
                old_phase_separated is None
                or old_phase_separated != new_phase_separated
            )

            # ----------------------------------------------------
            # Update dataframe values
            # ----------------------------------------------------
            df.loc[idx, "phase_separated"] = new_phase_separated

            for key, value in phase_metadata.items():
                if key == "phase_separated":
                    continue

                df.loc[idx, key] = value

            # ----------------------------------------------------
            # Only add changed rows to the report
            # ----------------------------------------------------
            if changed:
                report_rows.append({
                    "summary_path": str(summary_path),
                    "n_fcc_cells": row.get("n_fcc_cells", np.nan),
                    "target_rho": row.get("target_rho", np.nan),
                    "kT": row.get("kT", np.nan),

                    "old_phase_separated": old_phase_separated,
                    "new_phase_separated": new_phase_separated,
                    "changed": bool(changed),

                    "phase_sep_low_density_fraction": phase_metadata[
                        "phase_sep_low_density_fraction"
                    ],
                    "phase_sep_density_threshold": phase_metadata[
                        "phase_sep_density_threshold"
                    ],
                    "phase_sep_voxel_fraction_threshold": phase_metadata[
                        "phase_sep_voxel_fraction_threshold"
                    ],

                    "log_path": str(log_path),
                })

        except Exception as error:
            # Keep errors in the raw report only when show_only_changed=False.
            report_rows.append({
                "summary_path": str(summary_path),
                "n_fcc_cells": row.get("n_fcc_cells", np.nan),
                "target_rho": row.get("target_rho", np.nan),
                "kT": row.get("kT", np.nan),

                "old_phase_separated": row.get("phase_separated", np.nan),
                "new_phase_separated": np.nan,
                "changed": False,

                "error": repr(error),
                "log_path": str(row.get("log_path", "")),
            })

    report_df = pd.DataFrame(report_rows)

    # ============================================================
    # Save updated CSV
    # ============================================================

    if not dry_run:
        if backup:
            backup_path = summary_path.with_name(
                summary_path.stem + "_backup_before_phase_csv_update.csv"
            )

            shutil.copy2(
                summary_path,
                backup_path,
            )

        df.to_csv(
            summary_path,
            index=False,
        )

    # ============================================================
    # Only show rows whose phase_separated value changed
    # ============================================================

    if show_only_changed and len(report_df) > 0:
        if "changed" in report_df.columns:
            report_df = report_df[report_df["changed"] == True].copy()

        keep_columns = [
            "summary_path",
            "n_fcc_cells",
            "target_rho",
            "kT",
            "old_phase_separated",
            "new_phase_separated",
            "phase_sep_low_density_fraction",
            "phase_sep_density_threshold",
            "phase_sep_voxel_fraction_threshold",
            "log_path",
        ]

        keep_columns = [
            col for col in keep_columns
            if col in report_df.columns
        ]

        report_df = report_df[keep_columns].reset_index(drop=True)

    return report_df


# ============================================================
# Find V2 CSV files that can be updated from HDF5 metadata
# ============================================================

def find_v2_csvs_with_log_path(
    base_folder=THERMALIZED_STATES_V2_ROOT,
    include_backups=False,
    verbose=True,
):
    """
    Find all V2 CSV files that have a log_path column.

    This is intentionally broader than the old search:

        **/Sweep_Summaries/*.csv

    because now we also want to catch:
    - sweep summary CSVs
    - sliding-window CSVs
    - master CSVs
    - any future CSV that has a log_path column

    Backup CSVs are skipped by default.
    """

    base_folder = Path(base_folder)

    all_csv_paths = sorted(
        base_folder.glob("**/*.csv")
    )

    usable_csv_paths = []
    skipped_rows = []

    for csv_path in all_csv_paths:
        csv_path = Path(csv_path)

        name_lower = csv_path.name.lower()
        stem_lower = csv_path.stem.lower()
        path_lower = str(csv_path).lower()

        # ========================================================
        # Skip backup files by default
        # ========================================================

        is_backup = (
            "backup" in name_lower
            or "backup" in stem_lower
            or "_backup_" in path_lower
            or "backup_before" in path_lower
        )

        if is_backup and not include_backups:
            skipped_rows.append({
                "csv_path": str(csv_path),
                "reason": "backup_csv",
            })
            continue

        # ========================================================
        # Only keep CSVs that have a log_path column
        # ========================================================

        try:
            header_df = pd.read_csv(
                csv_path,
                nrows=0,
            )

            if "log_path" not in header_df.columns:
                skipped_rows.append({
                    "csv_path": str(csv_path),
                    "reason": "missing_log_path_column",
                })
                continue

            usable_csv_paths.append(csv_path)

        except Exception as error:
            skipped_rows.append({
                "csv_path": str(csv_path),
                "reason": repr(error),
            })

    skipped_df = pd.DataFrame(skipped_rows)

    if verbose:
        print("Finding V2 CSVs with log_path")
        print("=" * 70)
        print("base_folder =", base_folder)
        print("total CSVs found =", len(all_csv_paths))
        print("usable CSVs found =", len(usable_csv_paths))
        print("include_backups =", include_backups)
        print("=" * 70)

        if len(usable_csv_paths) > 0:
            print("\nUsable CSVs:")
            for path in usable_csv_paths:
                print(path)

        if len(skipped_df) > 0:
            print("\nSkipped CSV counts:")
            print(skipped_df["reason"].value_counts(dropna=False))

    return usable_csv_paths, skipped_df


# ============================================================
# Update all V2 CSV files from HDF5 metadata
# ============================================================

def update_all_v2_summary_csvs_from_metadata(
    base_folder=THERMALIZED_STATES_V2_ROOT,
    dry_run=True,
    backup=False,
    include_backups=False,
    verbose=True,
):
    """
    Update all V2 CSV files using the current HDF5 metadata.

    This searches for every CSV under:

        Thermalized_States_v2/**/*.csv

    Then it keeps only CSVs that:
    - are not backup CSVs, unless include_backups=True
    - contain a log_path column

    This catches:
    - normal sweep-summary CSVs
    - sliding-window CSVs
    - master CSVs
    - future CSVs with log_path columns

    It updates:
    - phase_separated
    - phase_sep_method
    - phase_sep_nbins
    - phase_sep_density_threshold
    - phase_sep_voxel_fraction_threshold
    - phase_sep_low_density_fraction
    - phase_sep_updated_from_saved_gsd

    Parameters
    ----------
    backup : bool
        If True, each original CSV is backed up before writing.
        Default is False now to avoid backup clutter.

    include_backups : bool
        If False, backup CSVs are ignored.
    """

    base_folder = Path(base_folder)

    csv_paths, skipped_df = find_v2_csvs_with_log_path(
        base_folder=base_folder,
        include_backups=include_backups,
        verbose=verbose,
    )

    if verbose:
        print("\nUpdating V2 CSV files from HDF5 metadata")
        print("=" * 70)
        print("base_folder =", base_folder)
        print("number of usable CSVs found =", len(csv_paths))
        print("dry_run =", dry_run)
        print("backup =", backup)
        print("include_backups =", include_backups)
        print("=" * 70)

    all_reports = []
    csv_summary_rows = []

    for csv_path in csv_paths:
        try:
            report_df = update_one_summary_csv_from_metadata(
                summary_path=csv_path,
                dry_run=dry_run,
                backup=backup,
                show_only_changed=True,
            )

            n_changed = len(report_df)

            if n_changed > 0:
                all_reports.append(report_df)

            csv_summary_rows.append({
                "csv_path": str(csv_path),
                "status": "updated" if not dry_run else "dry_run",
                "changed_rows": n_changed,
                "error": "",
            })

            if verbose:
                print(
                    csv_path.name,
                    "| changed rows:",
                    n_changed,
                )

        except Exception as error:
            csv_summary_rows.append({
                "csv_path": str(csv_path),
                "status": "failed",
                "changed_rows": np.nan,
                "error": repr(error),
            })

            if verbose:
                print(
                    csv_path.name,
                    "| failed:",
                    repr(error),
                )

    csv_summary_df = pd.DataFrame(csv_summary_rows)

    if len(all_reports) == 0:
        update_report = pd.DataFrame(
            columns=[
                "summary_path",
                "n_fcc_cells",
                "target_rho",
                "kT",
                "old_phase_separated",
                "new_phase_separated",
                "phase_sep_low_density_fraction",
                "phase_sep_density_threshold",
                "phase_sep_voxel_fraction_threshold",
                "log_path",
            ]
        )
    else:
        update_report = pd.concat(
            all_reports,
            ignore_index=True,
        )

    if verbose:
        print("\nCSV update summary")
        print("=" * 70)

        if len(csv_summary_df) > 0:
            print(csv_summary_df["status"].value_counts(dropna=False))

        print("\nTotal changed rows:", len(update_report))

    return update_report






# ============================================================
# Master V2 CSV helpers
# ============================================================

# ============================================================
# Master CSV path
# ============================================================

def get_v2_master_csv_path(
    n_fcc_cells,
    base_folder=THERMALIZED_STATES_V2_ROOT,
    phase_name="randomization",
):
    """
    Standard path for one n_cells-specific V2 master CSV.

    Example:
        Thermalized_States_v2/
            FCC/
                n_cells_30/
                    Master_Summaries/
                        master_v2_ncells_30_randomization.csv
    """

    master_folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / "Master_Summaries"
    )

    master_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_csv_path = (
        master_folder
        / (
            f"master_v2"
            f"_ncells_{int(n_fcc_cells)}"
            f"_{phase_name}.csv"
        )
    )

    return master_csv_path


def _safe_float(
    value,
    default=np.nan,
):
    """
    Convert HDF5/CSV-ish values to float safely.
    """

    value = _clean_hdf5_attr_value(value)

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return default


def _safe_int(
    value,
    default=np.nan,
):
    """
    Convert HDF5/CSV-ish values to int safely.
    """

    value = _clean_hdf5_attr_value(value)

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return int(value)
    except Exception:
        return default


def _safe_str(
    value,
    default="",
):
    """
    Convert HDF5/CSV-ish values to string safely.
    """

    value = _clean_hdf5_attr_value(value)

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return str(value)


def _safe_bool(
    value,
):
    """
    Convert HDF5/CSV-ish values to True/False/None.
    """

    value = _clean_hdf5_attr_value(value)

    return clean_bool_value(value)


def _infer_state_path_from_log_path(
    log_path,
):
    """
    Infer the matching GSD state path from a log path.

    Example:
        randomization_log.hdf5 -> randomization.gsd
    """

    log_path = Path(log_path)

    if log_path.name.endswith("_log.hdf5"):
        state_path = log_path.with_name(
            log_path.name.replace("_log.hdf5", ".gsd")
        )
    else:
        state_path = log_path.with_suffix(".gsd")

    return state_path


def summarize_one_v2_log_for_master_csv(
    log_path,
    n_last=100,
):
    """
    Summarize one completed V2 HDF5 log into one master-CSV row.

    This does not rerun simulations.
    It reads:
    - HDF5 metadata
    - final-state paths
    - phase-separation metadata
    - last-n logged thermodynamic statistics
    """

    log_path = Path(log_path)

    log = lh.read_hdf5_log(log_path)

    metadata = (
        log.get("metadata", {})
           .get("attrs", {})
    )

    # ============================================================
    # Tail statistics from the HDF5 log
    # ============================================================

    pressure_stats = get_log_tail_stats(
        log=log,
        quantity="pressure",
        n_last=n_last,
    )

    pe_stats = get_log_tail_stats(
        log=log,
        quantity="PE_per_particle",
        n_last=n_last,
    )

    ke_stats = get_log_tail_stats(
        log=log,
        quantity="KE_per_particle",
        n_last=n_last,
    )

    kinetic_temperature_stats = get_log_tail_stats(
        log=log,
        quantity="kinetic_temperature",
        n_last=n_last,
    )

    # ============================================================
    # Phase-separation metadata
    # ============================================================

    phase_metadata = read_phase_separation_metadata_flexible_for_csv(
        log_path=log_path,
    )

    # ============================================================
    # Main metadata values
    # ============================================================

    n_fcc_cells = _safe_int(
        metadata.get("n_fcc_cells", np.nan)
    )

    N = _safe_int(
        metadata.get("N", np.nan)
    )

    target_rho = _safe_float(
        metadata.get("target_rho", np.nan)
    )

    actual_rho = _safe_float(
        metadata.get("actual_rho", np.nan)
    )

    BoxLength = _safe_float(
        metadata.get("BoxLength", np.nan)
    )

    volume = _safe_float(
        metadata.get("volume", np.nan)
    )

    fcc_cell_size = _safe_float(
        metadata.get("fcc_cell_size", np.nan)
    )

    kT = _safe_float(
        metadata.get("kT", np.nan)
    )

    nsteps = _safe_int(
        metadata.get("nsteps", np.nan)
    )

    log_period = _safe_int(
        metadata.get("log_period", np.nan)
    )

    seed = _safe_int(
        metadata.get("seed", np.nan)
    )

    phase_name = _safe_str(
        metadata.get(
            "phase_name",
            log_path.name.replace("_log.hdf5", ""),
        )
    )

    state_path = _safe_str(
        metadata.get(
            "state_path",
            _infer_state_path_from_log_path(log_path),
        )
    )

    # ============================================================
    # Stable key
    # ============================================================

    try:
        master_key = make_v2_sweep_key(
            n_fcc_cells=n_fcc_cells,
            target_rho=target_rho,
            kT=kT,
            nsteps=nsteps,
            log_period=log_period,
            seed=seed,
            phase_name=phase_name,
        )
    except Exception:
        master_key = str(log_path)

    # ============================================================
    # Build row
    # ============================================================

    row = {
        "status": "completed",
        "master_key": master_key,

        "lattice_type": _safe_str(
            metadata.get("lattice_type", "fcc")
        ),
        "density_mode": _safe_str(
            metadata.get("density_mode", "fixed_N_variable_L")
        ),

        "n_fcc_cells": n_fcc_cells,
        "N": N,

        "target_rho": target_rho,
        "actual_rho": actual_rho,

        "BoxLength": BoxLength,
        "volume": volume,
        "fcc_cell_size": fcc_cell_size,

        "kT": kT,
        "nsteps": nsteps,
        "log_period": log_period,
        "seed": seed,
        "phase_name": phase_name,

        "dt": _safe_float(
            metadata.get("dt", np.nan)
        ),
        "epsilon_LJ": _safe_float(
            metadata.get("epsilon_LJ", np.nan)
        ),
        "sigma_LJ": _safe_float(
            metadata.get("sigma_LJ", np.nan)
        ),
        "r_cut_LJ": _safe_float(
            metadata.get("r_cut_LJ", np.nan)
        ),
        "r_on_LJ": _safe_float(
            metadata.get("r_on_LJ", np.nan)
        ),
        "buffer_LJ": _safe_float(
            metadata.get("buffer_LJ", np.nan)
        ),
        "lj_mode": _safe_str(
            metadata.get("lj_mode", "")
        ),

        "phase_separated": phase_metadata["phase_separated"],

        "phase_sep_location": phase_metadata["phase_sep_location"],
        "phase_sep_method": phase_metadata["phase_sep_method"],
        "phase_sep_nbins": phase_metadata["phase_sep_nbins"],
        "phase_sep_density_threshold": phase_metadata[
            "phase_sep_density_threshold"
        ],
        "phase_sep_voxel_fraction_threshold": phase_metadata[
            "phase_sep_voxel_fraction_threshold"
        ],
        "phase_sep_low_density_fraction": phase_metadata[
            "phase_sep_low_density_fraction"
        ],
        "phase_sep_updated_from_saved_gsd": phase_metadata[
            "phase_sep_updated_from_saved_gsd"
        ],
        "phase_sep_voxel_phase_separated": phase_metadata[
            "phase_sep_voxel_phase_separated"
        ],

        "pressure_mean_last100": pressure_stats["mean"],
        "pressure_std_last100": pressure_stats["std"],

        "PE_per_particle_mean_last100": pe_stats["mean"],
        "PE_per_particle_std_last100": pe_stats["std"],

        "KE_per_particle_mean_last100": ke_stats["mean"],
        "KE_per_particle_std_last100": ke_stats["std"],

        "kinetic_temperature_mean_last100": kinetic_temperature_stats["mean"],
        "kinetic_temperature_std_last100": kinetic_temperature_stats["std"],

        "n_values_used": pressure_stats["n_values_used"],
        "first_timestep_used": pressure_stats["first_timestep_used"],
        "last_timestep_used": pressure_stats["last_timestep_used"],

        "final_timestep": _safe_int(
            metadata.get(
                "final_timestep",
                pressure_stats["last_timestep_used"],
            )
        ),

        "starting_state_path": _safe_str(
            metadata.get("starting_state_path", "")
        ),

        "state_path": state_path,
        "log_path": str(log_path),

        "error": "",
        "traceback": "",
    }

    return row


def make_failed_master_csv_row(
    log_path,
    error,
):
    """
    Make a master-CSV row for a log that could not be summarized.
    """

    log_path = Path(log_path)

    row = {
        "status": "failed_to_read",
        "master_key": str(log_path),

        "n_fcc_cells": np.nan,
        "N": np.nan,
        "target_rho": np.nan,
        "actual_rho": np.nan,
        "BoxLength": np.nan,
        "volume": np.nan,
        "fcc_cell_size": np.nan,
        "kT": np.nan,
        "nsteps": np.nan,
        "log_period": np.nan,
        "seed": np.nan,
        "phase_name": log_path.name.replace("_log.hdf5", ""),

        "phase_separated": np.nan,

        "state_path": str(
            _infer_state_path_from_log_path(log_path)
        ),
        "log_path": str(log_path),

        "error": repr(error),
        "traceback": traceback.format_exc(),
    }

    return row


# ============================================================
# Build master V2 CSV for one n_cells value
# ============================================================

def build_v2_master_csv(
    n_fcc_cells,
    base_folder=THERMALIZED_STATES_V2_ROOT,
    phase_name="randomization",
    output_path=None,
    n_last=100,
    dry_run=False,
    backup=True,
    verbose=True,
):
    """
    Build one n_cells-specific V2 master CSV from saved HDF5 logs.

    This searches only inside:

        Thermalized_States_v2/FCC/n_cells_<n_fcc_cells>/

    Example output:

        Thermalized_States_v2/FCC/n_cells_30/Master_Summaries/
            master_v2_ncells_30_randomization.csv

    This does not rerun simulations.
    """

    base_folder = Path(base_folder)

    n_cells_folder = (
        base_folder
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
    )

    if output_path is None:
        output_path = get_v2_master_csv_path(
            n_fcc_cells=n_fcc_cells,
            base_folder=base_folder,
            phase_name=phase_name,
        )

    output_path = Path(output_path)

    log_paths = sorted(
        n_cells_folder.glob(f"**/{phase_name}_log.hdf5")
    )

    if verbose:
        print("Building V2 master CSV")
        print("=" * 70)
        print("n_fcc_cells =", n_fcc_cells)
        print("n_cells_folder =", n_cells_folder)
        print("phase_name =", phase_name)
        print("number of logs found =", len(log_paths))
        print("output_path =", output_path)
        print("n_last =", n_last)
        print("dry_run =", dry_run)
        print("=" * 70)

    rows = []

    for i, log_path in enumerate(log_paths, start=1):
        try:
            row = summarize_one_v2_log_for_master_csv(
                log_path=log_path,
                n_last=n_last,
            )

            rows.append(row)

        except Exception as error:
            rows.append(
                make_failed_master_csv_row(
                    log_path=log_path,
                    error=error,
                )
            )

        if verbose and i % 25 == 0:
            print(f"Processed {i}/{len(log_paths)} logs")

    master_df = pd.DataFrame(rows)

    # ============================================================
    # Sort master table
    # ============================================================

    sort_columns = [
        "kT",
        "target_rho",
        "nsteps",
        "seed",
        "phase_name",
    ]

    sort_columns = [
        col for col in sort_columns
        if col in master_df.columns
    ]

    if len(sort_columns) > 0 and len(master_df) > 0:
        master_df = master_df.sort_values(
            sort_columns,
            ignore_index=True,
            na_position="last",
        )

    # ============================================================
    # Put most useful columns first
    # ============================================================

    preferred_columns = [
        "status",
        "master_key",

        "n_fcc_cells",
        "N",
        "target_rho",
        "actual_rho",
        "BoxLength",
        "volume",
        "fcc_cell_size",

        "kT",
        "nsteps",
        "log_period",
        "seed",
        "phase_name",

        "phase_separated",
        "phase_sep_method",
        "phase_sep_nbins",
        "phase_sep_density_threshold",
        "phase_sep_voxel_fraction_threshold",
        "phase_sep_low_density_fraction",
        "phase_sep_location",
        "phase_sep_updated_from_saved_gsd",
        "phase_sep_voxel_phase_separated",

        "pressure_mean_last100",
        "pressure_std_last100",

        "PE_per_particle_mean_last100",
        "PE_per_particle_std_last100",

        "KE_per_particle_mean_last100",
        "KE_per_particle_std_last100",

        "kinetic_temperature_mean_last100",
        "kinetic_temperature_std_last100",

        "n_values_used",
        "first_timestep_used",
        "last_timestep_used",
        "final_timestep",

        "dt",
        "epsilon_LJ",
        "sigma_LJ",
        "r_cut_LJ",
        "r_on_LJ",
        "buffer_LJ",
        "lj_mode",

        "lattice_type",
        "density_mode",

        "starting_state_path",
        "state_path",
        "log_path",

        "error",
        "traceback",
    ]

    preferred_columns = [
        col for col in preferred_columns
        if col in master_df.columns
    ]

    remaining_columns = [
        col for col in master_df.columns
        if col not in preferred_columns
    ]

    master_df = master_df[
        preferred_columns + remaining_columns
    ]

    # ============================================================
    # Save CSV
    # ============================================================

    if not dry_run:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if backup and output_path.exists():
            timestamp = pd.Timestamp.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            backup_path = output_path.with_name(
                output_path.stem
                + f"_backup_{timestamp}"
                + output_path.suffix
            )

            shutil.copy2(
                output_path,
                backup_path,
            )

            if verbose:
                print("Backup written:")
                print(backup_path)

        master_df.to_csv(
            output_path,
            index=False,
        )

    # ============================================================
    # Print summary
    # ============================================================

    if verbose:
        print("\nMaster CSV summary")
        print("=" * 70)

        if len(master_df) == 0:
            print("No rows found.")
        else:
            print(master_df["status"].value_counts(dropna=False))

            print("\nRows:", len(master_df))

            if not dry_run:
                print("\nWrote:")
                print(output_path)
            else:
                print("\nDry run: did not write CSV.")

    return master_df
