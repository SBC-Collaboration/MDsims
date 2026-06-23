# classification.py

from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import gsd.hoomd

from .paths import THERMALIZED_STATES_V2_ROOT


# ============================================================
# Default voxel phase-separation settings
# ============================================================

DEFAULT_PHASE_SEP_NBINS = 10
DEFAULT_PHASE_SEP_DENSITY_THRESHOLD = 0.2
DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD = 0.01


# ============================================================
# Default PE-drop phase-separation settings
# ============================================================

# PE_drop is defined as:
#
#     PE_drop = last_PE_per_particle_mean - starting_PE_per_particle
#
# So a drop in PE/N is negative.
#
# Example:
#     start = -3.0
#     end   = -5.0
#
#     PE_drop = -5.0 - (-3.0) = -2.0
#
# A state passes the raw PE-drop threshold if:
#
#     PE_drop <= DEFAULT_PHASE_SEP_PE_DROP_THRESHOLD
#
# because more negative means a larger drop.
DEFAULT_PHASE_SEP_PE_DROP_THRESHOLD = -0.15

# Signed Z score:
#
#     PE_drop_z_score = PE_drop / last_PE_per_particle_std
#
# Since PE_drop is negative for a drop, this Z score is also negative.
# A state passes the Z-score threshold if:
#
#     PE_drop_z_score <= -DEFAULT_PHASE_SEP_PE_DROP_Z_LIMIT
DEFAULT_PHASE_SEP_PE_DROP_Z_LIMIT = 5.0

# Use the last N logged points to define the final PE/N state.
DEFAULT_PHASE_SEP_PE_DROP_N_LAST = 100

# How to combine raw threshold and Z-score threshold for:
#
#     metadata/phase_separation/PE_drop.attrs["phase_separated"]
#
# Options:
#     "raw"     -> use only raw PE_drop threshold
#     "z_score" -> use only Z-score threshold
#     "either"  -> raw OR Z-score
#     "both"    -> raw AND Z-score
DEFAULT_PHASE_SEP_PE_DROP_DECISION_RULE = "either"


# ============================================================
# Future defaults, not used yet
# ============================================================

DEFAULT_PHASE_SEP_FIT_POISSON_HEIGHT_LIMIT = None


# ============================================================
# General value cleaners
# ============================================================

def _clean_hdf5_attr_value(value):
    """
    Convert HDF5 attribute values into normal Python values.
    """

    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode()

    if hasattr(value, "shape") and value.shape == ():
        return value.item()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _safe_bool(value):
    """
    Convert HDF5/CSV-ish values to bool, or None if missing.
    """

    value = _clean_hdf5_attr_value(value)

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower in ["true", "1", "yes"]:
            return True

        if value_lower in ["false", "0", "no"]:
            return False

        return None

    return bool(value)


def _safe_float(value, default=np.nan):
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


def _safe_int(value, default=np.nan):
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


def _safe_str(value, default=""):
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


# ============================================================
# Convert input object to snapshot-like object
# ============================================================

def _as_snapshot(obj):
    """
    Accept:
    - result dictionary from sh.get_or_make_thermalized_state(...)
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame

    Return something with:
    - configuration.box
    - particles.position
    """

    if isinstance(obj, dict):
        if "frame" in obj and obj["frame"] is not None:
            return _as_snapshot(obj["frame"])

        if "simulation" in obj and obj["simulation"] is not None:
            return _as_snapshot(obj["simulation"])

        raise TypeError(
            "Result dictionary does not contain a usable frame or simulation."
        )

    if obj is None:
        raise TypeError(
            "Cannot convert None to snapshot/frame."
        )

    if hasattr(obj, "state") and hasattr(obj.state, "get_snapshot"):
        return obj.state.get_snapshot()

    if hasattr(obj, "get_snapshot"):
        return obj.get_snapshot()

    if hasattr(obj, "configuration") and hasattr(obj, "particles"):
        return obj

    raise TypeError(
        "Expected a result dictionary, HOOMD simulation, "
        "HOOMD state, HOOMD snapshot, or GSD frame."
    )


# ============================================================
# Position helpers
# ============================================================

def _wrap_positions_into_box(
    positions,
    Lx,
    Ly,
    Lz,
):
    """
    Wrap positions into [-L/2, L/2).

    This makes the voxel counting robust if saved positions are slightly
    outside the primary box.
    """

    positions = np.asarray(
        positions,
        dtype=np.float64,
    ).copy()

    box_lengths = np.array(
        [Lx, Ly, Lz],
        dtype=np.float64,
    )

    positions = (
        (positions + 0.5 * box_lengths)
        % box_lengths
        - 0.5 * box_lengths
    )

    return positions


def _get_positions_and_box(obj):
    """
    Extract positions and box lengths from any supported object.
    """

    snapshot = _as_snapshot(obj)

    positions = np.asarray(
        snapshot.particles.position,
        dtype=np.float64,
    )

    box = np.asarray(
        snapshot.configuration.box,
        dtype=np.float64,
    )

    Lx = float(box[0])
    Ly = float(box[1])
    Lz = float(box[2])

    positions = _wrap_positions_into_box(
        positions=positions,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
    )

    return positions, Lx, Ly, Lz, snapshot


# ============================================================
# Voxel-density calculation
# ============================================================

def compute_voxel_densities(
    obj,
    nbins=DEFAULT_PHASE_SEP_NBINS,
):
    """
    Compute voxel densities.

    Input can be:
    - result dictionary
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame

    Returns
    -------
    voxel_densities, voxel_counts, voxel_volume
    """

    nbins = int(nbins)

    if nbins <= 0:
        raise ValueError("nbins must be positive")

    positions, Lx, Ly, Lz, snapshot = _get_positions_and_box(obj)

    bounds = [
        [-Lx / 2.0, Lx / 2.0],
        [-Ly / 2.0, Ly / 2.0],
        [-Lz / 2.0, Lz / 2.0],
    ]

    voxel_volume = (
        (Lx / nbins)
        * (Ly / nbins)
        * (Lz / nbins)
    )

    voxel_counts, _ = np.histogramdd(
        positions,
        bins=nbins,
        range=bounds,
    )

    voxel_counts = voxel_counts.ravel()
    voxel_densities = voxel_counts / voxel_volume

    return voxel_densities, voxel_counts, voxel_volume


# ============================================================
# Voxel fraction phase-separation test
# ============================================================

def compute_voxel_fraction_phase_separation(
    obj,
    nbins=DEFAULT_PHASE_SEP_NBINS,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
):
    """
    Compute phase separation using the low-density voxel fraction.

    Rule:
    - Split the box into nbins x nbins x nbins voxels.
    - Compute density in each voxel.
    - Count the fraction of voxels with density below density_threshold.
    - If that fraction is larger than voxel_fraction_threshold,
      phase_separated = True.
    """

    voxel_densities, voxel_counts, voxel_volume = compute_voxel_densities(
        obj=obj,
        nbins=nbins,
    )

    low_density_fraction = float(
        np.mean(voxel_densities < density_threshold)
    )

    phase_separated = bool(
        low_density_fraction > voxel_fraction_threshold
    )

    result = {
        "phase_separated": phase_separated,
        "method": "voxel_low_density_fraction",

        "nbins": int(nbins),
        "density_threshold": float(density_threshold),
        "voxel_fraction_threshold": float(voxel_fraction_threshold),

        "low_density_fraction": low_density_fraction,

        "n_voxels": int(len(voxel_densities)),
        "voxel_volume": float(voxel_volume),

        "min_voxel_density": float(np.min(voxel_densities)),
        "max_voxel_density": float(np.max(voxel_densities)),
        "mean_voxel_density": float(np.mean(voxel_densities)),
        "std_voxel_density": float(np.std(voxel_densities, ddof=1)),
    }

    return result


def compute_phase_separation_from_frame(
    frame,
    nbins=DEFAULT_PHASE_SEP_NBINS,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
):
    """
    Backward-compatible name for the voxel fraction test.
    """

    return compute_voxel_fraction_phase_separation(
        obj=frame,
        nbins=nbins,
        density_threshold=density_threshold,
        voxel_fraction_threshold=voxel_fraction_threshold,
    )


def compute_phase_separation(
    obj,
    nbins=DEFAULT_PHASE_SEP_NBINS,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
):
    """
    Main frame-based phase-separation calculation for now.

    Currently this runs only the voxel fraction test.

    PE_drop is log-based, not frame-based, so it is handled separately by:

        compute_PE_drop_phase_separation_from_log(...)
    """

    return compute_voxel_fraction_phase_separation(
        obj=obj,
        nbins=nbins,
        density_threshold=density_threshold,
        voxel_fraction_threshold=voxel_fraction_threshold,
    )


def check_phase_separated(
    obj,
    nbins=DEFAULT_PHASE_SEP_NBINS,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
):
    """
    Return only the boolean voxel phase-separated result.
    """

    result = compute_phase_separation(
        obj=obj,
        nbins=nbins,
        density_threshold=density_threshold,
        voxel_fraction_threshold=voxel_fraction_threshold,
    )

    return bool(result["phase_separated"])


# ============================================================
# HDF5 log helpers
# ============================================================

def _read_metadata_attrs_from_hdf5(
    log_path,
):
    """
    Read metadata.attrs from one HDF5 log.
    """

    log_path = Path(log_path)

    attrs = {}

    with h5py.File(log_path, mode="r") as hdf:
        if "metadata" not in hdf:
            return attrs

        for key, value in hdf["metadata"].attrs.items():
            attrs[key] = _clean_hdf5_attr_value(value)

    return attrs


def _read_PE_arrays_from_hdf5(
    log_path,
):
    """
    Read timestep, total PE, and PE/N from a saved HDF5 log.
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    metadata = _read_metadata_attrs_from_hdf5(
        log_path=log_path,
    )

    if "N" not in metadata:
        raise KeyError(f"metadata.attrs['N'] is missing in log: {log_path}")

    N = int(metadata["N"])

    with h5py.File(log_path, mode="r") as hdf:
        timestep = np.asarray(
            hdf["hoomd-data"]["Simulation"]["timestep"],
            dtype=int,
        )

        potential_energy = np.asarray(
            hdf["hoomd-data"]["md"]
               ["compute"]
               ["ThermodynamicQuantities"]
               ["potential_energy"],
            dtype=float,
        )

    if len(potential_energy) == 0:
        raise ValueError(
            f"No potential_energy values found in log: {log_path}"
        )

    if len(timestep) != len(potential_energy):
        raise ValueError(
            "Timestep and potential_energy arrays have different lengths."
        )

    PE_per_particle = potential_energy / N

    return timestep, potential_energy, PE_per_particle, metadata


# ============================================================
# Last-N log statistics
# ============================================================

def _compute_last_n_stats(
    values,
    n_last=DEFAULT_PHASE_SEP_PE_DROP_N_LAST,
):
    """
    Compute only mean/std for the last n_last values.

    If fewer than n_last values exist, use all values.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        raise ValueError("Cannot compute last-window stats for empty values.")

    n_last = int(n_last)

    if n_last <= 0:
        raise ValueError("n_last must be positive")

    n_used = min(
        n_last,
        len(values),
    )

    window_values = values[-n_used:]

    if len(window_values) > 1:
        std = float(
            np.std(
                window_values,
                ddof=1,
            )
        )
    else:
        std = 0.0

    stats = {
        "mean": float(np.mean(window_values)),
        "std": std,
    }

    return stats


# ============================================================
# PE-drop phase-separation test
# ============================================================

def _combine_PE_drop_decisions(
    passes_PE_drop_threshold,
    passes_z_score_threshold,
    decision_rule=DEFAULT_PHASE_SEP_PE_DROP_DECISION_RULE,
):
    """
    Combine raw-threshold and Z-score decisions.

    Options:
        raw
        z_score
        either
        both
    """

    decision_rule = str(decision_rule).strip().lower()

    if decision_rule == "raw":
        return bool(passes_PE_drop_threshold)

    if decision_rule == "z_score":
        return bool(passes_z_score_threshold)

    if decision_rule == "either":
        return bool(
            passes_PE_drop_threshold
            or passes_z_score_threshold
        )

    if decision_rule == "both":
        return bool(
            passes_PE_drop_threshold
            and passes_z_score_threshold
        )

    raise ValueError(
        "decision_rule must be one of: "
        "'raw', 'z_score', 'either', 'both'"
    )


# ============================================================
# PE-drop phase-separation test
# ============================================================

def compute_PE_drop_phase_separation_from_log(
    log_path,
    PE_drop_threshold=DEFAULT_PHASE_SEP_PE_DROP_THRESHOLD,
    z_score_threshold=DEFAULT_PHASE_SEP_PE_DROP_Z_LIMIT,
    n_last=DEFAULT_PHASE_SEP_PE_DROP_N_LAST,
    decision_rule=DEFAULT_PHASE_SEP_PE_DROP_DECISION_RULE,
):
    """
    Compute phase separation using a PE/N drop test.

    Stored metadata is intentionally minimal.

    Definition:
        PE_drop = last_PE_per_particle_mean - starting_PE_per_particle

    So a real PE/N drop is negative.
    """

    log_path = Path(log_path)

    timestep, potential_energy, PE_per_particle, metadata = _read_PE_arrays_from_hdf5(
        log_path=log_path,
    )

    last_PE_per_particle_stats = _compute_last_n_stats(
        values=PE_per_particle,
        n_last=n_last,
    )

    last_total_PE_stats = _compute_last_n_stats(
        values=potential_energy,
        n_last=n_last,
    )

    starting_PE_per_particle = float(PE_per_particle[0])
    starting_total_PE = float(potential_energy[0])

    last_PE_per_particle_mean = float(last_PE_per_particle_stats["mean"])
    last_PE_per_particle_std = float(last_PE_per_particle_stats["std"])

    last_total_PE_mean = float(last_total_PE_stats["mean"])
    last_total_PE_std = float(last_total_PE_stats["std"])

    # Negative means PE/N dropped.
    PE_drop = float(
        last_PE_per_particle_mean - starting_PE_per_particle
    )

    # Negative means total PE dropped.
    total_PE_drop = float(
        last_total_PE_mean - starting_total_PE
    )

    if last_PE_per_particle_std > 0:
        PE_drop_z_score = float(
            PE_drop / last_PE_per_particle_std
        )

    else:
        if PE_drop < 0:
            PE_drop_z_score = float(-np.inf)
        elif PE_drop > 0:
            PE_drop_z_score = float(np.inf)
        else:
            PE_drop_z_score = 0.0

    passes_PE_drop_threshold = bool(
        PE_drop <= PE_drop_threshold
    )

    passes_z_score_threshold = bool(
        PE_drop_z_score <= -float(z_score_threshold)
    )

    phase_separated = _combine_PE_drop_decisions(
        passes_PE_drop_threshold=passes_PE_drop_threshold,
        passes_z_score_threshold=passes_z_score_threshold,
        decision_rule=decision_rule,
    )

    result = {
        "phase_separated": bool(phase_separated),
        "method": "PE_drop",

        "quantity": "PE_per_particle",
        "definition": "PE_drop = last_PE_per_particle_mean - starting_PE_per_particle",

        "decision_rule": str(decision_rule),

        "PE_drop_threshold": float(PE_drop_threshold),
        "z_score_threshold": float(z_score_threshold),

        "passes_PE_drop_threshold": bool(passes_PE_drop_threshold),
        "passes_z_score_threshold": bool(passes_z_score_threshold),

        "starting_PE_per_particle": float(starting_PE_per_particle),
        "starting_total_PE": float(starting_total_PE),

        "last_PE_per_particle_mean": float(last_PE_per_particle_mean),
        "last_PE_per_particle_std": float(last_PE_per_particle_std),

        "last_total_PE_mean": float(last_total_PE_mean),
        "last_total_PE_std": float(last_total_PE_std),

        "PE_drop": float(PE_drop),
        "PE_drop_z_score": float(PE_drop_z_score),

        "total_PE_drop": float(total_PE_drop),

        "updated_from_saved_log": True,
    }

    return result


# ============================================================
# Find matching GSD state path from log path
# ============================================================

def _get_state_path_for_phase_log(
    log_path,
):
    """
    Find the matching GSD state path for a phase log.

    Preferred:
    - use metadata.attrs["state_path"]

    Fallback:
    - randomization_log.hdf5 -> randomization.gsd
    """

    log_path = Path(log_path)

    state_path = None

    if log_path.exists():
        with h5py.File(log_path, mode="r") as hdf:
            if "metadata" in hdf:
                raw_state_path = hdf["metadata"].attrs.get(
                    "state_path",
                    None,
                )

                raw_state_path = _clean_hdf5_attr_value(
                    raw_state_path
                )

                if raw_state_path not in [None, ""]:
                    state_path = Path(raw_state_path)

    if state_path is not None and state_path.exists():
        return state_path

    if log_path.name.endswith("_log.hdf5"):
        state_path = log_path.with_name(
            log_path.name.replace("_log.hdf5", ".gsd")
        )

        if state_path.exists():
            return state_path

    return None


# ============================================================
# Generic phase-method metadata writer
# ============================================================

def write_phase_method_metadata(
    log_path,
    method_name,
    attrs,
    set_main_phase_separated=None,
    clear_existing=True,
    clear_parent_attrs=False,
):
    """
    Write one method group under:

        metadata/phase_separation/{method_name}.attrs[...]

    Examples:
        method_name = "voxel"
        method_name = "fit"
        method_name = "PE_drop"

    This is the helper that future fit and PE_drop tests should use.
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    with h5py.File(log_path, mode="a") as hdf:
        metadata_group = hdf.require_group("metadata")

        if set_main_phase_separated is not None:
            metadata_group.attrs["phase_separated"] = bool(
                set_main_phase_separated
            )

        phase_group = metadata_group.require_group(
            "phase_separation"
        )

        if clear_parent_attrs:
            for key in list(phase_group.attrs.keys()):
                del phase_group.attrs[key]

        method_group = phase_group.require_group(method_name)

        if clear_existing:
            for key in list(method_group.attrs.keys()):
                del method_group.attrs[key]

        for key, value in attrs.items():
            if value is None:
                continue

            value = _clean_hdf5_attr_value(value)

            if isinstance(value, Path):
                value = str(value)

            method_group.attrs[key] = value


# ============================================================
# Write voxel phase-separation metadata
# ============================================================

def write_voxel_phase_separation_metadata(
    log_path,
    state_path=None,
    nbins=DEFAULT_PHASE_SEP_NBINS,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
    updated_from_saved_gsd=True,
    dry_run=False,
):
    """
    Fill out the voxel phase-separation metadata for one completed simulation.

    Writes the main boolean here:

        metadata.attrs["phase_separated"]

    and writes voxel-method details here:

        metadata/phase_separation/voxel.attrs["phase_separated"]

    This does not rerun the simulation.
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    if state_path is None:
        state_path = _get_state_path_for_phase_log(log_path)

    if state_path is None:
        raise FileNotFoundError(
            f"Could not find matching GSD state for log: {log_path}"
        )

    state_path = Path(state_path)

    if not state_path.exists():
        raise FileNotFoundError(f"State file does not exist: {state_path}")

    old_phase_separated = None
    n_fcc_cells = np.nan
    target_rho = np.nan
    kT = np.nan

    with h5py.File(log_path, mode="r") as hdf:
        if "metadata" in hdf:
            metadata_attrs = hdf["metadata"].attrs

            old_phase_separated = _safe_bool(
                metadata_attrs.get("phase_separated", None)
            )

            n_fcc_cells = _safe_int(
                metadata_attrs.get("n_fcc_cells", np.nan)
            )

            target_rho = _safe_float(
                metadata_attrs.get("target_rho", np.nan)
            )

            kT = _safe_float(
                metadata_attrs.get("kT", np.nan)
            )

    with gsd.hoomd.open(
        name=str(state_path),
        mode="r",
    ) as traj:
        frame = traj[-1]

    result = compute_voxel_fraction_phase_separation(
        obj=frame,
        nbins=nbins,
        density_threshold=density_threshold,
        voxel_fraction_threshold=voxel_fraction_threshold,
    )

    new_phase_separated = bool(result["phase_separated"])

    changed = (
        old_phase_separated is None
        or old_phase_separated != new_phase_separated
    )

    voxel_attrs = {
        "phase_separated": new_phase_separated,
        "method": result["method"],

        "nbins": int(result["nbins"]),
        "density_threshold": float(result["density_threshold"]),
        "voxel_fraction_threshold": float(
            result["voxel_fraction_threshold"]
        ),
        "low_density_fraction": float(result["low_density_fraction"]),

        "n_voxels": int(result["n_voxels"]),
        "voxel_volume": float(result["voxel_volume"]),

        "min_voxel_density": float(result["min_voxel_density"]),
        "max_voxel_density": float(result["max_voxel_density"]),
        "mean_voxel_density": float(result["mean_voxel_density"]),
        "std_voxel_density": float(result["std_voxel_density"]),

        "updated_from_saved_gsd": bool(updated_from_saved_gsd),
    }

    if not dry_run:
        write_phase_method_metadata(
            log_path=log_path,
            method_name="voxel",
            attrs=voxel_attrs,
            set_main_phase_separated=new_phase_separated,
            clear_existing=True,
            clear_parent_attrs=True,
        )

    row = {
        "status": "dry_run" if dry_run else "updated",

        "n_fcc_cells": n_fcc_cells,
        "target_rho": target_rho,
        "kT": kT,

        "old_phase_separated": old_phase_separated,
        "new_phase_separated": new_phase_separated,
        "changed": bool(changed),

        "phase_separated": new_phase_separated,
        "method": result["method"],
        "nbins": int(nbins),
        "density_threshold": float(density_threshold),
        "voxel_fraction_threshold": float(voxel_fraction_threshold),
        "low_density_fraction": float(result["low_density_fraction"]),

        "log_path": str(log_path),
        "state_path": str(state_path),
    }

    return row


# Backward-compatible name
write_phase_separation_metadata = write_voxel_phase_separation_metadata


# ============================================================
# Write PE-drop phase-separation metadata
# ============================================================

def write_PE_drop_phase_separation_metadata(
    log_path,
    PE_drop_threshold=DEFAULT_PHASE_SEP_PE_DROP_THRESHOLD,
    z_score_threshold=DEFAULT_PHASE_SEP_PE_DROP_Z_LIMIT,
    n_last=DEFAULT_PHASE_SEP_PE_DROP_N_LAST,
    decision_rule=DEFAULT_PHASE_SEP_PE_DROP_DECISION_RULE,
    dry_run=False,
):
    """
    Compute and write PE-drop phase-separation metadata.

    Writes only to:

        metadata/phase_separation/PE_drop.attrs[...]

    This deliberately does NOT overwrite:

        metadata.attrs["phase_separated"]

    Old PE_drop metadata attrs are deleted before writing the new smaller set.
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    old_PE_drop_phase_separated = None
    old_main_phase_separated = None
    n_fcc_cells = np.nan
    target_rho = np.nan
    kT = np.nan

    with h5py.File(log_path, mode="r") as hdf:
        if "metadata" in hdf:
            metadata_attrs = hdf["metadata"].attrs

            old_main_phase_separated = _safe_bool(
                metadata_attrs.get("phase_separated", None)
            )

            n_fcc_cells = _safe_int(
                metadata_attrs.get("n_fcc_cells", np.nan)
            )

            target_rho = _safe_float(
                metadata_attrs.get("target_rho", np.nan)
            )

            kT = _safe_float(
                metadata_attrs.get("kT", np.nan)
            )

            if "phase_separation" in hdf["metadata"]:
                phase_group = hdf["metadata"]["phase_separation"]

                if "PE_drop" in phase_group:
                    old_PE_drop_phase_separated = _safe_bool(
                        phase_group["PE_drop"].attrs.get(
                            "phase_separated",
                            None,
                        )
                    )

    result = compute_PE_drop_phase_separation_from_log(
        log_path=log_path,
        PE_drop_threshold=PE_drop_threshold,
        z_score_threshold=z_score_threshold,
        n_last=n_last,
        decision_rule=decision_rule,
    )

    new_PE_drop_phase_separated = bool(
        result["phase_separated"]
    )

    changed = (
        old_PE_drop_phase_separated is None
        or old_PE_drop_phase_separated != new_PE_drop_phase_separated
    )

    # This is the full and only PE_drop metadata that will be written.
    attrs = {
        "phase_separated": bool(result["phase_separated"]),
        "method": result["method"],

        "quantity": result["quantity"],
        "definition": result["definition"],

        "decision_rule": result["decision_rule"],

        "PE_drop_threshold": float(result["PE_drop_threshold"]),
        "z_score_threshold": float(result["z_score_threshold"]),

        "passes_PE_drop_threshold": bool(result["passes_PE_drop_threshold"]),
        "passes_z_score_threshold": bool(result["passes_z_score_threshold"]),

        "starting_PE_per_particle": float(result["starting_PE_per_particle"]),
        "starting_total_PE": float(result["starting_total_PE"]),

        "last_PE_per_particle_mean": float(result["last_PE_per_particle_mean"]),
        "last_PE_per_particle_std": float(result["last_PE_per_particle_std"]),

        "last_total_PE_mean": float(result["last_total_PE_mean"]),
        "last_total_PE_std": float(result["last_total_PE_std"]),

        "PE_drop": float(result["PE_drop"]),
        "PE_drop_z_score": float(result["PE_drop_z_score"]),

        "total_PE_drop": float(result["total_PE_drop"]),

        "updated_from_saved_log": bool(result["updated_from_saved_log"]),
    }

    if not dry_run:
        write_phase_method_metadata(
            log_path=log_path,
            method_name="PE_drop",
            attrs=attrs,
            set_main_phase_separated=None,
            clear_existing=True,
            clear_parent_attrs=False,
        )

    row = {
        "status": "dry_run" if dry_run else "updated",

        "n_fcc_cells": n_fcc_cells,
        "target_rho": target_rho,
        "kT": kT,

        "old_main_phase_separated": old_main_phase_separated,
        "old_PE_drop_phase_separated": old_PE_drop_phase_separated,
        "new_PE_drop_phase_separated": new_PE_drop_phase_separated,
        "changed": bool(changed),

        "decision_rule": result["decision_rule"],

        "passes_PE_drop_threshold": result["passes_PE_drop_threshold"],
        "passes_z_score_threshold": result["passes_z_score_threshold"],

        "starting_PE_per_particle": result["starting_PE_per_particle"],
        "last_PE_per_particle_mean": result["last_PE_per_particle_mean"],
        "last_PE_per_particle_std": result["last_PE_per_particle_std"],

        "PE_drop": result["PE_drop"],
        "PE_drop_threshold": result["PE_drop_threshold"],

        "PE_drop_z_score": result["PE_drop_z_score"],
        "z_score_threshold": result["z_score_threshold"],

        "log_path": str(log_path),
    }

    return row


# ============================================================
# Update all V2 voxel phase-separation metadata
# ============================================================

def update_all_v2_phase_separation_metadata(
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
    nbins=DEFAULT_PHASE_SEP_NBINS,
    base_folder=THERMALIZED_STATES_V2_ROOT,
    phase_name="randomization",
    dry_run=False,
    verbose=True,
    show_only_changed=True,
):
    """
    Recompute and update voxel phase-separation metadata for all completed V2 runs.

    This searches for:

        Thermalized_States_v2/**/{phase_name}_log.hdf5

    Then for each matching log:
    - finds the matching saved GSD state
    - recomputes voxel phase separation
    - writes metadata.attrs["phase_separated"]
    - writes metadata/phase_separation/voxel.attrs[...]

    It does not rerun simulations.
    """

    base_folder = Path(base_folder)

    log_paths = sorted(
        base_folder.glob(f"**/{phase_name}_log.hdf5")
    )

    if verbose:
        print("Updating V2 phase-separation metadata")
        print("=" * 70)
        print("base_folder =", base_folder)
        print("phase_name =", phase_name)
        print("number of logs found =", len(log_paths))
        print("nbins =", nbins)
        print("density_threshold =", density_threshold)
        print("voxel_fraction_threshold =", voxel_fraction_threshold)
        print("dry_run =", dry_run)
        print("=" * 70)

    rows = []

    for i, log_path in enumerate(log_paths, start=1):
        try:
            row = write_voxel_phase_separation_metadata(
                log_path=log_path,
                state_path=None,
                nbins=nbins,
                density_threshold=density_threshold,
                voxel_fraction_threshold=voxel_fraction_threshold,
                updated_from_saved_gsd=True,
                dry_run=dry_run,
            )

            rows.append(row)

        except Exception as error:
            rows.append({
                "status": "failed",
                "log_path": str(log_path),
                "state_path": "",
                "old_phase_separated": np.nan,
                "new_phase_separated": np.nan,
                "changed": np.nan,
                "nbins": int(nbins),
                "density_threshold": float(density_threshold),
                "voxel_fraction_threshold": float(voxel_fraction_threshold),
                "low_density_fraction": np.nan,
                "error": repr(error),
            })

        if verbose and i % 25 == 0:
            print(f"Processed {i}/{len(log_paths)} logs")

    update_df = pd.DataFrame(rows)

    if verbose:
        print("\nUpdate summary")
        print("=" * 70)

        if len(update_df) == 0:
            print("No logs found.")
        else:
            print(update_df["status"].value_counts(dropna=False))

            if "changed" in update_df.columns:
                print("\nChanged counts")
                print(update_df["changed"].value_counts(dropna=False))

    if (
        show_only_changed
        and len(update_df) > 0
        and "changed" in update_df.columns
    ):
        update_df = update_df[update_df["changed"] == True].copy()

        keep_columns = [
            "n_fcc_cells",
            "target_rho",
            "kT",
            "old_phase_separated",
            "new_phase_separated",
            "low_density_fraction",
            "density_threshold",
            "voxel_fraction_threshold",
            "nbins",
            "status",
            "log_path",
            "state_path",
        ]

        keep_columns = [
            col for col in keep_columns
            if col in update_df.columns
        ]

        update_df = update_df[keep_columns].reset_index(drop=True)

    return update_df


# ============================================================
# Update all V2 PE-drop metadata
# ============================================================

def update_all_v2_PE_drop_metadata(
    PE_drop_threshold=DEFAULT_PHASE_SEP_PE_DROP_THRESHOLD,
    z_score_threshold=DEFAULT_PHASE_SEP_PE_DROP_Z_LIMIT,
    n_last=DEFAULT_PHASE_SEP_PE_DROP_N_LAST,
    decision_rule=DEFAULT_PHASE_SEP_PE_DROP_DECISION_RULE,
    base_folder=THERMALIZED_STATES_V2_ROOT,
    phase_name="randomization",
    dry_run=False,
    verbose=True,
    show_only_changed=True,
):
    """
    Recompute and update PE-drop metadata for all completed V2 logs.

    This searches for:

        Thermalized_States_v2/**/{phase_name}_log.hdf5

    and writes:

        metadata/phase_separation/PE_drop.attrs[...]

    It does not overwrite metadata.attrs["phase_separated"].
    """

    base_folder = Path(base_folder)

    log_paths = sorted(
        base_folder.glob(f"**/{phase_name}_log.hdf5")
    )

    if verbose:
        print("Updating V2 PE-drop metadata")
        print("=" * 70)
        print("base_folder =", base_folder)
        print("phase_name =", phase_name)
        print("number of logs found =", len(log_paths))
        print("PE_drop_threshold =", PE_drop_threshold)
        print("z_score_threshold =", z_score_threshold)
        print("n_last =", n_last)
        print("decision_rule =", decision_rule)
        print("dry_run =", dry_run)
        print("=" * 70)

    rows = []

    for i, log_path in enumerate(log_paths, start=1):
        try:
            row = write_PE_drop_phase_separation_metadata(
                log_path=log_path,
                PE_drop_threshold=PE_drop_threshold,
                z_score_threshold=z_score_threshold,
                n_last=n_last,
                decision_rule=decision_rule,
                dry_run=dry_run,
            )

            rows.append(row)

        except Exception as error:
            rows.append({
                "status": "failed",
                "log_path": str(log_path),

                "old_PE_drop_phase_separated": np.nan,
                "new_PE_drop_phase_separated": np.nan,
                "changed": np.nan,

                "decision_rule": decision_rule,

                "passes_PE_drop_threshold": np.nan,
                "passes_z_score_threshold": np.nan,

                "starting_PE_per_particle": np.nan,
                "last_PE_per_particle_mean": np.nan,
                "last_PE_per_particle_std": np.nan,

                "PE_drop": np.nan,
                "PE_drop_threshold": float(PE_drop_threshold),

                "PE_drop_z_score": np.nan,
                "z_score_threshold": float(z_score_threshold),

                "error": repr(error),
            })

        if verbose and i % 25 == 0:
            print(f"Processed {i}/{len(log_paths)} logs")

    update_df = pd.DataFrame(rows)

    if verbose:
        print("\nPE-drop update summary")
        print("=" * 70)

        if len(update_df) == 0:
            print("No logs found.")
        else:
            print(update_df["status"].value_counts(dropna=False))

            if "changed" in update_df.columns:
                print("\nChanged counts")
                print(update_df["changed"].value_counts(dropna=False))

    if (
        show_only_changed
        and len(update_df) > 0
        and "changed" in update_df.columns
    ):
        update_df = update_df[update_df["changed"] == True].copy()

        keep_columns = [
            "n_fcc_cells",
            "target_rho",
            "kT",

            "old_main_phase_separated",
            "old_PE_drop_phase_separated",
            "new_PE_drop_phase_separated",

            "decision_rule",

            "passes_PE_drop_threshold",
            "passes_z_score_threshold",

            "starting_PE_per_particle",
            "last_PE_per_particle_mean",
            "last_PE_per_particle_std",

            "PE_drop",
            "PE_drop_threshold",

            "PE_drop_z_score",
            "z_score_threshold",

            "status",
            "log_path",
        ]

        keep_columns = [
            col for col in keep_columns
            if col in update_df.columns
        ]

        update_df = update_df[keep_columns].reset_index(drop=True)

    return update_df


# ============================================================
# Read phase-separation metadata for CSVs
# ============================================================

def read_phase_separation_metadata_for_csv(
    log_path,
):
    """
    Read phase-separation metadata from one HDF5 log.

    Supports:
        metadata/phase_separation/voxel.attrs[...]
        metadata/phase_separation/PE_drop.attrs[...]

    Main output value:
        phase_separated

    For now, phase_separated still means metadata.attrs["phase_separated"],
    which is normally controlled by the voxel method.
    """

    log_path = Path(log_path)

    output = {
        "phase_separated": np.nan,

        # Voxel columns
        "phase_sep_location": "",
        "phase_sep_method": "",
        "phase_sep_nbins": np.nan,
        "phase_sep_density_threshold": np.nan,
        "phase_sep_voxel_fraction_threshold": np.nan,
        "phase_sep_low_density_fraction": np.nan,
        "phase_sep_updated_from_saved_gsd": None,
        "phase_sep_voxel_phase_separated": None,

        # PE-drop columns
        "phase_sep_PE_drop_phase_separated": None,
        "phase_sep_PE_drop_method": "",
        "phase_sep_PE_drop_decision_rule": "",

        "phase_sep_PE_drop_passes_raw_threshold": None,
        "phase_sep_PE_drop_passes_z_threshold": None,

        "phase_sep_PE_drop": np.nan,
        "phase_sep_PE_drop_threshold": np.nan,

        "phase_sep_PE_drop_z_score": np.nan,
        "phase_sep_PE_drop_z_threshold": np.nan,

        "phase_sep_PE_drop_starting_PE_per_particle": np.nan,
        "phase_sep_PE_drop_last_PE_per_particle_mean": np.nan,
        "phase_sep_PE_drop_last_PE_per_particle_std": np.nan,

        "phase_sep_PE_drop_starting_total_PE": np.nan,
        "phase_sep_PE_drop_last_total_PE_mean": np.nan,
        "phase_sep_PE_drop_last_total_PE_std": np.nan,
        "phase_sep_PE_drop_total_PE_drop": np.nan,

        "phase_sep_PE_drop_updated_from_saved_log": None,
    }

    with h5py.File(log_path, mode="r") as hdf:
        if "metadata" not in hdf:
            return output

        metadata_group = hdf["metadata"]

        main_phase_separated = _safe_bool(
            metadata_group.attrs.get(
                "phase_separated",
                None,
            )
        )

        if main_phase_separated is not None:
            output["phase_separated"] = main_phase_separated

        if "phase_separation" not in metadata_group:
            return output

        phase_group = metadata_group["phase_separation"]

        # ========================================================
        # Voxel metadata
        # ========================================================

        if "voxel" in phase_group:
            voxel_attrs = phase_group["voxel"].attrs
            output["phase_sep_location"] = (
                "metadata/phase_separation/voxel"
            )

        else:
            # Backward compatibility with old parent attrs.
            voxel_attrs = phase_group.attrs
            output["phase_sep_location"] = (
                "metadata/phase_separation"
            )

        voxel_phase_separated = _safe_bool(
            voxel_attrs.get(
                "phase_separated",
                None,
            )
        )

        output["phase_sep_voxel_phase_separated"] = voxel_phase_separated

        if main_phase_separated is None and voxel_phase_separated is not None:
            output["phase_separated"] = voxel_phase_separated

        output["phase_sep_method"] = _safe_str(
            voxel_attrs.get("method", "")
        )

        output["phase_sep_nbins"] = _safe_int(
            voxel_attrs.get("nbins", np.nan)
        )

        output["phase_sep_density_threshold"] = _safe_float(
            voxel_attrs.get("density_threshold", np.nan)
        )

        output["phase_sep_voxel_fraction_threshold"] = _safe_float(
            voxel_attrs.get("voxel_fraction_threshold", np.nan)
        )

        output["phase_sep_low_density_fraction"] = _safe_float(
            voxel_attrs.get("low_density_fraction", np.nan)
        )

        output["phase_sep_updated_from_saved_gsd"] = _safe_bool(
            voxel_attrs.get("updated_from_saved_gsd", None)
        )

        # ========================================================
        # PE-drop metadata
        # ========================================================

        if "PE_drop" in phase_group:
            PE_attrs = phase_group["PE_drop"].attrs

            output["phase_sep_PE_drop_phase_separated"] = _safe_bool(
                PE_attrs.get("phase_separated", None)
            )
            
            output["phase_sep_PE_drop_method"] = _safe_str(
                PE_attrs.get("method", "")
            )
            
            output["phase_sep_PE_drop_decision_rule"] = _safe_str(
                PE_attrs.get("decision_rule", "")
            )
            
            output["phase_sep_PE_drop_passes_raw_threshold"] = _safe_bool(
                PE_attrs.get("passes_PE_drop_threshold", None)
            )
            
            output["phase_sep_PE_drop_passes_z_threshold"] = _safe_bool(
                PE_attrs.get("passes_z_score_threshold", None)
            )
            
            output["phase_sep_PE_drop"] = _safe_float(
                PE_attrs.get("PE_drop", np.nan)
            )
            
            output["phase_sep_PE_drop_threshold"] = _safe_float(
                PE_attrs.get("PE_drop_threshold", np.nan)
            )
            
            output["phase_sep_PE_drop_z_score"] = _safe_float(
                PE_attrs.get("PE_drop_z_score", np.nan)
            )
            
            output["phase_sep_PE_drop_z_threshold"] = _safe_float(
                PE_attrs.get("z_score_threshold", np.nan)
            )
            
            output["phase_sep_PE_drop_starting_PE_per_particle"] = _safe_float(
                PE_attrs.get("starting_PE_per_particle", np.nan)
            )
            
            output["phase_sep_PE_drop_last_PE_per_particle_mean"] = _safe_float(
                PE_attrs.get("last_PE_per_particle_mean", np.nan)
            )
            
            output["phase_sep_PE_drop_last_PE_per_particle_std"] = _safe_float(
                PE_attrs.get("last_PE_per_particle_std", np.nan)
            )
            
            output["phase_sep_PE_drop_starting_total_PE"] = _safe_float(
                PE_attrs.get("starting_total_PE", np.nan)
            )
            
            output["phase_sep_PE_drop_last_total_PE_mean"] = _safe_float(
                PE_attrs.get("last_total_PE_mean", np.nan)
            )
            
            output["phase_sep_PE_drop_last_total_PE_std"] = _safe_float(
                PE_attrs.get("last_total_PE_std", np.nan)
            )
            
            output["phase_sep_PE_drop_total_PE_drop"] = _safe_float(
                PE_attrs.get("total_PE_drop", np.nan)
            )
            
            output["phase_sep_PE_drop_updated_from_saved_log"] = _safe_bool(
                PE_attrs.get("updated_from_saved_log", None)
            )

    return output


# ============================================================
# Delete failed phase-separation runs from a phase report
# ============================================================

def delete_failed_phase_runs_from_report(
    report_df,
    dry_run=True,
    delete_log=True,
    delete_state=True,
    delete_empty_folder=True,
):
    """
    Delete failed runs listed in a phase-separation update report.

    Expected input:
        report_df from ps.update_all_v2_phase_separation_metadata(
            show_only_changed=False,
        )

    By default this only prints what it would delete.
    """

    if report_df is None or len(report_df) == 0:
        print("No rows provided.")
        return pd.DataFrame()

    if "status" not in report_df.columns:
        raise KeyError("report_df must have a 'status' column")

    failed_df = report_df[
        report_df["status"] == "failed"
    ].copy()

    rows = []

    for _, row in failed_df.iterrows():
        log_path = row.get("log_path", "")
        state_path = row.get("state_path", "")

        paths_to_delete = []

        # --------------------------------------------------------
        # Log path
        # --------------------------------------------------------
        if delete_log and log_path not in [None, ""]:
            try:
                if not pd.isna(log_path):
                    paths_to_delete.append(Path(log_path))
            except Exception:
                paths_to_delete.append(Path(log_path))

        # --------------------------------------------------------
        # State path
        # --------------------------------------------------------
        if delete_state:
            use_state_path = None

            try:
                if state_path not in [None, ""] and not pd.isna(state_path):
                    use_state_path = Path(state_path)
            except Exception:
                if state_path not in [None, ""]:
                    use_state_path = Path(state_path)

            # If the failed report did not have a state_path,
            # infer it from the log path:
            # randomization_log.hdf5 -> randomization.gsd
            if use_state_path is None and log_path not in [None, ""]:
                log_path_obj = Path(log_path)

                if log_path_obj.name.endswith("_log.hdf5"):
                    use_state_path = log_path_obj.with_name(
                        log_path_obj.name.replace("_log.hdf5", ".gsd")
                    )

            if use_state_path is not None:
                paths_to_delete.append(use_state_path)

        # --------------------------------------------------------
        # Delete files
        # --------------------------------------------------------
        deleted_files = []
        missing_files = []

        for path in paths_to_delete:
            path = Path(path)

            if path.exists():
                if dry_run:
                    print("[dry run] would delete:", path)
                else:
                    path.unlink()
                    print("deleted:", path)

                deleted_files.append(str(path))

            else:
                missing_files.append(str(path))

        # --------------------------------------------------------
        # Remove empty parent folder
        # --------------------------------------------------------
        removed_folder = ""

        if delete_empty_folder and log_path not in [None, ""]:
            folder = Path(log_path).parent

            if folder.exists():
                try:
                    if dry_run:
                        remaining = list(folder.iterdir())
                        print("[dry run] parent folder exists:", folder)
                        print("[dry run] current files in folder:", len(remaining))
                    else:
                        folder.rmdir()
                        removed_folder = str(folder)
                        print("removed empty folder:", folder)

                except OSError:
                    # Folder was not empty. That is fine.
                    pass

        rows.append({
            "log_path": str(log_path),
            "state_path": str(state_path),
            "dry_run": bool(dry_run),
            "deleted_files": deleted_files,
            "missing_files": missing_files,
            "removed_folder": removed_folder,
            "error": row.get("error", ""),
        })

    return pd.DataFrame(rows)


# ============================================================
# General V3-style classification wrappers
# ============================================================

def classify_final_state(
    state_path,
    log_path,
    nbins=DEFAULT_PHASE_SEP_NBINS,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
    dry_run=False,
):
    """
    Run the current voxel phase-separation code on any final state.

    Intended inputs:
    - thermalized/randomized final states
    - cavitation evolved final states
    - excitation evolved final states

    This is intentionally not meant for artificial starting states such as
    cavitation_initial.gsd or excitation_initial.gsd.
    """

    return write_voxel_phase_separation_metadata(
        log_path=Path(log_path),
        state_path=Path(state_path),
        nbins=nbins,
        density_threshold=density_threshold,
        voxel_fraction_threshold=voxel_fraction_threshold,
        updated_from_saved_gsd=True,
        dry_run=dry_run,
    )


def classify_PE_drop(
    log_path,
    dry_run=False,
    **kwargs,
):
    """
    Run the current PE-drop classifier on any evolved run log.
    """

    result = compute_PE_drop_phase_separation_from_log(
        log_path=Path(log_path),
        **kwargs,
    )

    if dry_run:
        return result

    write_PE_drop_phase_separation_metadata(
        log_path=Path(log_path),
        **kwargs,
    )

    return result
