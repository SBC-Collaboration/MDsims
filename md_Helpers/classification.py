from pathlib import Path

import gsd.hoomd
import h5py
import numpy as np

from .spatial import compute_voxel_densities, nbins_for_ncells


DEFAULT_PHASE_SEP_NBINS = 10
DEFAULT_PHASE_SEP_DENSITY_THRESHOLD = 0.2
DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD = 0.01

DEFAULT_PHASE_SEP_PE_DROP_THRESHOLD = -0.15
DEFAULT_PHASE_SEP_PE_DROP_Z_LIMIT = 5.0
DEFAULT_PHASE_SEP_PE_DROP_N_LAST = 100
DEFAULT_PHASE_SEP_PE_DROP_DECISION_RULE = "either"

PHASE_SEPARATION_METADATA_PATH = "metadata/classification/phase_separation"


def _clean_value(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _read_group_attrs(hdf, group_path):
    if group_path not in hdf:
        return {}
    return {
        key: _clean_value(value)
        for key, value in hdf[group_path].attrs.items()
    }


def read_phase_method_attrs(log_path, method_name):
    """Read one classifier's attributes from the canonical V3 location."""

    method_path = f"{PHASE_SEPARATION_METADATA_PATH}/{method_name}"
    with h5py.File(Path(log_path), mode="r") as hdf:
        return _read_group_attrs(hdf, method_path), method_path


def compute_voxel_fraction_phase_separation(
    obj,
    nbins=None,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
    n_fcc_cells=None,
):
    """Classify a frame from its fraction of low-density voxels.

    When ``nbins`` is omitted, use the Seitz resolution rule when
    ``n_fcc_cells`` is available, otherwise retain the legacy default.
    """

    if nbins is not None:
        nbins = int(nbins)
        nbins_source = "explicit"
    elif n_fcc_cells is not None:
        nbins = nbins_for_ncells(n_fcc_cells)
        nbins_source = "n_fcc_cells_rule"
    else:
        nbins = DEFAULT_PHASE_SEP_NBINS
        nbins_source = "default_no_n_fcc_cells"

    voxel_densities, _, voxel_volume = compute_voxel_densities(obj, nbins)
    low_density_fraction = float(
        np.mean(voxel_densities < density_threshold)
    )

    return {
        "phase_separated": bool(
            low_density_fraction > voxel_fraction_threshold
        ),
        "method": "voxel_low_density_fraction",
        "nbins": int(nbins),
        "nbins_source": nbins_source,
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


def check_phase_separated(obj, **kwargs):
    return bool(
        compute_voxel_fraction_phase_separation(obj, **kwargs)[
            "phase_separated"
        ]
    )


def _read_log_metadata(log_path):
    attrs = {}
    with h5py.File(Path(log_path), mode="r") as hdf:
        for group_path in [
            "metadata/state",
            "metadata/run",
            "metadata/lj",
            "metadata/paths",
            "metadata/source",
        ]:
            attrs.update(_read_group_attrs(hdf, group_path))
    return attrs


def _read_PE_arrays(log_path):
    metadata = _read_log_metadata(log_path)
    if "N" not in metadata:
        raise KeyError(f"metadata/state.attrs['N'] is missing: {log_path}")

    with h5py.File(Path(log_path), mode="r") as hdf:
        timestep = np.asarray(
            hdf["hoomd-data/Simulation/timestep"],
            dtype=int,
        )
        potential_energy = np.asarray(
            hdf[
                "hoomd-data/md/compute/ThermodynamicQuantities/"
                "potential_energy"
            ],
            dtype=float,
        )

    if len(potential_energy) == 0:
        raise ValueError(f"No potential-energy values found: {log_path}")
    if len(timestep) != len(potential_energy):
        raise ValueError("Timestep and potential-energy lengths differ.")

    return timestep, potential_energy, potential_energy / int(metadata["N"])


def _last_window_stats(values, n_last):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot summarize an empty array.")

    window = values[-min(int(n_last), len(values)):]
    std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
    return float(np.mean(window)), std


def _combine_PE_decisions(raw_pass, z_pass, decision_rule):
    rules = {
        "raw": raw_pass,
        "z_score": z_pass,
        "either": raw_pass or z_pass,
        "both": raw_pass and z_pass,
    }
    try:
        return bool(rules[str(decision_rule).lower()])
    except KeyError as error:
        raise ValueError(
            "decision_rule must be 'raw', 'z_score', 'either', or 'both'"
        ) from error


def compute_PE_drop_phase_separation_from_log(
    log_path,
    PE_drop_threshold=DEFAULT_PHASE_SEP_PE_DROP_THRESHOLD,
    z_score_threshold=DEFAULT_PHASE_SEP_PE_DROP_Z_LIMIT,
    n_last=DEFAULT_PHASE_SEP_PE_DROP_N_LAST,
    decision_rule=DEFAULT_PHASE_SEP_PE_DROP_DECISION_RULE,
):
    """Classify a run from its change in potential energy per particle."""

    _, total_PE, PE_per_particle = _read_PE_arrays(log_path)
    last_PE_mean, last_PE_std = _last_window_stats(PE_per_particle, n_last)
    last_total_mean, last_total_std = _last_window_stats(total_PE, n_last)

    starting_PE = float(PE_per_particle[0])
    starting_total = float(total_PE[0])
    PE_drop = float(last_PE_mean - starting_PE)

    if last_PE_std > 0:
        z_score = float(PE_drop / last_PE_std)
    elif PE_drop < 0:
        z_score = float(-np.inf)
    elif PE_drop > 0:
        z_score = float(np.inf)
    else:
        z_score = 0.0

    raw_pass = bool(PE_drop <= PE_drop_threshold)
    z_pass = bool(z_score <= -float(z_score_threshold))

    return {
        "phase_separated": _combine_PE_decisions(
            raw_pass,
            z_pass,
            decision_rule,
        ),
        "method": "PE_drop",
        "quantity": "PE_per_particle",
        "definition": (
            "PE_drop = last_PE_per_particle_mean - "
            "starting_PE_per_particle"
        ),
        "decision_rule": str(decision_rule),
        "PE_drop_threshold": float(PE_drop_threshold),
        "z_score_threshold": float(z_score_threshold),
        "passes_PE_drop_threshold": raw_pass,
        "passes_z_score_threshold": z_pass,
        "starting_PE_per_particle": starting_PE,
        "starting_total_PE": starting_total,
        "last_PE_per_particle_mean": last_PE_mean,
        "last_PE_per_particle_std": last_PE_std,
        "last_total_PE_mean": last_total_mean,
        "last_total_PE_std": last_total_std,
        "PE_drop": PE_drop,
        "PE_drop_z_score": z_score,
        "total_PE_drop": float(last_total_mean - starting_total),
    }


def write_phase_method_metadata(
    log_path,
    method_name,
    attrs,
    set_main_phase_separated=None,
):
    """Replace one classifier's canonical V3 metadata attributes."""

    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    with h5py.File(log_path, mode="a") as hdf:
        phase_group = hdf.require_group(PHASE_SEPARATION_METADATA_PATH)
        if set_main_phase_separated is not None:
            phase_group.attrs["phase_separated"] = bool(
                set_main_phase_separated
            )

        method_group = phase_group.require_group(method_name)
        for key in list(method_group.attrs):
            del method_group.attrs[key]
        for key, value in attrs.items():
            if value is not None:
                method_group.attrs[key] = value


def _state_path_for_log(log_path):
    log_path = Path(log_path)
    with h5py.File(log_path, mode="r") as hdf:
        paths = _read_group_attrs(hdf, "metadata/paths")

    state_path = paths.get("state_path") or paths.get("final_state_path")
    if state_path and Path(state_path).exists():
        return Path(state_path)

    candidate = log_path.with_name(
        log_path.name.replace("_log.hdf5", ".gsd")
    )
    return candidate if candidate.exists() else None


def write_voxel_phase_separation_metadata(
    log_path,
    state_path=None,
    nbins=None,
    density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
    updated_from_saved_gsd=True,
    dry_run=False,
):
    """Compute and store the voxel classifier for one completed state.

    By default, infer ``nbins`` from ``metadata/state:n_fcc_cells`` using the
    same rule as the Seitz voxel-density fit. Pass ``nbins`` to override it.
    """

    log_path = Path(log_path)
    state_path = Path(state_path) if state_path else _state_path_for_log(log_path)
    if state_path is None or not state_path.exists():
        raise FileNotFoundError(f"Could not find state for log: {log_path}")

    state_attrs = _read_group_attrs(log_path, "metadata/state")
    n_fcc_cells = state_attrs.get("n_fcc_cells")

    with gsd.hoomd.open(name=str(state_path), mode="r") as trajectory:
        result = compute_voxel_fraction_phase_separation(
            trajectory[-1],
            nbins=nbins,
            n_fcc_cells=n_fcc_cells,
            density_threshold=density_threshold,
            voxel_fraction_threshold=voxel_fraction_threshold,
        )

    attrs = dict(result)
    attrs["updated_from_saved_gsd"] = bool(updated_from_saved_gsd)
    if not dry_run:
        write_phase_method_metadata(
            log_path,
            "voxel",
            attrs,
            set_main_phase_separated=result["phase_separated"],
        )

    return {
        "status": "dry_run" if dry_run else "updated",
        "log_path": str(log_path),
        "state_path": str(state_path),
        **result,
    }


def write_PE_drop_phase_separation_metadata(log_path, dry_run=False, **kwargs):
    """Compute and store the PE-drop classifier for one completed run."""

    result = compute_PE_drop_phase_separation_from_log(log_path, **kwargs)
    if not dry_run:
        write_phase_method_metadata(log_path, "PE_drop", result)
    return {
        "status": "dry_run" if dry_run else "updated",
        "log_path": str(log_path),
        **result,
    }


def classify_final_state(state_path, log_path, **kwargs):
    """Run the current frame-based classifier on an evolved final state."""

    return write_voxel_phase_separation_metadata(
        log_path=log_path,
        state_path=state_path,
        updated_from_saved_gsd=True,
        **kwargs,
    )


def classify_PE_drop(log_path, dry_run=False, **kwargs):
    """Run the current PE-drop classifier on an evolved run log."""

    result = compute_PE_drop_phase_separation_from_log(log_path, **kwargs)
    if not dry_run:
        write_phase_method_metadata(log_path, "PE_drop", result)
    return result
