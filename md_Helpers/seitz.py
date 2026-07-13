import numpy as np
from pathlib import Path


THERMO_BASE = "hoomd-data/md/compute/ThermodynamicQuantities"
DEFAULT_EOS_TABLE_NAME = "thermalization_master_ncells_30.csv"

THERMO_COLUMN_ALIASES = {
    "Pressure_mean_last100": "pressure_mean_last100",
    "Pressure_std_last100": "pressure_std_last100",
    "PE_mean_last100_per_particle": "PE_per_particle_mean_last100",
    "PE_std_last100_per_particle": "PE_per_particle_std_last100",
}


def default_eos_table_path():
    """Return the default homogeneous-liquid EOS CSV path."""

    from .paths import MASTER_CSVS_V3_ROOT

    return MASTER_CSVS_V3_ROOT / DEFAULT_EOS_TABLE_NAME


def _resolve_eos_table(eos_table):
    return default_eos_table_path() if eos_table is None else eos_table


def _eos_table_source(eos_table):
    resolved = _resolve_eos_table(eos_table)
    if isinstance(resolved, (str, bytes)) or hasattr(resolved, "__fspath__"):
        return str(resolved)
    return "provided_table"


def _require_existing_hdf5(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}. "
            "This usually means the cavitation evolution did not complete. "
            "Check the result status before extracting Seitz terms."
        )
    return path


def seitz_threshold(nc, uc, u0, p0, rho_c, rho_0):
    """
    Compute the intensive-form Seitz threshold.

        Q = Nc * [(uc - u0) + P0 * (1/rho_c - 1/rho_0)]

    Pass ``uc`` and ``u0`` as PE per particle, ``P0`` as pressure, and
    ``rho_c``/``rho_0`` as number densities.
    """

    nc = np.asarray(nc, dtype=np.float64)
    uc = np.asarray(uc, dtype=np.float64)
    u0 = np.asarray(u0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    rho_c = np.asarray(rho_c, dtype=np.float64)
    rho_0 = np.asarray(rho_0, dtype=np.float64)

    if np.any(nc <= 0):
        raise ValueError("nc must be positive")
    if np.any(rho_c <= 0):
        raise ValueError("rho_c must be positive")
    if np.any(rho_0 <= 0):
        raise ValueError("rho_0 must be positive")

    return nc * ((uc - u0) + p0 * (1.0 / rho_c - 1.0 / rho_0))


def _clean_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def prepare_liquid_eos_table(
    eos_table=None,
    completed_only=True,
    non_phase_separated_only=True,
):
    """
    Clean the homogeneous-liquid EOS table used for reference interpolation.
    """

    import pandas as pd

    eos_table = _resolve_eos_table(eos_table)
    if isinstance(eos_table, (str, bytes)) or hasattr(eos_table, "__fspath__"):
        df = pd.read_csv(eos_table)
    else:
        df = eos_table.copy()

    for old_col, new_col in THERMO_COLUMN_ALIASES.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    if completed_only and "status" in df.columns:
        df = df[df["status"] == "completed"].copy()

    if non_phase_separated_only and "phase_separated" in df.columns:
        df = df[~df["phase_separated"].apply(_clean_bool)].copy()

    if "actual_rho" not in df.columns:
        df["actual_rho"] = np.nan

    missing_rho = df["actual_rho"].isna()
    if missing_rho.any():
        if {"N", "volume"}.issubset(df.columns):
            df.loc[missing_rho, "actual_rho"] = (
                df.loc[missing_rho, "N"] / df.loc[missing_rho, "volume"]
            )
        elif {"N", "BoxLength"}.issubset(df.columns):
            df.loc[missing_rho, "actual_rho"] = (
                df.loc[missing_rho, "N"]
                / df.loc[missing_rho, "BoxLength"] ** 3
            )
        elif "target_rho" in df.columns:
            df.loc[missing_rho, "actual_rho"] = df.loc[
                missing_rho,
                "target_rho",
            ]

    required = [
        "kT",
        "actual_rho",
        "PE_per_particle_mean_last100",
        "pressure_mean_last100",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError("EOS table is missing columns: " + ", ".join(missing))

    for col in required:
        df[col] = np.asarray(df[col], dtype=np.float64)

    return df.dropna(subset=required).sort_values(
        ["kT", "actual_rho"]
    ).copy()


def _same_temperature_eos(eos_table, kT, target_rho, kT_atol=1e-8):
    if kT is None:
        raise ValueError("kT is required for EOS interpolation")
    if target_rho is None:
        raise ValueError("target_rho is required for EOS interpolation")

    df = prepare_liquid_eos_table(eos_table)
    kT = float(kT)
    target_rho = float(target_rho)

    same_temperature = df[np.isclose(df["kT"], kT, atol=float(kT_atol))]
    if same_temperature.empty:
        available = np.array(sorted(df["kT"].dropna().unique()))
        raise ValueError(
            f"No EOS rows found for kT={kT:g}. "
            f"Available kT values include: {available[:10]}"
        )

    rho_min = float(same_temperature["actual_rho"].min())
    rho_max = float(same_temperature["actual_rho"].max())
    if not rho_min <= target_rho <= rho_max:
        raise ValueError(
            f"target_rho={target_rho:g} is outside the non-phase-separated "
            f"EOS range for kT={kT:g}: {rho_min:g} to {rho_max:g}"
        )

    return same_temperature


def _estimate_eos_quantity(
    eos_table,
    kT,
    target_rho,
    quantity_column,
    method="linear",
    kT_atol=1e-8,
):
    """
    Estimate one intensive homogeneous-liquid EOS quantity.

    Keep interpolation/fitting choices behind this helper so downstream Seitz
    code keeps receiving the same scalar return value.
    """

    same_temperature = _same_temperature_eos(
        eos_table=eos_table,
        kT=kT,
        target_rho=target_rho,
        kT_atol=kT_atol,
    )

    if quantity_column not in same_temperature.columns:
        raise KeyError(f"EOS table is missing column: {quantity_column}")

    method = str(method).lower()
    if method != "linear":
        raise ValueError(f"Unsupported EOS interpolation method: {method}")

    same_temperature = (
        same_temperature.groupby("actual_rho", as_index=False)[
            quantity_column
        ]
        .mean()
        .sort_values("actual_rho")
    )

    values = np.asarray(same_temperature[quantity_column], dtype=np.float64)
    rho = np.asarray(same_temperature["actual_rho"], dtype=np.float64)
    order = np.argsort(rho)
    rho_sorted = rho[order]

    return float(np.interp(float(target_rho), rho_sorted, values[order]))


def estimate_u0_from_eos(
    eos_table=None,
    kT=None,
    target_rho=None,
    method="linear",
    kT_atol=1e-8,
):
    """Return homogeneous reference PE per particle, ``u0``."""

    return _estimate_eos_quantity(
        eos_table=eos_table,
        kT=kT,
        target_rho=target_rho,
        quantity_column="PE_per_particle_mean_last100",
        method=method,
        kT_atol=kT_atol,
    )


def estimate_p0_from_eos(
    eos_table=None,
    kT=None,
    target_rho=None,
    method="linear",
    kT_atol=1e-8,
):
    """Return homogeneous reference pressure, ``P0``."""

    return _estimate_eos_quantity(
        eos_table=eos_table,
        kT=kT,
        target_rho=target_rho,
        quantity_column="pressure_mean_last100",
        method=method,
        kT_atol=kT_atol,
    )


def _last_window_mean(values, n_last):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot average an empty thermo series")

    if n_last is None:
        return float(values[-1])

    window = values[-min(int(n_last), values.size):]
    return float(np.mean(window))


def _read_thermo_mean(log_path, quantity, n_last=100):
    import h5py

    dataset_path = f"{THERMO_BASE}/{quantity}"
    with h5py.File(log_path, mode="r") as hdf:
        if dataset_path not in hdf:
            raise KeyError(f"{log_path} is missing dataset {dataset_path}")
        return _last_window_mean(hdf[dataset_path][()], n_last)


def _first_available(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value, key
    return None, None


def _infer_full_volume(state_attrs, creation_attrs):
    value, key = _first_available(state_attrs, ["volume"])
    if value is not None:
        return float(value), f"metadata/state:{key}"

    value, key = _first_available(creation_attrs, ["volume"])
    if value is not None:
        return float(value), f"metadata/creation:{key}"

    value, key = _first_available(state_attrs, ["BoxLength"])
    if value is not None:
        return float(value) ** 3, f"metadata/state:{key}**3"

    value, key = _first_available(creation_attrs, ["BoxLength"])
    if value is not None:
        return float(value) ** 3, f"metadata/creation:{key}**3"

    raise ValueError("Could not infer full simulation volume")


def _infer_temperature(state_attrs, explicit_kT=None):
    if explicit_kT is not None:
        return float(explicit_kT), "explicit"

    value = state_attrs.get("kT")
    if value is None:
        raise ValueError("Could not infer kT from metadata/state")
    return float(value), "metadata/state:kT"


def _infer_trajectory_path(paths_attrs, explicit_path=None):
    if explicit_path is not None:
        return str(explicit_path), "explicit"

    value, key = _first_available(
        paths_attrs,
        ["trajectory_path", "final_state_path", "state_path"],
    )
    if value is None:
        raise ValueError(
            "Could not infer trajectory_path. Pass trajectory_path explicitly "
            "or store it in metadata/paths."
        )
    return str(value), f"metadata/paths:{key}"


def _infer_bubble_particle_count(creation_attrs):
    value = creation_attrs.get("N_after")
    if value is None:
        raise ValueError(
            "Could not infer seeded particle count Nc. "
            "Expected metadata/creation['N_after']."
        )
    return float(value), "metadata/creation:N_after"


def extract_bubble_state_terms(
    metadata_path,
    trajectory_path=None,
    log_path=None,
    seeded_potential_energy=None,
    n_last=100,
    kT=None,
    nbins=12,
    nframes=5,
    nskip=5,
    tail_fraction=0.5,
    interface_void_fraction=0.5,
    interface_points=40,
    max_iterations=500,
    plot=True,
    show_residuals=True,
    animate=False,
    eos_table=None,
    estimate_reference=True,
    eos_method="linear",
    eos_kT_atol=1e-8,
):
    """
    Extract bubble-state terms for

        Q = Nc * [(uc - u0) + P0 * (1/rho_c - 1/rho_0)].

    This calls the existing notebook-style ``check`` helper
    ``visualization.fit_and_animate_final_bubble`` to fit and plot the voxel
    histogram.  The liquid Gaussian center from that fit is returned as
    ``rho_0``.  By default, ``u0`` and ``P0`` are also interpolated from the
    default EOS table at ``(kT, rho_0)``.
    """

    if isinstance(metadata_path, dict):
        return extract_cavitation_result_terms(
            metadata_path,
            trajectory_path=trajectory_path,
            log_path=log_path,
            seeded_potential_energy=seeded_potential_energy,
            n_last=n_last,
            kT=kT,
            nbins=nbins,
            nframes=nframes,
            nskip=nskip,
            tail_fraction=tail_fraction,
            interface_void_fraction=interface_void_fraction,
            interface_points=interface_points,
            max_iterations=max_iterations,
            plot=plot,
            show_residuals=show_residuals,
            animate=animate,
            eos_table=eos_table,
            estimate_reference=estimate_reference,
            eos_method=eos_method,
            eos_kT_atol=eos_kT_atol,
        )

    metadata_path = _require_existing_hdf5(
        metadata_path,
        "bubble/evolution metadata_path",
    )

    from . import metadata
    from .visualization import fit_and_animate_final_bubble

    state_attrs = metadata.read_attrs(metadata_path, "metadata/state")
    creation_attrs = metadata.read_attrs(metadata_path, "metadata/creation")
    paths_attrs = metadata.read_attrs(metadata_path, "metadata/paths")

    nc, nc_source = _infer_bubble_particle_count(creation_attrs)
    volume, volume_source = _infer_full_volume(state_attrs, creation_attrs)
    kT_value, kT_source = _infer_temperature(state_attrs, kT)

    if seeded_potential_energy is None:
        if log_path is None:
            log_path = paths_attrs.get("log_path") or metadata_path
        log_path = _require_existing_hdf5(log_path, "bubble/evolution log_path")
        seeded_potential_energy = _read_thermo_mean(
            log_path,
            "potential_energy",
            n_last=n_last,
        )
        uc_source = f"{log_path}:potential_energy"
    else:
        uc_source = "explicit"

    trajectory_path, trajectory_source = _infer_trajectory_path(
        paths_attrs,
        explicit_path=trajectory_path,
    )

    check = fit_and_animate_final_bubble(
        trajectory_path,
        nbins=nbins,
        nframes=nframes,
        skip=nskip,
        tail_fraction=tail_fraction,
        interface_void_fraction=interface_void_fraction,
        interface_points=interface_points,
        max_iterations=max_iterations,
        show_histogram=plot,
        show_residuals=show_residuals,
    )
    fit = check["fit"]

    rho_c = nc / volume
    uc = seeded_potential_energy / nc
    rho_0 = float(fit["liquid_density"])

    result = {
        "Nc": float(nc),
        "nc_source": nc_source,
        "Uc": float(seeded_potential_energy),
        "uc": float(uc),
        "uc_source": uc_source,
        "V": float(volume),
        "volume_source": volume_source,
        "kT": float(kT_value),
        "kT_source": kT_source,
        "rho_c": float(rho_c),
        "rho_0": rho_0,
        "rho_0_source": "check.fit.liquid_density",
        "check": check,
        "voxel_fit": fit,
        "trajectory_path": trajectory_path,
        "trajectory_path_source": trajectory_source,
    }

    if estimate_reference:
        u0 = estimate_u0_from_eos(
            eos_table=eos_table,
            kT=kT_value,
            target_rho=rho_0,
            method=eos_method,
            kT_atol=eos_kT_atol,
        )
        p0 = estimate_p0_from_eos(
            eos_table=eos_table,
            kT=kT_value,
            target_rho=rho_0,
            method=eos_method,
            kT_atol=eos_kT_atol,
        )
        q_seitz = seitz_threshold(
            nc=nc,
            uc=uc,
            u0=u0,
            p0=p0,
            rho_c=rho_c,
            rho_0=rho_0,
        )
        result.update({
            "u0": float(u0),
            "u0_source": "EOS:PE_per_particle_mean_last100",
            "P0": float(p0),
            "p0": float(p0),
            "P0_source": "EOS:pressure_mean_last100",
            "q_seitz": float(q_seitz),
            "Q": float(q_seitz),
            "eos_table": _eos_table_source(eos_table),
            "eos_method": eos_method,
        })

    if not animate:
        result.pop("check", None)

    return result


def extract_cavitation_result_terms(cavitation_result, **kwargs):
    """
    Extract Seitz terms from a ``get_or_create_cavitation`` result.

    If cavitation was skipped or did not complete, return a status row with
    ``Q``/``q_seitz`` set to ``nan`` instead of trying to open a missing log.
    """

    status = cavitation_result.get("status", "unknown")
    completed_statuses = {"created_evolution", "loaded_evolution"}

    if status not in completed_statuses:
        result = {
            "status": "seitz_not_computed",
            "cavitation_status": status,
            "Q": np.nan,
            "q_seitz": np.nan,
            "reason": "cavitation evolution did not complete",
        }

        initial = cavitation_result.get("initial_result", {})
        source_phase = initial.get("source_phase_separation", {})
        if source_phase:
            result["source_phase_separated"] = source_phase.get(
                "phase_separated"
            )
            result["source_low_density_fraction"] = source_phase.get(
                "low_density_fraction"
            )

        paths = cavitation_result.get("paths", {})
        if paths:
            result["expected_log_path"] = str(paths.get("log_path", ""))
            result["expected_trajectory_path"] = str(
                paths.get("trajectory_path", "")
            )

        return result

    terms = extract_bubble_state_terms(
        metadata_path=cavitation_result["paths"]["log_path"],
        **kwargs,
    )
    terms["status"] = "seitz_computed"
    terms["cavitation_status"] = status
    return terms
