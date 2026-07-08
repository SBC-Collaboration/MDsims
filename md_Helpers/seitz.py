import numpy as np


THERMO_COLUMN_ALIASES = {
    "Pressure_mean_last100": "pressure_mean_last100",
    "Pressure_std_last100": "pressure_std_last100",
    "PE_mean_last100_per_particle": "PE_per_particle_mean_last100",
    "PE_std_last100_per_particle": "PE_per_particle_std_last100",
    "gsd_path": "state_path",
}


def seitz_threshold(
    volume,
    n_cavity,
    u_cavity,
    rho0,
    u0,
    p0,
):
    """
    Compute the Seitz threshold energy for a cavity/bubble volume.

    Parameters
    ----------
    volume:
        Bubble/cavity volume.
    n_cavity:
        Number of particles actually in the cavity volume.
    u_cavity:
        Total internal/potential energy of particles in the cavity volume.
    rho0:
        Reference liquid number density at the same temperature.
    u0:
        Reference liquid energy per particle at ``rho0`` and the same
        temperature.
    p0:
        Reference liquid pressure at ``rho0`` and the same temperature.

    Notes
    -----
    This implements the whiteboard form

        Qs = (Uc - U0) + ((N0 - N) / N0) * (U0 + P0 * V)

    with ``N0 = rho0 * V`` and ``U0 = u0 * N0``.
    """

    volume = np.asarray(volume, dtype=np.float64)
    n_cavity = np.asarray(n_cavity, dtype=np.float64)
    u_cavity = np.asarray(u_cavity, dtype=np.float64)
    rho0 = np.asarray(rho0, dtype=np.float64)
    u0 = np.asarray(u0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)

    n0 = rho0 * volume

    if np.any(volume <= 0):
        raise ValueError("volume must be positive")

    if np.any(rho0 <= 0):
        raise ValueError("rho0 must be positive")

    u0_total = u0 * n0
    removed_fraction = (n0 - n_cavity) / n0

    return (u_cavity - u0_total) + removed_fraction * (
        u0_total + p0 * volume
    )


def sphere_volume(radius):
    """Return the volume of a sphere with the given radius."""

    radius = np.asarray(radius, dtype=np.float64)

    if np.any(radius <= 0):
        raise ValueError("radius must be positive")

    return (4.0 / 3.0) * np.pi * radius ** 3


def interpolate_liquid_reference(
    target_rho,
    rho,
    u_per_particle,
    pressure,
):
    """
    Interpolate liquid-only reference values at a target density.

    Use this for the note "from liquid-only at same T; need to interpolate
    in rho". All arrays should come from simulations at one temperature.
    """

    target_rho = np.asarray(target_rho, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    u_per_particle = np.asarray(u_per_particle, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)

    if rho.ndim != 1:
        raise ValueError("rho must be one-dimensional")

    if rho.size < 2:
        raise ValueError("at least two rho points are required")

    if u_per_particle.shape != rho.shape:
        raise ValueError("u_per_particle must have the same shape as rho")

    if pressure.shape != rho.shape:
        raise ValueError("pressure must have the same shape as rho")

    order = np.argsort(rho)
    rho_sorted = rho[order]

    if np.any(np.diff(rho_sorted) == 0):
        raise ValueError("rho values must be unique")

    return {
        "rho0": target_rho,
        "u0": np.interp(target_rho, rho_sorted, u_per_particle[order]),
        "p0": np.interp(target_rho, rho_sorted, pressure[order]),
    }


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
    eos_table,
    completed_only=True,
    non_phase_separated_only=True,
):
    """
    Clean the liquid-only EOS table used for the pressure/PE density plots.

    ``eos_table`` may be a pandas DataFrame or a path to the CSV that made
    the pressure-vs-density and PE/N-vs-density plots.
    """

    import pandas as pd

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=required).sort_values(
        ["kT", "actual_rho"]
    ).copy()


def liquid_reference_from_eos(
    eos_table,
    kT,
    target_rho,
    kT_atol=1e-8,
):
    """
    Get ``rho0``, ``u0``, and ``p0`` from the liquid-only EOS table.

    This is the programmatic version of reading your two plots:
    pressure-vs-density supplies ``p0`` and PE/N-vs-density supplies ``u0``.
    """

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

    ref = interpolate_liquid_reference(
        target_rho=target_rho,
        rho=same_temperature["actual_rho"],
        u_per_particle=same_temperature["PE_per_particle_mean_last100"],
        pressure=same_temperature["pressure_mean_last100"],
    )

    ref["kT"] = kT
    ref["rho_min"] = rho_min
    ref["rho_max"] = rho_max
    return ref


def _lj_pair_energy(
    r,
    epsilon=1.0,
    sigma=1.0,
    r_cut=2.5,
    mode="xplor",
    r_on=2.0,
):
    r = np.asarray(r, dtype=np.float64)
    sr6 = (float(sigma) / r) ** 6
    energy = 4.0 * float(epsilon) * (sr6 ** 2 - sr6)

    mode = str(mode or "none").lower()

    if mode == "shift":
        src6 = (float(sigma) / float(r_cut)) ** 6
        energy = energy - 4.0 * float(epsilon) * (src6 ** 2 - src6)
    elif mode == "xplor":
        r_on = float(r_on)
        r_cut = float(r_cut)
        switch = np.ones_like(r)
        switching = r > r_on

        if np.any(switching):
            r2 = r[switching] ** 2
            r_on2 = r_on ** 2
            r_cut2 = r_cut ** 2
            switch[switching] = (
                (r_cut2 - r2) ** 2
                * (r_cut2 + 2.0 * r2 - 3.0 * r_on2)
                / (r_cut2 - r_on2) ** 3
            )

        energy = energy * switch

    return energy


def cavity_terms_from_frame(
    frame,
    radius,
    center=(0.0, 0.0, 0.0),
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    lj_mode="xplor",
    r_on_LJ=2.0,
    chunk_size=256,
):
    """
    Measure ``N_cavity`` and ``U_cavity`` from a frame.

    ``U_cavity`` is the sum of per-particle LJ energies for particles inside
    the cavity sphere. Each pair contributes half its pair energy to each
    particle, matching the usual per-particle potential-energy convention.
    """

    from .spatial import periodic_distances

    box = np.asarray(frame.configuration.box, dtype=np.float64)
    box_lengths = box[:3]
    positions = np.asarray(frame.particles.position, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)

    cavity_distances = periodic_distances(positions, center, box_lengths)
    inside_mask = cavity_distances <= radius
    inside_positions = positions[inside_mask]
    n_cavity = int(inside_positions.shape[0])

    if n_cavity == 0:
        return {
            "n_cavity": 0,
            "u_cavity": 0.0,
            "cavity_radius": radius,
            "cavity_volume": float(sphere_volume(radius)),
            "cavity_center_x": float(center[0]),
            "cavity_center_y": float(center[1]),
            "cavity_center_z": float(center[2]),
        }

    r_cut_LJ = float(r_cut_LJ)
    total = 0.0

    for start in range(0, n_cavity, int(chunk_size)):
        stop = min(start + int(chunk_size), n_cavity)
        delta = inside_positions[start:stop, None, :] - positions[None, :, :]
        delta -= box_lengths * np.round(delta / box_lengths)
        r2 = np.sum(delta * delta, axis=-1)
        mask = (r2 > 0.0) & (r2 < r_cut_LJ ** 2)

        if np.any(mask):
            total += 0.5 * float(np.sum(_lj_pair_energy(
                np.sqrt(r2[mask]),
                epsilon=epsilon_LJ,
                sigma=sigma_LJ,
                r_cut=r_cut_LJ,
                mode=lj_mode,
                r_on=r_on_LJ,
            )))

    return {
        "n_cavity": n_cavity,
        "u_cavity": float(total),
        "cavity_radius": radius,
        "cavity_volume": float(sphere_volume(radius)),
        "cavity_center_x": float(center[0]),
        "cavity_center_y": float(center[1]),
        "cavity_center_z": float(center[2]),
    }


def cavity_terms_from_files(
    metadata_path,
    state_path=None,
    frame_index=-1,
    chunk_size=256,
):
    """
    Read cavitation metadata and return ``N_cavity``/``U_cavity``.

    For a freshly created cavitated state, the bubble was made by removing
    particles, so this returns ``N_cavity=0`` and ``U_cavity=0`` directly from
    metadata. For evolved states, pass ``state_path`` or use a log whose
    ``metadata/paths`` contains ``final_state_path``.
    """

    from . import cavitation, metadata

    state_attrs = metadata.read_attrs(metadata_path, "metadata/state")
    creation = metadata.read_attrs(metadata_path, "metadata/creation")
    lj = metadata.read_attrs(metadata_path, "metadata/lj")
    paths = metadata.read_attrs(metadata_path, "metadata/paths")

    radius = creation.get("bubble_radius", creation.get("radius"))
    if radius is None:
        raise KeyError("metadata/creation is missing bubble_radius")

    center = np.array([
        creation.get("bubble_center_x", 0.0),
        creation.get("bubble_center_y", 0.0),
        creation.get("bubble_center_z", 0.0),
    ], dtype=np.float64)

    state_kind = state_attrs.get("state_kind")
    if state_kind == "cavitation_initial":
        return {
            "n_cavity": 0,
            "u_cavity": 0.0,
            "cavity_radius": float(radius),
            "cavity_volume": float(sphere_volume(radius)),
            "cavity_center_x": float(center[0]),
            "cavity_center_y": float(center[1]),
            "cavity_center_z": float(center[2]),
            "from_metadata_only": True,
            "particles_removed": int(creation.get("particles_removed", 0)),
        }

    if state_path is None:
        state_path = (
            paths.get("final_state_path")
            or paths.get("state_path")
            or paths.get("trajectory_path")
        )

    if state_path is None:
        raise ValueError(
            "state_path is required for evolved cavity terms because the "
            "particle positions are not stored in metadata"
        )

    frame = cavitation.load_frame_from_gsd(state_path, frame_index=frame_index)

    return cavity_terms_from_frame(
        frame=frame,
        radius=radius,
        center=center,
        epsilon_LJ=lj.get("epsilon_LJ", 1.0),
        sigma_LJ=lj.get("sigma_LJ", 1.0),
        r_cut_LJ=lj.get("r_cut_LJ", 2.5),
        lj_mode=lj.get("lj_mode", "xplor"),
        r_on_LJ=lj.get("r_on_LJ", 2.0),
        chunk_size=chunk_size,
    )


def seitz_threshold_from_files(
    metadata_path,
    eos_table,
    state_path=None,
    reference_rho=None,
    reference_kT=None,
    frame_index=-1,
):
    """
    Convenience wrapper: cavity terms + liquid EOS reference + Seitz Q.
    """

    from . import metadata

    state_attrs = metadata.read_attrs(metadata_path, "metadata/state")
    creation = metadata.read_attrs(metadata_path, "metadata/creation")

    kT = float(reference_kT if reference_kT is not None else state_attrs["kT"])
    rho = float(
        reference_rho
        if reference_rho is not None
        else creation.get("rho_before", state_attrs.get("source_rho"))
    )

    cavity = cavity_terms_from_files(
        metadata_path=metadata_path,
        state_path=state_path,
        frame_index=frame_index,
    )
    reference = liquid_reference_from_eos(
        eos_table=eos_table,
        kT=kT,
        target_rho=rho,
    )
    q = seitz_threshold(
        volume=cavity["cavity_volume"],
        n_cavity=cavity["n_cavity"],
        u_cavity=cavity["u_cavity"],
        rho0=reference["rho0"],
        u0=reference["u0"],
        p0=reference["p0"],
    )

    return {
        **cavity,
        **reference,
        "q_seitz": float(q),
    }
