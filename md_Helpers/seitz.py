import numpy as np


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
