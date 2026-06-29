from pathlib import Path

import numpy as np
import pandas as pd

from . import metadata
from .spatial import periodic_distances


def _result_path(result, key, explicit=None):
    if explicit is not None:
        return Path(explicit)
    if isinstance(result, dict):
        for container in [result.get("paths", {}), result.get("run_result", {})]:
            if key in container:
                return Path(container[key])
    raise ValueError(f"Could not infer {key}; pass it explicitly.")


def _creation_info(result, log_path):
    if isinstance(result, dict):
        if "creation_info" in result:
            return dict(result["creation_info"])
        initial = result.get("initial_result", {})
        if isinstance(initial, dict):
            return dict(initial.get("creation_info", {}))
    if log_path and Path(log_path).exists():
        return metadata.read_attrs(log_path, "metadata/creation")
    return {}


def _bubble_center(info, supplied_center=None):
    if supplied_center is not None:
        center = np.asarray(supplied_center, dtype=np.float64)
    elif "bubble_center" in info:
        center = np.asarray(info["bubble_center"], dtype=np.float64)
    else:
        center = np.array([
            info.get("bubble_center_x", 0.0),
            info.get("bubble_center_y", 0.0),
            info.get("bubble_center_z", 0.0),
        ], dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("bubble_center must contain three coordinates")
    return center


def estimate_bubble_from_radial_density(
    distances,
    box_lengths,
    bulk_density,
    n_radial_bins=80,
    density_threshold_fraction=0.5,
    recovery_bins=3,
):
    """Estimate bubble radius from sustained radial-density recovery."""

    max_radius = 0.5 * float(np.min(box_lengths))
    edges = np.linspace(0.0, max_radius, int(n_radial_bins) + 1)
    counts, _ = np.histogram(distances, bins=edges)
    shell_volumes = (
        (4.0 / 3.0)
        * np.pi
        * (edges[1:] ** 3 - edges[:-1] ** 3)
    )
    densities = counts / shell_volumes
    threshold = float(density_threshold_fraction) * float(bulk_density)

    recovery_bins = max(1, int(recovery_bins))
    radius = max_radius
    for index in range(0, len(densities) - recovery_bins + 1):
        if np.all(densities[index:index + recovery_bins] >= threshold):
            radius = float(edges[index])
            break

    void_volume = (4.0 / 3.0) * np.pi * radius ** 3
    return {
        "bubble_radius_estimate": radius,
        "void_volume_estimate": float(void_volume),
        "radial_density_threshold": threshold,
        "min_shell_density": float(np.min(densities)),
        "max_shell_density": float(np.max(densities)),
        "mean_shell_density": float(np.mean(densities)),
    }


def _thermo_dataframe(log_path):
    if log_path is None or not Path(log_path).exists():
        return pd.DataFrame()

    from . import runs

    log = runs.read_hdf5_log(log_path)
    try:
        timestep = np.asarray(
            log["hoomd-data"]["Simulation"]["timestep"],
            dtype=np.int64,
        )
        thermo = log["hoomd-data"]["md"]["compute"][
            "ThermodynamicQuantities"
        ]
    except KeyError:
        return pd.DataFrame()

    data = {"timestep": timestep}
    for quantity in [
        "kinetic_temperature",
        "pressure",
        "potential_energy",
        "kinetic_energy",
    ]:
        if quantity in thermo:
            data[quantity] = np.asarray(thermo[quantity], dtype=float)
    return pd.DataFrame(data)


def measure_cavitation_trajectory(
    evolution=None,
    trajectory_path=None,
    log_path=None,
    bubble_center=None,
    n_radial_bins=80,
    density_threshold_fraction=0.5,
    recovery_bins=3,
    initial_radius=None,
    save_csv_path=None,
):
    """Measure bubble geometry and thermodynamics across a trajectory."""

    import gsd.hoomd

    trajectory_path = _result_path(
        evolution,
        "trajectory_path",
        trajectory_path,
    )
    try:
        log_path = _result_path(evolution, "log_path", log_path)
    except ValueError:
        log_path = None

    creation = _creation_info(evolution, log_path)
    center = _bubble_center(creation, bubble_center)
    if initial_radius is None:
        initial_radius = creation.get("bubble_radius")
    if initial_radius is not None:
        initial_radius = float(initial_radius)

    rows = []
    with gsd.hoomd.open(name=str(trajectory_path), mode="r") as trajectory:
        for frame_index, frame in enumerate(trajectory):
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            box_lengths = np.asarray(frame.configuration.box[:3], dtype=np.float64)
            volume = float(np.prod(box_lengths))
            particle_count = int(frame.particles.N)
            bulk_density = particle_count / volume
            distances = periodic_distances(positions, center, box_lengths)

            row = {
                "frame_index": frame_index,
                "timestep": int(frame.configuration.step),
                "N": particle_count,
                "BoxLength_x": float(box_lengths[0]),
                "BoxLength_y": float(box_lengths[1]),
                "BoxLength_z": float(box_lengths[2]),
                "volume": volume,
                "bulk_density": bulk_density,
                "bubble_center_x": float(center[0]),
                "bubble_center_y": float(center[1]),
                "bubble_center_z": float(center[2]),
            }
            row.update(estimate_bubble_from_radial_density(
                distances,
                box_lengths,
                bulk_density,
                n_radial_bins=n_radial_bins,
                density_threshold_fraction=density_threshold_fraction,
                recovery_bins=recovery_bins,
            ))
            row["void_fraction_estimate"] = (
                row["void_volume_estimate"] / volume
            )

            if initial_radius is not None:
                initial_volume = (4.0 / 3.0) * np.pi * initial_radius ** 3
                inside = int(np.sum(distances <= initial_radius))
                row.update({
                    "initial_bubble_radius": initial_radius,
                    "particles_inside_initial_radius": inside,
                    "density_inside_initial_radius": inside / initial_volume,
                })
            rows.append(row)

    measurements = pd.DataFrame(rows)
    thermo = _thermo_dataframe(log_path)
    if not thermo.empty:
        measurements = measurements.merge(thermo, on="timestep", how="left")
        if "potential_energy" in measurements:
            measurements["PE_per_particle"] = (
                measurements["potential_energy"] / measurements["N"]
            )
        if "kinetic_energy" in measurements:
            measurements["KE_per_particle"] = (
                measurements["kinetic_energy"] / measurements["N"]
            )

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        measurements.to_csv(save_csv_path, index=False)
    return measurements
