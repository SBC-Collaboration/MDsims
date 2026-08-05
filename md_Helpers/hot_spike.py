from pathlib import Path

import gsd.hoomd
import numpy as np

from . import cavitation as cavitation_helpers
from . import classification as classification_helpers
from . import metadata as metadata_helpers
from . import runs as run_helpers
from . import simulation as simulation_helpers
from .run_logs import simulation_progress
from .paths import (
    EXCITATION_STATES_V3_ROOT,
    excitation_state_paths,
    legacy_excitation_evolved_paths,
)
from .spatial import periodic_distances


SUPPORTED_METHODS = {
    "velocity_rescale_raw",
    "velocity_rescale_com",
}


def _load_frame_from_gsd(state_path, frame_index=-1):
    return cavitation_helpers.load_frame_from_gsd(
        state_path=state_path,
        frame_index=frame_index,
    )


def _save_frame_to_gsd(frame, state_path, overwrite=False):
    return cavitation_helpers.save_frame_to_gsd(
        frame=frame,
        state_path=state_path,
        overwrite=overwrite,
    )


def _source_seed_from_metadata(source_result, fallback_seed):
    source_log_path = Path(source_result["paths"]["log_path"])
    run_attrs = metadata_helpers.read_attrs(
        source_log_path,
        "metadata/run",
    )
    return int(run_attrs.get("seed", fallback_seed))


def _source_lj_kwargs(source_result):
    source_log_path = Path(source_result["paths"]["log_path"])
    lj_attrs = metadata_helpers.read_attrs(
        source_log_path,
        "metadata/lj",
    )

    required = [
        "epsilon_LJ",
        "sigma_LJ",
        "r_cut_LJ",
        "buffer_LJ",
        "lj_mode",
        "r_on_LJ",
    ]
    missing = [
        key
        for key in required
        if key not in lj_attrs
    ]

    if missing:
        raise KeyError(
            "Source thermalization log is missing LJ metadata: "
            + ", ".join(missing)
            + f". Source log: {source_log_path}"
        )

    return {
        key: lj_attrs[key]
        for key in required
    }


def _choose_spike_center(frame, random_location, seed):
    box = np.asarray(
        frame.configuration.box,
        dtype=np.float64,
    )
    box_lengths = np.array(
        [box[0], box[1], box[2]],
        dtype=np.float64,
    )

    if random_location:
        rng = np.random.default_rng(int(seed))
        return rng.uniform(
            low=-0.5 * box_lengths,
            high=0.5 * box_lengths,
        )

    return np.array(
        [0.0, 0.0, 0.0],
        dtype=np.float64,
    )


def _particle_masses(frame):
    N = int(frame.particles.N)

    try:
        masses = frame.particles.mass
    except Exception:
        masses = None

    if masses is None:
        return np.ones(N, dtype=np.float64)

    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape[0] != N:
        return np.ones(N, dtype=np.float64)

    return masses


def _copy_frame_with_velocities(frame, velocities):
    new_frame = gsd.hoomd.Frame()
    new_frame.configuration.step = int(frame.configuration.step)
    new_frame.configuration.box = list(frame.configuration.box)
    new_frame.particles.N = int(frame.particles.N)

    try:
        new_frame.particles.types = list(frame.particles.types)
    except Exception:
        new_frame.particles.types = ["A"]

    keep_mask = np.ones(int(frame.particles.N), dtype=bool)
    cavitation_helpers._copy_masked_particle_fields(
        source_frame=frame,
        new_frame=new_frame,
        keep_mask=keep_mask,
    )
    new_frame.particles.velocity = np.asarray(
        velocities,
        dtype=np.float64,
    ).copy()

    return new_frame


def _kinetic_energy(velocities, masses):
    velocities = np.asarray(velocities, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    return 0.5 * float(
        np.sum(masses * np.sum(velocities * velocities, axis=1))
    )


def make_hot_spike_frame_from_frame(
    frame,
    radius,
    injected_energy,
    method="velocity_rescale_com",
    random_location=False,
    location_seed=1,
    return_info=False,
):
    """
    Add kinetic energy inside a spherical region by velocity rescaling.

    ``injected_energy`` is the total kinetic energy to add in reduced LJ units.
    ``velocity_rescale_com`` preserves the selected group's COM velocity;
    ``velocity_rescale_raw`` rescales selected velocities directly.
    """

    method = str(method)
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            "method must be one of: " + ", ".join(sorted(SUPPORTED_METHODS))
        )

    radius = float(radius)
    injected_energy = float(injected_energy)

    if radius <= 0:
        raise ValueError("radius must be positive")
    if injected_energy <= 0:
        raise ValueError("injected_energy must be positive")

    source_box = np.asarray(
        frame.configuration.box,
        dtype=np.float64,
    )
    box_lengths = np.array(
        [source_box[0], source_box[1], source_box[2]],
        dtype=np.float64,
    )
    max_radius = 0.5 * float(np.min(box_lengths))
    if radius >= max_radius:
        raise ValueError(
            "radius must be smaller than half the shortest box length "
            f"({max_radius:g})"
        )

    positions = np.asarray(
        frame.particles.position,
        dtype=np.float64,
    )
    velocities = np.asarray(
        frame.particles.velocity,
        dtype=np.float64,
    ).copy()
    masses = _particle_masses(frame)

    center = _choose_spike_center(
        frame=frame,
        random_location=random_location,
        seed=location_seed,
    )
    distances = periodic_distances(positions, center, box_lengths)
    selected_mask = distances <= radius
    selected_indices = np.flatnonzero(selected_mask).astype(np.int64)

    if selected_indices.size == 0:
        raise RuntimeError(
            "Hot spike selected zero particles. Use a larger radius."
        )

    selected_velocities = velocities[selected_mask]
    selected_masses = masses[selected_mask]
    selected_ke_before = _kinetic_energy(
        selected_velocities,
        selected_masses,
    )

    total_mass = float(np.sum(selected_masses))
    selected_momentum_before = np.sum(
        selected_masses[:, None] * selected_velocities,
        axis=0,
    )
    com_velocity = selected_momentum_before / total_mass

    momentum_conserving = method == "velocity_rescale_com"

    if momentum_conserving:
        rescale_velocities = selected_velocities - com_velocity
    else:
        rescale_velocities = selected_velocities

    rescale_ke_before = _kinetic_energy(
        rescale_velocities,
        selected_masses,
    )
    if rescale_ke_before <= 0:
        raise RuntimeError(
            "Selected particles have zero rescalable kinetic energy."
        )

    scale_factor = float(
        np.sqrt((rescale_ke_before + injected_energy) / rescale_ke_before)
    )
    rescale_velocities_after = scale_factor * rescale_velocities

    if momentum_conserving:
        selected_velocities_after = rescale_velocities_after + com_velocity
    else:
        selected_velocities_after = rescale_velocities_after

    velocities[selected_mask] = selected_velocities_after

    selected_ke_after = _kinetic_energy(
        selected_velocities_after,
        selected_masses,
    )
    rescale_ke_after = _kinetic_energy(
        rescale_velocities_after,
        selected_masses,
    )
    selected_momentum_after = np.sum(
        selected_masses[:, None] * selected_velocities_after,
        axis=0,
    )

    new_frame = _copy_frame_with_velocities(
        frame=frame,
        velocities=velocities,
    )

    info = {
        "energy_dump_method": method,
        "radius": float(radius),
        "radius_definition": "absolute radius in simulation length units",
        "spike_center": center.copy(),
        "spike_center_x": float(center[0]),
        "spike_center_y": float(center[1]),
        "spike_center_z": float(center[2]),
        "random_location": bool(random_location),
        "location_seed_source": "source_thermalization_metadata_seed",
        "location_seed": int(location_seed),
        "periodic_distance": True,
        "requested_injected_energy": float(injected_energy),
        "actual_injected_energy": float(selected_ke_after - selected_ke_before),
        "energy_units": "reduced_lj",
        "energy_reference": "direct_input",
        "selected_particle_count": int(selected_indices.size),
        "energy_per_selected_particle": (
            float(injected_energy) / int(selected_indices.size)
        ),
        "selected_ke_before": float(selected_ke_before),
        "selected_ke_after": float(selected_ke_after),
        "selected_rescaled_ke_before": float(rescale_ke_before),
        "selected_rescaled_ke_after": float(rescale_ke_after),
        "velocity_scale_factor": float(scale_factor),
        "momentum_conserving": bool(momentum_conserving),
        "center_of_mass_velocity_preserved": bool(momentum_conserving),
        "selected_total_mass": float(total_mass),
        "selected_com_velocity_x": float(com_velocity[0]),
        "selected_com_velocity_y": float(com_velocity[1]),
        "selected_com_velocity_z": float(com_velocity[2]),
        "selected_momentum_before_x": float(selected_momentum_before[0]),
        "selected_momentum_before_y": float(selected_momentum_before[1]),
        "selected_momentum_before_z": float(selected_momentum_before[2]),
        "selected_momentum_after_x": float(selected_momentum_after[0]),
        "selected_momentum_after_y": float(selected_momentum_after[1]),
        "selected_momentum_after_z": float(selected_momentum_after[2]),
        "selected_particle_indices": selected_indices,
        "selected_particle_positions": positions[selected_mask].copy(),
    }

    if return_info:
        return new_frame, info

    return new_frame


def _frame_state_metadata(
    frame,
    n_fcc_cells,
    source_rho,
    kT,
    state_kind,
):
    box = np.asarray(
        frame.configuration.box,
        dtype=np.float64,
    )
    volume = float(box[0] * box[1] * box[2])
    N = int(frame.particles.N)

    return {
        "state_kind": state_kind,
        "data_version": "v3",
        "lattice_type": "fcc",
        "density_mode": "fixed_N_fixed_volume_velocity_rescaled",
        "n_fcc_cells": int(n_fcc_cells),
        "N": N,
        "source_rho": float(source_rho),
        "target_rho": float(N / volume),
        "actual_rho": float(N / volume),
        "kT": float(kT),
        "BoxLength": float(box[0]),
        "volume": volume,
        "fcc_cell_size": float(box[0]) / int(n_fcc_cells),
    }


def _build_source_metadata(
    source_result,
    source_rho,
    source_kT,
    source_nsteps,
    source_seed,
):
    source_paths = source_result["paths"]
    source_state_path = Path(source_paths["state_path"])
    source_log_path = Path(source_paths["log_path"])

    source = {
        "source_state_kind": "thermalized",
        "source_data_version": "v3",
        "source_state_path": str(source_state_path),
        "source_log_path": str(source_log_path),
        "source_rho": float(source_rho),
        "source_kT": float(source_kT),
        "source_nsteps": int(source_nsteps),
        "source_seed": int(source_seed),
    }

    if source_log_path.exists():
        source_state_attrs = metadata_helpers.read_attrs(
            source_log_path,
            "metadata/state",
        )
        source_run_attrs = metadata_helpers.read_attrs(
            source_log_path,
            "metadata/run",
        )

        for key in [
            "N",
            "actual_rho",
            "target_rho",
            "BoxLength",
            "volume",
        ]:
            if key in source_state_attrs:
                source[f"source_{key}"] = source_state_attrs[key]

        if "final_timestep" in source_run_attrs:
            source["source_final_timestep"] = source_run_attrs[
                "final_timestep"
            ]
        if "seed" in source_run_attrs:
            source["source_metadata_seed"] = source_run_attrs["seed"]

    return source


def _creation_metadata(info):
    skip_keys = {
        "selected_particle_indices",
        "selected_particle_positions",
    }
    return {
        key: value
        for key, value in info.items()
        if key not in skip_keys
    }


def _write_hot_spike_creation_metadata(
    metadata_path,
    frame,
    paths,
    info,
    source_metadata,
    n_fcc_cells,
    source_rho,
    kT,
    overwrite=False,
):
    state = _frame_state_metadata(
        frame=frame,
        n_fcc_cells=n_fcc_cells,
        source_rho=source_rho,
        kT=kT,
        state_kind="excitation_initial",
    )

    metadata_groups = {
        "metadata/state": state,
        "metadata/creation": _creation_metadata(info),
        "metadata/source": source_metadata,
        "metadata/paths": {
            "state_path": str(paths["state_path"]),
            "creation_metadata_path": str(paths["creation_metadata_path"]),
        },
    }

    datasets = {
        "metadata/creation/selected_particle_indices": info.get(
            "selected_particle_indices"
        ),
        "metadata/creation/selected_particle_positions": info.get(
            "selected_particle_positions"
        ),
    }

    metadata_helpers.write_metadata_groups(
        hdf5_path=metadata_path,
        groups=metadata_groups,
        mode="w" if overwrite else "a",
        overwrite=True,
    )
    metadata_helpers.write_datasets(
        hdf5_path=metadata_path,
        datasets=datasets,
        mode="a",
        overwrite=True,
    )
    metadata_helpers.clear_attrs(
        hdf5_path=metadata_path,
        group_path="metadata",
    )


def get_or_create_hot_spike_state(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius,
    injected_energy,
    method="velocity_rescale_com",
    source_seed=1,
    source_log_period=1_000,
    random_location=False,
    overwrite=False,
    overwrite_source=False,
    create_source_if_missing=True,
    reject_phase_separated_source=True,
    base_folder=EXCITATION_STATES_V3_ROOT,
):
    """
    Load or create a localized kinetic-energy excitation initial state.
    """

    source_result = cavitation_helpers.get_source_randomization_result(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        source_log_period=source_log_period,
        overwrite_source=overwrite_source,
        create_source_if_missing=create_source_if_missing,
    )

    source_metadata_seed = int(source_seed)
    if source_result["frame"] is not None:
        source_metadata_seed = _source_seed_from_metadata(
            source_result=source_result,
            fallback_seed=source_seed,
        )

    paths = excitation_state_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        method=method,
        radius=radius,
        energy=injected_energy,
        center=None,
        random_location=random_location,
        excitation_seed=source_metadata_seed,
        base_folder=base_folder,
    )

    if source_result["frame"] is None:
        return {
            "frame": None,
            "paths": paths,
            "source_result": source_result,
            "creation_info": {},
            "created_new": False,
            "status": "missing_source",
        }

    source_phase_separation = cavitation_helpers._source_phase_separation(
        source_result
    )
    if (
        reject_phase_separated_source
        and source_phase_separation["phase_separated"]
    ):
        print("Thermalized state phase separated; no hot spike will be done.")
        print("Skipping hot spike: thermalized source is phase separated.")
        print("=" * 70)
        print("source_state_path:", source_result["paths"]["state_path"])
        print("source_log_path:", source_result["paths"]["log_path"])
        print(
            "source_low_density_fraction:",
            source_phase_separation.get("low_density_fraction"),
        )
        print("=" * 70)
        return {
            "frame": None,
            "paths": paths,
            "source_result": source_result,
            "source_phase_separation": source_phase_separation,
            "creation_info": {},
            "created_new": False,
            "status": "source_phase_separated",
        }

    state_path = Path(paths["state_path"])
    metadata_path = Path(paths["creation_metadata_path"])
    source_metadata = _build_source_metadata(
        source_result=source_result,
        source_rho=target_rho,
        source_kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
    )

    if state_path.exists() and not overwrite:
        print()
        print()
        print("Loaded existing hot-spike initial state:")
        print(state_path)

        frame = _load_frame_from_gsd(state_path)
        creation_metadata = {}
        if metadata_path.exists():
            creation_metadata = metadata_helpers.read_attrs(
                metadata_path,
                "metadata/creation",
            )

        info = {
            "created_new": False,
            "state_path": str(state_path),
            "creation_metadata_path": str(metadata_path),
            "N": int(frame.particles.N),
        }
        info.update(creation_metadata)

        return {
            "frame": frame,
            "paths": paths,
            "source_result": source_result,
            "source_phase_separation": source_phase_separation,
            "creation_info": info,
            "created_new": False,
            "status": "loaded_initial",
        }

    frame, info = make_hot_spike_frame_from_frame(
        frame=source_result["frame"],
        radius=radius,
        injected_energy=injected_energy,
        method=method,
        random_location=random_location,
        location_seed=source_metadata_seed,
        return_info=True,
    )

    _save_frame_to_gsd(
        frame=frame,
        state_path=state_path,
        overwrite=overwrite,
    )
    _write_hot_spike_creation_metadata(
        metadata_path=metadata_path,
        frame=frame,
        paths=paths,
        info=info,
        source_metadata=source_metadata,
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        kT=kT,
        overwrite=True,
    )

    info["created_new"] = True
    info["state_path"] = str(state_path)
    info["creation_metadata_path"] = str(metadata_path)

    print("Created new hot-spike initial state")
    print("=" * 70)
    print("state_path:", state_path)
    print("creation_metadata_path:", metadata_path)
    print("method:", method)
    print("radius:", info["radius"])
    print("requested_injected_energy:", info["requested_injected_energy"])
    print("actual_injected_energy:", info["actual_injected_energy"])
    print("selected_particle_count:", info["selected_particle_count"])
    print("=" * 70)

    return {
        "frame": frame,
        "paths": paths,
        "source_result": source_result,
        "source_phase_separation": source_phase_separation,
        "creation_info": info,
        "created_new": True,
        "status": "created_initial",
    }


def infer_integrator_metadata(simulation):
    integrator = simulation.operations.integrator
    method_classes = []
    thermostat_classes = []

    if integrator is None:
        return {
            "ensemble": "unknown",
            "integrator_dt": np.nan,
            "integrator_method_classes": "",
            "thermostat_classes": "",
        }

    for method in integrator.methods:
        method_classes.append(type(method).__name__)
        thermostat = getattr(method, "thermostat", None)
        if thermostat is not None:
            thermostat_classes.append(type(thermostat).__name__)

    if method_classes == ["ConstantVolume"] and not thermostat_classes:
        ensemble = "NVE"
    elif method_classes == ["ConstantVolume"] and thermostat_classes:
        ensemble = "NVT"
    elif (
        method_classes in [
            ["ConstantPressure"],
            ["ConstantPressure", "ConstantVolume"],
        ]
        and not thermostat_classes
    ):
        ensemble = "NPH"
    elif method_classes == ["ConstantPressure"] and thermostat_classes:
        ensemble = "NPT"
    else:
        ensemble = "unknown"

    metadata = {
        "ensemble": ensemble,
        "integrator_dt": float(integrator.dt),
        "integrator_method_classes": ",".join(method_classes),
        "thermostat_classes": ",".join(thermostat_classes),
    }
    simulation_metadata = getattr(simulation, "metadata", {})
    if ensemble in {"NPH", "NPT"}:
        if simulation_metadata.get("pressure") is not None:
            metadata["pressure"] = float(simulation_metadata["pressure"])
        if simulation_metadata.get("tauS") is not None:
            metadata["tauS"] = float(simulation_metadata["tauS"])
        metadata["pressure_couple"] = str(
            simulation_metadata.get("pressure_couple", "xyz")
        )
        metadata["barostat_gamma"] = float(
            simulation_metadata.get("barostat_gamma", 0.0)
        )
        metadata["nph_masked"] = bool(
            simulation_metadata.get("nph_masked", False)
        )
        metadata["nph_outer_particle_count"] = int(
            simulation_metadata.get("nph_outer_particle_count", 0)
        )
        metadata["nph_inner_particle_count"] = int(
            simulation_metadata.get("nph_inner_particle_count", 0)
        )
        metadata["nph_rescale_all"] = bool(
            simulation_metadata.get("nph_mask_controls_box", False)
        )
        metadata["nph_mask_controls_box"] = bool(
            simulation_metadata.get("nph_mask_controls_box", False)
        )
        metadata["nph_pressure_filter"] = str(
            simulation_metadata.get("nph_pressure_filter", "all_particles")
        )
        for key in [
            "nph_mask_diameter_fraction",
            "nph_mask_radius",
            "nph_mask_reference_box_length",
            "nph_mask_center_x",
            "nph_mask_center_y",
            "nph_mask_center_z",
            "nph_mask_outer_fraction",
            "nph_mask_membership",
        ]:
            if key in simulation_metadata:
                metadata[key] = simulation_metadata[key]
    return metadata


def _build_evolution_metadata_groups(
    simulation,
    initial_result,
    evolved_paths,
    n_fcc_cells,
    source_rho,
    source_kT,
    source_nsteps,
    source_seed,
    evolve_nsteps,
    evolve_seed,
    log_period,
    trajectory_period,
    lj_kwargs,
):
    snapshot = simulation.state.get_snapshot()
    box = np.asarray(
        snapshot.configuration.box,
        dtype=np.float64,
    )
    volume = float(box[0] * box[1] * box[2])
    N = int(snapshot.particles.N)

    creation_paths = initial_result["paths"]
    creation_info = initial_result["creation_info"]
    source_result = initial_result["source_result"]
    integrator_metadata = infer_integrator_metadata(simulation)
    density_mode = "fixed_N_fixed_volume_velocity_rescaled"
    if integrator_metadata["ensemble"] in {"NPH", "NPT"}:
        density_mode = "fixed_N_variable_volume_velocity_rescaled"

    state = {
        "state_kind": "excitation_evolved",
        "data_version": "v3",
        "lattice_type": "fcc",
        "density_mode": density_mode,
        "n_fcc_cells": int(n_fcc_cells),
        "N": N,
        "source_rho": float(source_rho),
        "target_rho": float(N / volume),
        "actual_rho": float(N / volume),
        "kT": float(source_kT),
        "BoxLength": float(box[0]),
        "volume": volume,
        "fcc_cell_size": float(box[0]) / int(n_fcc_cells),
    }

    run = {
        "run_kind": "hot_spike_evolution",
        "phase_name": "hot_spike",
        "nsteps": int(evolve_nsteps),
        "seed": int(evolve_seed),
        "dt": float(integrator_metadata["integrator_dt"]),
        "log_period": int(log_period),
        "trajectory_period": int(trajectory_period),
        "includes_initial_frame": True,
        "includes_initial_log_row": True,
        "final_timestep": int(simulation.timestep) + int(evolve_nsteps),
    }
    run.update(integrator_metadata)

    source = {
        "source_state_kind": "excitation_initial",
        "source_state_path": str(creation_paths["state_path"]),
        "source_creation_metadata_path": str(
            creation_paths["creation_metadata_path"]
        ),
        "source_data_version": "v3",
        "parent_state_kind": "thermalized",
        "parent_state_path": str(source_result["paths"]["state_path"]),
        "parent_log_path": str(source_result["paths"]["log_path"]),
        "source_rho": float(source_rho),
        "source_kT": float(source_kT),
        "source_nsteps": int(source_nsteps),
        "source_seed": int(source_seed),
    }

    creation = _creation_metadata(creation_info)

    paths = {
        "log_path": str(evolved_paths["log_path"]),
        "trajectory_path": str(evolved_paths["trajectory_path"]),
        "final_state_path": str(evolved_paths["final_state_path"]),
    }

    groups = {
        "metadata/state": state,
        "metadata/run": run,
        "metadata/source": source,
        "metadata/creation": creation,
        "metadata/paths": paths,
        "metadata/lj": lj_kwargs,
    }

    return groups


def get_or_create_hot_spike_single_dt(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius,
    injected_energy,
    evolve_nsteps,
    method="velocity_rescale_com",
    source_seed=1,
    evolve_seed=1,
    dt=0.0005,
    source_log_period=1_000,
    log_period=1_000,
    trajectory_period=1_000,
    random_location=False,
    overwrite=False,
    overwrite_initial=False,
    overwrite_source=False,
    create_source_if_missing=True,
    reject_phase_separated_source=True,
):
    """
    Load or run a hot-spike excitation followed by NVE evolution.
    """

    initial_result = get_or_create_hot_spike_state(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        radius=radius,
        injected_energy=injected_energy,
        method=method,
        source_seed=source_seed,
        source_log_period=source_log_period,
        random_location=random_location,
        overwrite=overwrite_initial,
        overwrite_source=overwrite_source,
        create_source_if_missing=create_source_if_missing,
        reject_phase_separated_source=reject_phase_separated_source,
    )

    source_metadata_seed = int(source_seed)
    if initial_result["frame"] is not None:
        source_metadata_seed = _source_seed_from_metadata(
            source_result=initial_result["source_result"],
            fallback_seed=source_seed,
        )

    evolved_paths = legacy_excitation_evolved_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        method=method,
        radius=radius,
        energy=injected_energy,
        evolve_nsteps=evolve_nsteps,
        evolve_seed=evolve_seed,
        dt=dt,
        center=None,
        random_location=random_location,
        excitation_seed=source_metadata_seed,
    )

    if initial_result["frame"] is None:
        initial_status = initial_result.get("status", "missing_source")
        return {
            "frame": None,
            "paths": evolved_paths,
            "initial_result": initial_result,
            "created_new": False,
            "status": initial_status,
        }

    trajectory_path = Path(evolved_paths["trajectory_path"])
    final_state_path = Path(evolved_paths["final_state_path"])
    log_path = Path(evolved_paths["log_path"])

    if (
        trajectory_path.exists()
        and final_state_path.exists()
        and log_path.exists()
        and not overwrite
    ):
        print()
        print()
        print("Loaded existing hot-spike evolution:")
        print(final_state_path)

        return {
            "frame": _load_frame_from_gsd(final_state_path),
            "paths": evolved_paths,
            "initial_result": initial_result,
            "created_new": False,
            "status": "loaded_evolution",
        }

    lj_kwargs = _source_lj_kwargs(initial_result["source_result"])

    simulation = simulation_helpers.make_simulation(
        frame=initial_result["frame"],
        target_rho=target_rho,
        n_fcc_cells=n_fcc_cells,
        seed=evolve_seed,
        dt=dt,
        kT=kT,
        ensemble="NVE",
        starting_state_path=str(initial_result["paths"]["state_path"]),
        **lj_kwargs,
    )

    integrator_metadata = infer_integrator_metadata(simulation)
    if integrator_metadata["ensemble"] != "NVE":
        raise RuntimeError(
            "Hot-spike evolution must be NVE, but actual ensemble was "
            f"{integrator_metadata['ensemble']}."
        )

    metadata_groups = _build_evolution_metadata_groups(
        simulation=simulation,
        initial_result=initial_result,
        evolved_paths=evolved_paths,
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        source_kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        evolve_nsteps=evolve_nsteps,
        evolve_seed=evolve_seed,
        log_period=log_period,
        trajectory_period=trajectory_period,
        lj_kwargs=lj_kwargs,
    )

    with simulation_progress(
        "Excitation",
        ncells=n_fcc_cells,
        rho=target_rho,
        Source_kT=kT,
        nsteps=evolve_nsteps,
    ):
        run_result = run_helpers.run_logged_trajectory_phase(
            simulation=simulation,
            nsteps=evolve_nsteps,
            log_path=log_path,
            trajectory_path=trajectory_path,
            final_state_path=final_state_path,
            log_period=log_period,
            trajectory_period=trajectory_period,
            metadata_groups=metadata_groups,
            classify_final=True,
            classification_kwargs=None,
        )

    final_frame = _load_frame_from_gsd(final_state_path)
    classification_result = classification_helpers.read_phase_method_attrs(
        log_path,
        "voxel",
    )

    print("Created new hot-spike evolution")
    print("=" * 70)
    print("trajectory_path:", trajectory_path)
    print("final_state_path:", final_state_path)
    print("log_path:", log_path)
    print("trajectory_period:", trajectory_period)
    print("evolve_nsteps:", evolve_nsteps)
    print("dt:", dt)
    print("ensemble:", integrator_metadata["ensemble"])
    print("=" * 70)

    return {
        "frame": final_frame,
        "paths": evolved_paths,
        "initial_result": initial_result,
        "run_result": run_result,
        "classification_result": classification_result,
        "created_new": True,
        "status": "created_evolution",
    }


def get_or_create_hot_spike(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius,
    injected_energy,
    dt2=None,
    nsteps2=None,
    method="velocity_rescale_com",
    source_seed=1,
    evolve_seed=1,
    dt1=0.0005,
    nsteps1=200_000,
    source_log_period=1_000,
    log_period=1_000,
    trajectory_period=1_000,
    random_location=False,
    overwrite=False,
    overwrite_initial=False,
    overwrite_source=False,
    create_source_if_missing=True,
    reject_phase_separated_source=True,
    evolve_nsteps=None,
    dt=None,
    ensemble="NVE",
    pressure=None,
    tauS=None,
    pressure_couple="xyz",
    barostat_gamma=0.0,
    base_folder=None,
    outer_mask_diameter_fraction=0.75,
    pressure_tail_samples=100,
    nph_mask_controls_box=False,
):
    """
    Load or run the standard two-segment hot-spike evolution.

    Segment 1 defaults to ``dt1=0.0005`` for ``nsteps1=200_000``.
    Supply ``dt2`` and ``nsteps2`` for segment 2. The old ``dt`` and
    ``evolve_nsteps`` keywords are accepted as aliases during migration.
    Set ``ensemble='NPH'`` for an NPH comparison with a diagnostic outer
    pressure mask. When ``pressure`` is
    omitted, the helper uses the tail mean of the homogeneous thermalized
    source log before applying the excitation.
    When omitted, the conservative ``tauS`` default is ``10000 * dt2`` and
    remains constant across both segments.
    All particles control the box by default. The earlier split-integrator
    behavior is available only with ``nph_mask_controls_box=True``.
    """

    if dt2 is None:
        dt2 = dt
    if nsteps2 is None:
        nsteps2 = evolve_nsteps
    if dt2 is None:
        raise ValueError("dt2 is required")
    if nsteps2 is None:
        raise ValueError("nsteps2 is required")

    from .excitation_evolution import get_or_create_two_segment_hot_spike

    return get_or_create_two_segment_hot_spike(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        radius=radius,
        injected_energy=injected_energy,
        dt1=dt1,
        nsteps1=nsteps1,
        dt2=dt2,
        nsteps2=nsteps2,
        method=method,
        source_seed=source_seed,
        evolve_seed=evolve_seed,
        source_log_period=source_log_period,
        log_period=log_period,
        trajectory_period=trajectory_period,
        random_location=random_location,
        overwrite=overwrite,
        overwrite_initial=overwrite_initial,
        overwrite_source=overwrite_source,
        create_source_if_missing=create_source_if_missing,
        reject_phase_separated_source=reject_phase_separated_source,
        ensemble=ensemble,
        pressure=pressure,
        tauS=tauS,
        pressure_couple=pressure_couple,
        barostat_gamma=barostat_gamma,
        base_folder=base_folder,
        outer_mask_diameter_fraction=outer_mask_diameter_fraction,
        pressure_tail_samples=pressure_tail_samples,
        nph_mask_controls_box=nph_mask_controls_box,
    )
