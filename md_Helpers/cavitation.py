# cavitation.py

from pathlib import Path

import gsd.hoomd
import numpy as np

from . import classification as classification_helpers
from . import metadata as metadata_helpers
from . import runs as run_helpers
from . import simulation as simulation_helpers
from .paths import (
    CAVITATION_STATES_V3_ROOT,
    cavitation_evolved_paths,
    cavitation_state_paths,
    thermalized_run_paths,
)
from .spatial import periodic_distances


# ============================================================
# Small GSD helpers
# ============================================================

def load_frame_from_gsd(
    state_path,
    frame_index=-1,
):
    """
    Load one GSD frame from disk.
    """

    state_path = Path(state_path)

    if not state_path.exists():
        raise FileNotFoundError(f"GSD file does not exist: {state_path}")

    with gsd.hoomd.open(
        name=str(state_path),
        mode="r",
    ) as trajectory:
        frame = trajectory[frame_index]

    return frame


def save_frame_to_gsd(
    frame,
    state_path,
    overwrite=False,
):
    """
    Save one GSD frame.
    """

    state_path = Path(state_path)

    if state_path.exists() and not overwrite:
        raise FileExistsError(f"GSD file already exists: {state_path}")

    state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with gsd.hoomd.open(
        name=str(state_path),
        mode="w",
    ) as trajectory:
        trajectory.append(frame)

    return state_path


# ============================================================
# Bubble construction
# ============================================================

def choose_bubble_center(
    frame,
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
):
    """
    Choose the bubble center.

    Defaults to the center of the periodic box, (0, 0, 0). If
    random_location=True, choose a uniform random point inside the box.
    """

    box = np.asarray(
        frame.configuration.box,
        dtype=np.float64,
    )

    box_lengths = np.array(
        [box[0], box[1], box[2]],
        dtype=np.float64,
    )

    if random_location:
        if bubble_center is not None:
            raise ValueError(
                "Use either random_location=True or bubble_center, not both."
            )

        rng = np.random.default_rng(int(bubble_seed))

        return rng.uniform(
            low=-0.5 * box_lengths,
            high=0.5 * box_lengths,
        )

    if bubble_center is None:
        return np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    bubble_center = np.asarray(
        bubble_center,
        dtype=np.float64,
    )

    if bubble_center.shape != (3,):
        raise ValueError("bubble_center must have shape (3,)")

    return bubble_center


def _copy_masked_particle_fields(
    source_frame,
    new_frame,
    keep_mask,
):
    """
    Copy per-particle fields from source_frame to new_frame.
    """

    source_particles = source_frame.particles
    new_particles = new_frame.particles

    n_before = int(source_particles.N)

    particle_fields = [
        "position",
        "typeid",
        "velocity",
        "mass",
        "charge",
        "diameter",
        "body",
        "image",
        "orientation",
        "moment_inertia",
        "angular_momentum",
    ]

    copied_fields = []

    for field_name in particle_fields:
        try:
            value = getattr(source_particles, field_name)
        except Exception:
            continue

        if value is None:
            continue

        array = np.asarray(value)

        if array.ndim == 0:
            continue

        if array.shape[0] != n_before:
            continue

        setattr(
            new_particles,
            field_name,
            array[keep_mask].copy(),
        )

        copied_fields.append(field_name)

    if "position" not in copied_fields:
        raise RuntimeError(
            "Could not copy particle positions from source frame."
        )

    if "typeid" not in copied_fields:
        new_particles.typeid = np.zeros(
            int(new_particles.N),
            dtype=np.uint32,
        )

    return copied_fields


def make_cavitated_frame_from_frame(
    frame,
    radius,
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
    return_info=False,
):
    """
    Create a cavitated frame by removing particles inside one sphere.

    ``radius`` is the absolute bubble radius in simulation length units.

    The box is not resized, so the post-cavitation density is
    N_after / BoxLength**3.
    """

    radius = float(radius)

    if radius <= 0:
        raise ValueError("radius must be positive")

    source_box = np.asarray(
        frame.configuration.box,
        dtype=np.float64,
    )

    box_lengths = np.array(
        [source_box[0], source_box[1], source_box[2]],
        dtype=np.float64,
    )

    BoxLength = float(source_box[0])

    positions = np.asarray(
        frame.particles.position,
        dtype=np.float64,
    )

    N_before = int(frame.particles.N)
    volume = float(np.prod(box_lengths))
    rho_before = N_before / volume

    center = choose_bubble_center(
        frame=frame,
        random_location=random_location,
        bubble_seed=bubble_seed,
        bubble_center=bubble_center,
    )

    max_radius = 0.5 * float(np.min(box_lengths))
    if radius >= max_radius:
        raise ValueError(
            "radius must be smaller than half the shortest box length "
            f"({max_radius:g})"
        )

    distances = periodic_distances(positions, center, box_lengths)

    remove_mask = distances <= radius
    keep_mask = ~remove_mask

    removed_indices = np.flatnonzero(remove_mask).astype(np.int64)
    kept_indices = np.flatnonzero(keep_mask).astype(np.int64)

    N_removed = int(removed_indices.size)
    N_after = int(kept_indices.size)
    rho_after = N_after / volume

    if N_after <= 0:
        raise RuntimeError(
            "Cavitation removed all particles. "
            "Use a smaller radius."
        )

    new_frame = gsd.hoomd.Frame()
    new_frame.configuration.step = int(frame.configuration.step)
    new_frame.configuration.box = list(source_box)
    new_frame.particles.N = N_after

    try:
        new_frame.particles.types = list(frame.particles.types)
    except Exception:
        new_frame.particles.types = ["A"]

    copied_fields = _copy_masked_particle_fields(
        source_frame=frame,
        new_frame=new_frame,
        keep_mask=keep_mask,
    )

    info = {
        "bubble_method": "remove_particles_in_sphere",
        "radius": float(radius),
        "radius_definition": "absolute radius in simulation length units",
        "bubble_radius": float(radius),
        "bubble_center": center.copy(),
        "bubble_center_x": float(center[0]),
        "bubble_center_y": float(center[1]),
        "bubble_center_z": float(center[2]),
        "random_location": bool(random_location),
        "bubble_seed": int(bubble_seed),
        "BoxLength": float(BoxLength),
        "volume": float(volume),
        "N_before": int(N_before),
        "N_after": int(N_after),
        "particles_removed": int(N_removed),
        "particle_fraction_removed": float(N_removed / N_before),
        "rho_before": float(rho_before),
        "rho_after": float(rho_after),
        "periodic_distance": True,
        "copied_particle_fields": copied_fields,
        "removed_particle_indices": removed_indices,
        "removed_particle_positions": positions[remove_mask].copy(),
    }

    if return_info:
        return new_frame, info

    return new_frame


# ============================================================
# Source state loading
# ============================================================

def get_source_randomization_result(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    source_seed=1,
    source_phase_name="randomization",
    source_log_period=1_000,
    overwrite_source=False,
    create_source_if_missing=True,
):
    """
    Get the thermalized source state used to create the bubble.

    When create_source_if_missing=True, missing source files trigger a new
    thermalization run. When False, print the missing paths and return with
    status="missing_source" without starting a simulation.
    """

    source_paths = thermalized_run_paths(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=source_nsteps,
        seed=source_seed,
        phase_name=source_phase_name,
    )

    source_state_path = Path(source_paths["state_path"])
    source_log_path = Path(source_paths["log_path"])

    missing_paths = []

    if not source_state_path.exists():
        missing_paths.append(source_state_path)

    if not source_log_path.exists():
        missing_paths.append(source_log_path)

    if missing_paths:
        print("No source thermalized state found for the specified values.")
        print("=" * 70)
        print("n_fcc_cells       =", n_fcc_cells)
        print("target_rho        =", target_rho)
        print("kT                =", kT)
        print("source_nsteps     =", source_nsteps)
        print("source_seed       =", source_seed)
        print("source_phase_name =", source_phase_name)
        print()
        print("Missing paths:")

        for missing_path in missing_paths:
            print(missing_path)

        print("=" * 70)

        if not create_source_if_missing:
            print(
                "create_source_if_missing=False; "
                "no thermalization or cavitation was started."
            )
            return {
                "frame": None,
                "simulation": None,
                "paths": source_paths,
                "created_new": False,
                "status": "missing_source",
            }

        print("create_source_if_missing=True; starting thermalization now.")

    result = simulation_helpers.get_or_make_thermalized_state(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=source_nsteps,
        phase_name=source_phase_name,
        log_period=source_log_period,
        seed=source_seed,
        overwrite=overwrite_source,
    )
    result["status"] = "created_source" if result["created_new"] else "loaded_source"
    return result


def _source_phase_separation(source_result):
    """Read, or backfill, the source thermalization voxel classifier."""

    source_paths = source_result["paths"]
    source_log_path = Path(source_paths["log_path"])
    source_state_path = Path(source_paths["state_path"])
    result, metadata_path = classification_helpers.read_phase_method_attrs(
        source_log_path,
        "voxel",
    )

    if "phase_separated" not in result:
        result = classification_helpers.write_voxel_phase_separation_metadata(
            log_path=source_log_path,
            state_path=source_state_path,
            updated_from_saved_gsd=True,
        )

    result = dict(result)
    result["metadata_path"] = metadata_path
    return result


def _frame_state_metadata(
    frame,
    n_fcc_cells,
    source_rho,
    kT,
    state_kind,
    density_mode="fixed_volume_particle_removed",
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
        "density_mode": density_mode,
        "n_fcc_cells": int(n_fcc_cells),
        "N": N,
        "source_rho": float(source_rho),
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
    source_phase_name,
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
        "source_phase_name": source_phase_name,
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

    return source


def _build_creation_metadata(
    info,
):
    skip_keys = {
        "removed_particle_indices",
        "removed_particle_positions",
    }

    return {
        key: value
        for key, value in info.items()
        if key not in skip_keys
    }


def _write_cavitation_creation_metadata(
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
        state_kind="cavitation_initial",
    )

    state["target_rho"] = float(info["rho_after"])

    metadata_groups = {
        "metadata/state": state,
        "metadata/creation": _build_creation_metadata(info),
        "metadata/source": source_metadata,
        "metadata/paths": {
            "state_path": str(paths["state_path"]),
            "creation_metadata_path": str(paths["creation_metadata_path"]),
        },
    }

    datasets = {
        "metadata/creation/removed_particle_indices": info.get(
            "removed_particle_indices"
        ),
        "metadata/creation/removed_particle_positions": info.get(
            "removed_particle_positions"
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


# ============================================================
# Cavitation initial state
# ============================================================

def get_or_create_cavitation_state(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius,
    source_seed=1,
    source_phase_name="randomization",
    source_log_period=1_000,
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
    overwrite=False,
    overwrite_source=False,
    create_source_if_missing=True,
    reject_phase_separated_source=True,
    base_folder=CAVITATION_STATES_V3_ROOT,
):
    """
    Load or create a V3 cavitation starting state.

    ``radius`` is the absolute starting radius in simulation length units.

    Missing thermalized source states are created automatically by default.
    Set create_source_if_missing=False for a no-run existence check.
    Phase-separated thermalized sources are rejected by default.

    Saves:
        cavitation_initial.gsd
        cavitation_creation.hdf5
    """

    paths = cavitation_state_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        radius=radius,
        source_phase_name=source_phase_name,
        center=bubble_center,
        random_location=random_location,
        bubble_seed=bubble_seed,
        base_folder=base_folder,
    )

    source_result = get_source_randomization_result(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
        source_log_period=source_log_period,
        overwrite_source=overwrite_source,
        create_source_if_missing=create_source_if_missing,
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

    source_phase_separation = _source_phase_separation(source_result)
    if (
        reject_phase_separated_source
        and source_phase_separation["phase_separated"]
    ):
        print("Skipping cavitation: thermalized source is phase separated.")
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

    source_frame = source_result["frame"]
    state_path = Path(paths["state_path"])
    metadata_path = Path(paths["creation_metadata_path"])

    source_metadata = _build_source_metadata(
        source_result=source_result,
        source_rho=target_rho,
        source_kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
    )

    if state_path.exists() and not overwrite:
        print("Loaded existing cavitation initial state:")
        print(state_path)

        frame = load_frame_from_gsd(state_path)

        if metadata_path.exists():
            creation_metadata = metadata_helpers.read_attrs(
                metadata_path,
                "metadata/creation",
            )
        else:
            creation_metadata = {}

        info = {
            "created_new": False,
            "state_path": str(state_path),
            "creation_metadata_path": str(metadata_path),
            "N_after": int(frame.particles.N),
            "rho_after": (
                int(frame.particles.N)
                / float(frame.configuration.box[0])**3
            ),
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

    frame, info = make_cavitated_frame_from_frame(
        frame=source_frame,
        radius=radius,
        random_location=random_location,
        bubble_seed=bubble_seed,
        bubble_center=bubble_center,
        return_info=True,
    )

    save_frame_to_gsd(
        frame=frame,
        state_path=state_path,
        overwrite=overwrite,
    )

    _write_cavitation_creation_metadata(
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

    print("Created new cavitation initial state")
    print("=" * 70)
    print("state_path:", state_path)
    print("creation_metadata_path:", metadata_path)
    print("radius:", info["radius"])
    print("bubble_center:", info["bubble_center"])
    print("particles_removed:", info["particles_removed"])
    print("N_before:", info["N_before"])
    print("N_after:", info["N_after"])
    print("rho_before:", info["rho_before"])
    print("rho_after:", info["rho_after"])
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


# ============================================================
# Cavitation evolution
# ============================================================

def _build_evolution_metadata_groups(
    simulation,
    initial_result,
    evolved_paths,
    n_fcc_cells,
    source_rho,
    source_kT,
    source_nsteps,
    source_seed,
    source_phase_name,
    evolve_kT,
    evolve_nsteps,
    evolve_seed,
    dt,
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

    state = {
        "state_kind": "cavitation_evolved",
        "data_version": "v3",
        "lattice_type": "fcc",
        "density_mode": "fixed_volume_particle_removed",
        "n_fcc_cells": int(n_fcc_cells),
        "N": N,
        "source_rho": float(source_rho),
        "target_rho": float(N / volume),
        "actual_rho": float(N / volume),
        "kT": float(evolve_kT),
        "BoxLength": float(box[0]),
        "volume": volume,
        "fcc_cell_size": float(box[0]) / int(n_fcc_cells),
    }

    run = {
        "run_kind": "cavitation_evolution",
        "phase_name": "cavitation",
        "nsteps": int(evolve_nsteps),
        "seed": int(evolve_seed),
        "dt": float(dt),
        "kT": float(evolve_kT),
        "log_period": int(log_period),
        "trajectory_period": int(trajectory_period),
        "includes_initial_frame": True,
        "includes_initial_log_row": True,
        "final_timestep": int(simulation.timestep) + int(evolve_nsteps),
    }

    source = {
        "source_state_kind": "cavitation_initial",
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
        "source_phase_name": source_phase_name,
    }

    creation = {
        key: value
        for key, value in creation_info.items()
        if key not in {
            "removed_particle_indices",
            "removed_particle_positions",
            "created_new",
            "state_path",
            "creation_metadata_path",
        }
    }

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
    }

    lj_metadata = {
        key: value
        for key, value in lj_kwargs.items()
        if value is not None
    }

    if lj_metadata:
        groups["metadata/lj"] = lj_metadata

    return groups


def get_or_create_cavitation(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius,
    evolve_nsteps,
    evolve_kT=None,
    evolve_seed=1,
    source_seed=1,
    source_phase_name="randomization",
    source_log_period=1_000,
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
    dt=0.005,
    log_period=1_000,
    trajectory_period=1_000,
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    buffer_LJ=0.4,
    lj_mode="xplor",
    r_on_LJ=2.0,
    overwrite=False,
    overwrite_initial=False,
    overwrite_source=False,
    create_source_if_missing=True,
    reject_phase_separated_source=True,
    classify_final=True,
    classification_kwargs=None,
):
    """
    Load or run a V3 cavitation evolution.

    ``radius`` is the absolute starting radius in simulation length units.

    Missing thermalized source states are created automatically by default.
    Set create_source_if_missing=False for a no-run existence check.
    Phase-separated thermalized sources are rejected by default.

    Saves:
        cavitation_trajectory.gsd
        cavitation_final.gsd
        cavitation_log.hdf5
    """

    if evolve_kT is None:
        evolve_kT = kT

    evolved_paths = cavitation_evolved_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        radius=radius,
        evolve_kT=evolve_kT,
        evolve_nsteps=evolve_nsteps,
        evolve_seed=evolve_seed,
        source_phase_name=source_phase_name,
        center=bubble_center,
        random_location=random_location,
        bubble_seed=bubble_seed,
    )

    initial_result = get_or_create_cavitation_state(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        radius=radius,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
        source_log_period=source_log_period,
        random_location=random_location,
        bubble_seed=bubble_seed,
        bubble_center=bubble_center,
        overwrite=overwrite_initial,
        overwrite_source=overwrite_source,
        create_source_if_missing=create_source_if_missing,
        reject_phase_separated_source=reject_phase_separated_source,
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
        print("Loaded existing cavitation evolution:")
        print(final_state_path)

        return {
            "frame": load_frame_from_gsd(final_state_path),
            "paths": evolved_paths,
            "initial_result": initial_result,
            "created_new": False,
            "status": "loaded_evolution",
        }

    initial_frame = initial_result["frame"]
    creation_info = initial_result["creation_info"]

    simulation = simulation_helpers.make_simulation(
        frame=initial_frame,
        target_rho=creation_info.get("rho_after"),
        n_fcc_cells=n_fcc_cells,
        seed=evolve_seed,
        dt=dt,
        kT=evolve_kT,
        epsilon_LJ=epsilon_LJ,
        sigma_LJ=sigma_LJ,
        r_cut_LJ=r_cut_LJ,
        buffer_LJ=buffer_LJ,
        lj_mode=lj_mode,
        r_on_LJ=r_on_LJ,
        starting_state_path=str(initial_result["paths"]["state_path"]),
    )

    lj_kwargs = {
        "epsilon_LJ": epsilon_LJ,
        "sigma_LJ": sigma_LJ,
        "r_cut_LJ": r_cut_LJ,
        "r_on_LJ": r_on_LJ,
        "buffer_LJ": buffer_LJ,
        "lj_mode": lj_mode,
    }

    metadata_groups = _build_evolution_metadata_groups(
        simulation=simulation,
        initial_result=initial_result,
        evolved_paths=evolved_paths,
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        source_kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
        evolve_kT=evolve_kT,
        evolve_nsteps=evolve_nsteps,
        evolve_seed=evolve_seed,
        dt=dt,
        log_period=log_period,
        trajectory_period=trajectory_period,
        lj_kwargs=lj_kwargs,
    )

    run_result = run_helpers.run_logged_trajectory_phase(
        simulation=simulation,
        nsteps=evolve_nsteps,
        log_path=log_path,
        trajectory_path=trajectory_path,
        final_state_path=final_state_path,
        log_period=log_period,
        trajectory_period=trajectory_period,
        metadata_groups=metadata_groups,
        classify_final=classify_final,
        classification_kwargs=classification_kwargs,
    )

    final_frame = load_frame_from_gsd(final_state_path)

    print("Created new cavitation evolution")
    print("=" * 70)
    print("trajectory_path:", trajectory_path)
    print("final_state_path:", final_state_path)
    print("log_path:", log_path)
    print("trajectory_period:", trajectory_period)
    print("evolve_nsteps:", evolve_nsteps)
    print("=" * 70)

    return {
        "frame": final_frame,
        "paths": evolved_paths,
        "initial_result": initial_result,
        "run_result": run_result,
        "created_new": True,
        "status": "created_evolution",
    }
