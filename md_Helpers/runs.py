# runs.py

from pathlib import Path

import hoomd
import pandas as pd
import h5py
import gsd.hoomd
import numpy as np

from . import classification as ps
from . import metadata as metadata_helpers
from .classification import classify_final_state
from .paths import THERMALIZED_STATES_V2_ROOT, THERMALIZED_STATES_V3_ROOT

# ============================================================
# Default phase-separation settings
# ============================================================

DEFAULT_PHASE_SEP_NBINS = ps.DEFAULT_PHASE_SEP_NBINS
DEFAULT_PHASE_SEP_DENSITY_THRESHOLD = ps.DEFAULT_PHASE_SEP_DENSITY_THRESHOLD
DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD = ps.DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD




# ============================================================
# Backward-compatible phase-separation aliases
# ============================================================

check_phase_separated = ps.check_phase_separated
compute_phase_separation_from_frame = ps.compute_phase_separation_from_frame
write_phase_separation_metadata = ps.write_phase_separation_metadata
update_all_v2_phase_separation_metadata = ps.update_all_v2_phase_separation_metadata




# ============================================================
# Start HDF5 logger
# ============================================================

def start_hdf5_logger(
    simulation,
    log_path,
    log_period=1_000,
):
    """
    Start an HDF5 logger for a HOOMD simulation.
    """

    # ============================================================
    # Prepare path
    # ============================================================
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Thermodynamic compute
    # ============================================================
    thermo = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )

    simulation.operations.computes.append(thermo)

    # ============================================================
    # Logger
    # ============================================================
    logger = hoomd.logging.Logger(
        categories=["scalar", "sequence"]
    )

    logger.add(
        simulation,
        quantities=[
            "timestep",
            "tps",
        ],
    )

    logger.add(
        thermo,
        quantities=[
            "kinetic_temperature",
            "pressure",
            "pressure_tensor",
            "potential_energy",
            "kinetic_energy",
        ],
    )

    # ============================================================
    # HDF5 writer
    # ============================================================
    hdf5_writer = hoomd.write.HDF5Log(
        trigger=hoomd.trigger.Periodic(log_period),
        filename=str(log_path),
        mode="w",
        logger=logger,
    )

    simulation.operations.writers.append(hdf5_writer)

    # ============================================================
    # Return useful objects
    # ============================================================
    logger_objects = {
        "thermo": thermo,
        "logger": logger,
        "writer": hdf5_writer,
        "log_path": log_path,
        "log_period": log_period,
    }

    print("Started HDF5 logger")
    print("Log file:", log_path)
    print("Log period:", log_period)

    return logger_objects


# ============================================================
# Stop HDF5 logger
# ============================================================

def stop_hdf5_logger(
    simulation,
    logger_objects,
):
    """
    Stop an active HDF5 logger.
    """

    # ============================================================
    # Get logger objects
    # ============================================================
    thermo = logger_objects["thermo"]
    hdf5_writer = logger_objects["writer"]

    # ============================================================
    # Remove writer
    # ============================================================
    if hdf5_writer in simulation.operations.writers:
        simulation.operations.writers.remove(hdf5_writer)

    # ============================================================
    # Remove compute
    # ============================================================
    if thermo in simulation.operations.computes:
        simulation.operations.computes.remove(thermo)

    print("Stopped HDF5 logger")


# ============================================================
# Write HDF5 metadata
# ============================================================

def write_hdf5_metadata(
    log_path,
    metadata,
    group_name="metadata",
):
    """
    Write V3 structured metadata into an existing HDF5 log file.

    Bare /metadata is kept as a container only. Attributes are written under:
        metadata/state
        metadata/run
        metadata/lj
        metadata/source
        metadata/paths
        metadata/classification/phase_separation
    """

    # ============================================================
    # Prepare path
    # ============================================================
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    if group_name != "metadata":
        metadata_helpers.write_metadata_groups(
            hdf5_path=log_path,
            groups={group_name: metadata},
            mode="a",
            overwrite=True,
        )

    else:
        groups = metadata_helpers.split_simulation_metadata(
            metadata,
            state_kind=metadata.get("state_kind", "thermalized"),
            run_kind=metadata.get("run_kind", "thermalization"),
            data_version=metadata.get("data_version", "v3"),
        )

        metadata_helpers.write_metadata_groups(
            hdf5_path=log_path,
            groups=groups,
            mode="a",
            overwrite=True,
        )

        metadata_helpers.clear_attrs(
            hdf5_path=log_path,
            group_path="metadata",
        )

    print("Wrote HDF5 metadata")
    print("Log file:", log_path)
    print("Metadata group:", group_name)


# ============================================================
# Save final simulation state
# ============================================================

# ============================================================
# Save final simulation state
# ============================================================

def simulation_state_to_gsd_frame(
    simulation,
):
    """
    Convert the current HOOMD simulation state into a GSD frame.
    """

    # ============================================================
    # Get HOOMD snapshot
    # ============================================================
    snapshot = simulation.state.get_snapshot()

    # ============================================================
    # Convert snapshot to GSD frame
    # ============================================================
    frame = gsd.hoomd.Frame()

    frame.configuration.step = simulation.timestep
    frame.configuration.box = [
        snapshot.configuration.box[0],
        snapshot.configuration.box[1],
        snapshot.configuration.box[2],
        snapshot.configuration.box[3],
        snapshot.configuration.box[4],
        snapshot.configuration.box[5],
    ]

    frame.particles.N = snapshot.particles.N
    frame.particles.types = list(snapshot.particles.types)

    frame.particles.position = snapshot.particles.position
    frame.particles.typeid = snapshot.particles.typeid
    frame.particles.velocity = snapshot.particles.velocity

    return frame


def save_final_state(
    simulation,
    gsd_path,
):
    """
    Save the current simulation state as a single-frame GSD file.

    This now saves:
    - positions
    - type IDs
    - velocities
    """

    frame = simulation_state_to_gsd_frame(simulation)

    # ============================================================
    # Prepare path
    # ============================================================
    gsd_path = Path(gsd_path)
    gsd_path.parent.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Save GSD
    # ============================================================
    with gsd.hoomd.open(
        name=str(gsd_path),
        mode="w",
    ) as f:
        f.append(frame)

    print("Saved final state")
    print("GSD file:", gsd_path)


def write_current_state_to_trajectory(
    simulation,
    trajectory_path,
    mode="w",
):
    """
    Write the current simulation state into a trajectory GSD file.

    Use mode="w" to create a new trajectory with the current state as frame 0,
    or mode="a" to append the current state to an existing trajectory.
    """

    frame = simulation_state_to_gsd_frame(simulation)

    trajectory_path = Path(trajectory_path)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(
        name=str(trajectory_path),
        mode=mode,
    ) as trajectory:
        trajectory.append(frame)

    return trajectory_path


# ============================================================
# Build simulation metadata
# ============================================================

def build_simulation_metadata(
    simulation,
    phase_name="simulation",
    lattice_type="fcc",
    density_mode="fixed_N_variable_L",
    n_fcc_cells=None,
    target_rho=None,
    seed=None,
    dt=None,
    kT=None,
    epsilon_LJ=None,
    sigma_LJ=None,
    r_cut_LJ=None,
    r_on_LJ=None,
    buffer_LJ=None,
    lj_mode=None,
    log_period=None,
    nsteps=None,
    starting_state_path=None,
    phase_separated=None,
):
    """
    Build a metadata dictionary for the current simulation state.

    V3 convention:
    - n_fcc_cells is the chosen system size
    - N = 4 * n_fcc_cells**3
    - target_rho is the requested density
    - BoxLength is derived from N / target_rho
    - actual_rho is recomputed from the saved state
    """

    # ============================================================
    # Extract current simulation state information
    # ============================================================
    snapshot = simulation.state.get_snapshot()

    N = int(snapshot.particles.N)

    Lx = float(snapshot.configuration.box[0])
    Ly = float(snapshot.configuration.box[1])
    Lz = float(snapshot.configuration.box[2])

    volume = Lx * Ly * Lz
    actual_rho = N / volume

    # ============================================================
    # Derived FCC metadata
    # ============================================================
    fcc_cell_size = None
    
    if n_fcc_cells is not None:
        n_fcc_cells = int(n_fcc_cells)
        fcc_cell_size = Lx / n_fcc_cells

    
    # ============================================================
    # Build metadata dictionary
    # ============================================================
    metadata = {
        "phase_name": phase_name,
        "lattice_type": lattice_type,
        "density_mode": density_mode,

        "n_fcc_cells": n_fcc_cells,

        "N": N,
        "target_rho": target_rho,
        "actual_rho": actual_rho,

        "BoxLength": Lx,
        "volume": volume,

        "fcc_cell_size": fcc_cell_size,

        "seed": seed,
        "dt": dt,
        "kT": kT,

        "epsilon_LJ": epsilon_LJ,
        "sigma_LJ": sigma_LJ,
        "r_cut_LJ": r_cut_LJ,
        "r_on_LJ": r_on_LJ,
        "buffer_LJ": buffer_LJ,
        "lj_mode": lj_mode,

        "log_period": log_period,
        "nsteps": nsteps,
        "final_timestep": simulation.timestep,

        "starting_state_path": starting_state_path,
        "phase_separated": None,
    }

    # ============================================================
    # Remove None values
    # ============================================================
    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }

    return metadata


# ============================================================
# Build phase file paths
# ============================================================

def get_phase_paths(
    n_fcc_cells,
    target_rho,
    kT,
    nsteps,
    seed,
    phase_name,
    base_folder=THERMALIZED_STATES_V3_ROOT,
):
    """
    Build standard V3 paths for a logged simulation phase.

    Folder structure:

        Thermalized_States_v3/
            FCC/
                n_cells_30/
                    rho_0.500/
                        kT_0.400/
                            nsteps_1000000/
                                seed_1/
                                    randomization.gsd
                                    randomization_log.hdf5
    """

    # ============================================================
    # Format folder names
    # ============================================================
    n_cells_str = f"{int(n_fcc_cells)}"
    rho_str = f"{target_rho:.3f}"
    kT_str = f"{kT:.3f}"
    nsteps_str = f"{int(nsteps)}"

    if seed is None:
        seed_str = "unknown"
    else:
        seed_str = f"{int(seed)}"

    # ============================================================
    # Build folder
    # ============================================================
    folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{n_cells_str}"
        / f"rho_{rho_str}"
        / f"kT_{kT_str}"
        / f"nsteps_{nsteps_str}"
        / f"seed_{seed_str}"
    )

    # ============================================================
    # Build files
    # ============================================================
    log_path = folder / f"{phase_name}_log.hdf5"
    state_path = folder / f"{phase_name}.gsd"

    return {
        "folder": folder,
        "log_path": log_path,
        "state_path": state_path,

        "phase_name": phase_name,
        "n_fcc_cells": int(n_fcc_cells),
        "target_rho": target_rho,
        "kT": kT,
        "nsteps": int(nsteps),
        "seed": seed,
    }


# ============================================================
# Run logged phase
# ============================================================

def run_logged_phase(
    simulation,
    n_fcc_cells,
    target_rho,
    phase_name="simulation",
    nsteps=500_000,
    log_period=1_000,
    seed=None,
    dt=None,
    kT=None,
    epsilon_LJ=None,
    sigma_LJ=None,
    r_cut_LJ=None,
    r_on_LJ=None,
    buffer_LJ=None,
    lj_mode=None,
    starting_state_path="unknown",
    base_folder=THERMALIZED_STATES_V3_ROOT,
):
    """
    Run a simulation phase with HDF5 logging and save the final state.

    V3 path convention:
    - input system size is n_fcc_cells
    - input density is target_rho
    - BoxLength is not part of the path because it is derived
    """

    # ============================================================
    # Pull missing values from simulation metadata if available
    # ============================================================
    if hasattr(simulation, "metadata"):
        sim_metadata = simulation.metadata

        if seed is None:
            seed = sim_metadata.get("seed", seed)

        if dt is None:
            dt = sim_metadata.get("dt", dt)

        if kT is None:
            kT = sim_metadata.get("kT", kT)

        if epsilon_LJ is None:
            epsilon_LJ = sim_metadata.get("epsilon_LJ", epsilon_LJ)

        if sigma_LJ is None:
            sigma_LJ = sim_metadata.get("sigma_LJ", sigma_LJ)

        if r_cut_LJ is None:
            r_cut_LJ = sim_metadata.get("r_cut_LJ", r_cut_LJ)

        if r_on_LJ is None:
            r_on_LJ = sim_metadata.get("r_on_LJ", r_on_LJ)

        if buffer_LJ is None:
            buffer_LJ = sim_metadata.get("buffer_LJ", buffer_LJ)

        if lj_mode is None:
            lj_mode = sim_metadata.get("lj_mode", lj_mode)

        if starting_state_path in [None, "unknown"]:
            starting_state_path = sim_metadata.get(
                "starting_state_path",
                "unknown",
            )

    # ============================================================
    # Build paths
    # ============================================================
    paths = get_phase_paths(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=nsteps,
        seed=seed,
        phase_name=phase_name,
        base_folder=base_folder,
    )

    # ============================================================
    # Start logger
    # ============================================================
    logger_handle = start_hdf5_logger(
        simulation=simulation,
        log_path=paths["log_path"],
        log_period=log_period,
    )

    # ============================================================
    # Run simulation
    # ============================================================
    simulation.run(0)
    simulation.run(nsteps)

    # ============================================================
    # Stop logger
    # ============================================================
    stop_hdf5_logger(
        simulation=simulation,
        logger_objects=logger_handle,
    )

    # ============================================================
    # Build metadata
    # ============================================================
    metadata = build_simulation_metadata(
        simulation=simulation,
        phase_name=phase_name,
        lattice_type="fcc",
        density_mode="fixed_N_variable_L",
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        seed=seed,
        dt=dt,
        kT=kT,
        epsilon_LJ=epsilon_LJ,
        sigma_LJ=sigma_LJ,
        r_cut_LJ=r_cut_LJ,
        r_on_LJ=r_on_LJ,
        buffer_LJ=buffer_LJ,
        lj_mode=lj_mode,
        log_period=log_period,
        nsteps=nsteps,
        starting_state_path=starting_state_path,
        phase_separated=None,
    )

    metadata["log_path"] = str(paths["log_path"])
    metadata["state_path"] = str(paths["state_path"])

    # ============================================================
    # Write metadata and save state
    # ============================================================
    write_hdf5_metadata(
        log_path=paths["log_path"],
        metadata=metadata,
    )

    save_final_state(
        simulation=simulation,
        gsd_path=paths["state_path"],
    )

    ps.write_voxel_phase_separation_metadata(
        log_path=paths["log_path"],
        state_path=paths["state_path"],
        nbins=DEFAULT_PHASE_SEP_NBINS,
        density_threshold=DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
        voxel_fraction_threshold=DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
        updated_from_saved_gsd=False,
        dry_run=False,
    )


    return paths


# ============================================================
# Read HDF5 log
# ============================================================

def read_hdf5_log(
    log_path,
):
    """
    Read an HDF5 log file into a nested dictionary.

    Special convention:
    - normal group attributes are loaded under "attrs"
    - phase-separation method groups are loaded directly by method name

    Example:
        log["metadata"]["attrs"]
        log["metadata"]["phase_separation"]["voxel"]
        log["metadata"]["phase_separation"]["fit"]
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    data = {}

    def clean_attr_value(value):
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

    def attrs_to_dict(group):
        attrs = {}

        for key, value in group.attrs.items():
            attrs[key] = clean_attr_value(value)

        return attrs

    def read_group(group, output):
        # --------------------------------------------------------
        # Read child datasets and groups first
        # --------------------------------------------------------
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                output[key] = item[()]
            elif isinstance(item, h5py.Group):
                output[key] = {}
                read_group(item, output[key])

        # --------------------------------------------------------
        # Read attributes
        # --------------------------------------------------------
        if len(group.attrs) == 0:
            return

        attrs = attrs_to_dict(group)

        # --------------------------------------------------------
        # Backward compatibility:
        #
        # Old files had:
        #     metadata/phase_separation.attrs["density_threshold"]
        #
        # New loaded form should be:
        #     log["metadata"]["phase_separation"]["voxel"]
        # --------------------------------------------------------
        if group.name == "/metadata/phase_separation":
            if "voxel" not in output:
                output["voxel"] = attrs
            return

        # --------------------------------------------------------
        # New method groups:
        #
        #     metadata/phase_separation/voxel.attrs
        #     metadata/classification/phase_separation/voxel.attrs
        #
        # Load method attrs directly into the method dictionary.
        #
        #     log["metadata"]["classification"]["phase_separation"]["voxel"]
        # --------------------------------------------------------
        if (
            group.name.startswith("/metadata/phase_separation/")
            or group.name.startswith(
                "/metadata/classification/phase_separation/"
            )
        ):
            output.update(attrs)
            return

        # --------------------------------------------------------
        # Normal HDF5 behavior everywhere else
        # --------------------------------------------------------
        output["attrs"] = attrs

    with h5py.File(log_path, mode="r") as hdf:
        read_group(hdf, data)

    return data






# ============================================================
# Delete one V2 saved state and log
# ============================================================

def delete_v2_state_files(
    n_fcc_cells,
    target_rho,
    kT,
    nsteps,
    seed=1,
    phase_name="randomization",
    base_folder=THERMALIZED_STATES_V2_ROOT,
    dry_run=True,
    confirm_delete=False,
    delete_empty_folder=True,
    verbose=True,
):
    """
    Delete the saved GSD state file and HDF5 log file for one chosen V2 state.

    This deletes:

        randomization.gsd
        randomization_log.hdf5

    for the state identified by:

        n_fcc_cells
        target_rho
        kT
        nsteps
        seed
        phase_name

    Parameters
    ----------
    dry_run : bool
        If True, only show what would be deleted.

    confirm_delete : bool
        Must be True if dry_run=False.
        This is just a safety catch against accidental deletion.

    delete_empty_folder : bool
        If True, remove the state folder afterward if it is empty.

    Returns
    -------
    report_df : pandas.DataFrame
        Summary of what was deleted or what would be deleted.
    """

    # ============================================================
    # Safety check
    # ============================================================

    if not dry_run and not confirm_delete:
        raise ValueError(
            "This is a destructive operation. "
            "Set confirm_delete=True if you really want to delete files."
        )

    # ============================================================
    # Build expected paths
    # ============================================================

    paths = get_phase_paths(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=nsteps,
        seed=seed,
        phase_name=phase_name,
        base_folder=base_folder,
    )

    state_path = Path(paths["state_path"])
    log_path = Path(paths["log_path"])
    folder = Path(paths["folder"])

    files_to_delete = [
        ("state_path", state_path),
        ("log_path", log_path),
    ]

    rows = []

    # ============================================================
    # Delete files, or show what would be deleted
    # ============================================================

    for file_label, file_path in files_to_delete:
        exists_before = file_path.exists()

        if dry_run:
            if exists_before:
                action = "would_delete"
            else:
                action = "missing"

        else:
            if exists_before:
                file_path.unlink()
                action = "deleted"
            else:
                action = "missing"

        rows.append({
            "file_label": file_label,
            "path": str(file_path),
            "exists_before": bool(exists_before),
            "action": action,
        })

    # ============================================================
    # Optionally remove empty folder
    # ============================================================

    folder_exists_before = folder.exists()
    folder_action = "not_checked"

    if delete_empty_folder:
        if dry_run:
            if folder_exists_before:
                try:
                    folder_is_empty = not any(folder.iterdir())
                except Exception:
                    folder_is_empty = False

                if folder_is_empty:
                    folder_action = "would_remove_empty_folder"
                else:
                    folder_action = "folder_not_empty"
            else:
                folder_action = "folder_missing"

        else:
            if folder_exists_before:
                try:
                    if not any(folder.iterdir()):
                        folder.rmdir()
                        folder_action = "removed_empty_folder"
                    else:
                        folder_action = "folder_not_empty"
                except Exception as error:
                    folder_action = f"failed_to_remove_folder: {repr(error)}"
            else:
                folder_action = "folder_missing"

        rows.append({
            "file_label": "folder",
            "path": str(folder),
            "exists_before": bool(folder_exists_before),
            "action": folder_action,
        })

    report_df = pd.DataFrame(rows)

    # ============================================================
    # Print summary
    # ============================================================

    if verbose:
        print("Delete V2 state files")
        print("=" * 70)
        print("n_fcc_cells =", n_fcc_cells)
        print("target_rho  =", target_rho)
        print("kT          =", kT)
        print("nsteps      =", nsteps)
        print("seed        =", seed)
        print("phase_name  =", phase_name)
        print("dry_run     =", dry_run)
        print("=" * 70)

        for _, row in report_df.iterrows():
            print(row["action"], "|", row["path"])

    return report_df


# ============================================================
# V3 trajectory runs
# ============================================================

def start_gsd_trajectory_writer(
    simulation,
    trajectory_path,
    trajectory_period=1_000,
    mode="wb",
):
    """
    Start a many-frame GSD trajectory writer for evolved runs.
    """

    trajectory_path = Path(trajectory_path)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    writer = hoomd.write.GSD(
        filename=str(trajectory_path),
        trigger=hoomd.trigger.Periodic(int(trajectory_period)),
        mode=mode,
        filter=hoomd.filter.All(),
        dynamic=[
            "property",
            "momentum",
        ],
    )

    simulation.operations.writers.append(writer)

    return {
        "writer": writer,
        "trajectory_path": trajectory_path,
        "trajectory_period": int(trajectory_period),
    }


def stop_gsd_trajectory_writer(simulation, writer_handle):
    writer = writer_handle["writer"]

    if writer in simulation.operations.writers:
        simulation.operations.writers.remove(writer)


def remove_duplicate_initial_trajectory_frame(
    trajectory_path,
):
    """
    Remove a duplicate timestep-0 frame if HOOMD also wrote one.
    """

    trajectory_path = Path(trajectory_path)

    with gsd.hoomd.open(
        name=str(trajectory_path),
        mode="r",
    ) as trajectory:
        if len(trajectory) < 2:
            return False

        first_step = int(trajectory[0].configuration.step)
        second_step = int(trajectory[1].configuration.step)

        if first_step != second_step:
            return False

        tmp_path = trajectory_path.with_suffix(
            trajectory_path.suffix + ".tmp"
        )

        with gsd.hoomd.open(
            name=str(tmp_path),
            mode="w",
        ) as cleaned:
            cleaned.append(trajectory[0])

            for frame_index in range(2, len(trajectory)):
                cleaned.append(trajectory[frame_index])

    tmp_path.replace(trajectory_path)

    return True


def _collect_initial_log_row(
    simulation,
    logger_handle,
):
    """
    Collect the timestep-0 thermodynamic values for evolved-run logs.
    """

    thermo = logger_handle["thermo"]
    tps = getattr(simulation, "tps", 0.0)

    if tps is None:
        tps = 0.0

    initial_row = {
        "hoomd-data/Simulation/timestep": int(simulation.timestep),
        "hoomd-data/Simulation/tps": float(tps),
        (
            "hoomd-data/md/compute/ThermodynamicQuantities/"
            "kinetic_temperature"
        ): float(thermo.kinetic_temperature),
        (
            "hoomd-data/md/compute/ThermodynamicQuantities/"
            "pressure"
        ): float(thermo.pressure),
        (
            "hoomd-data/md/compute/ThermodynamicQuantities/"
            "pressure_tensor"
        ): np.asarray(thermo.pressure_tensor, dtype=float),
        (
            "hoomd-data/md/compute/ThermodynamicQuantities/"
            "potential_energy"
        ): float(thermo.potential_energy),
        (
            "hoomd-data/md/compute/ThermodynamicQuantities/"
            "kinetic_energy"
        ): float(thermo.kinetic_energy),
    }

    return initial_row


def _prepend_value_to_dataset(
    hdf,
    dataset_path,
    value,
):
    """
    Prepend one value to a dataset, creating it if it does not exist.
    """

    parent_path, dataset_name = dataset_path.rsplit("/", 1)
    parent = hdf.require_group(parent_path)

    value = np.asarray(value)

    if dataset_name not in parent:
        if value.shape == ():
            data = value.reshape(1)
        else:
            data = value.reshape((1,) + value.shape)

        parent.create_dataset(dataset_name, data=data)
        return True

    dataset = parent[dataset_name]
    old_data = dataset[()]

    if old_data.shape[0] > 0:
        first_value = old_data[0]

        if np.array_equal(first_value, value):
            return False

    if old_data.ndim == 1:
        new_value = value.reshape(1)
    else:
        new_value = value.reshape((1,) + old_data.shape[1:])

    new_data = np.concatenate(
        [new_value, old_data],
        axis=0,
    )

    del parent[dataset_name]
    parent.create_dataset(dataset_name, data=new_data)

    return True


def prepend_initial_log_row(
    log_path,
    initial_row,
):
    """
    Ensure an HDF5 log starts with the collected initial thermodynamic row.
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    changed = []

    with h5py.File(log_path, mode="a") as hdf:
        timestep_path = "hoomd-data/Simulation/timestep"

        if timestep_path in hdf:
            timestep_data = hdf[timestep_path][()]

            if (
                len(timestep_data) > 0
                and int(timestep_data[0]) == int(initial_row[timestep_path])
            ):
                return changed

        for dataset_path, value in initial_row.items():
            did_change = _prepend_value_to_dataset(
                hdf=hdf,
                dataset_path=dataset_path,
                value=value,
            )

            if did_change:
                changed.append(dataset_path)

    return changed


def run_logged_trajectory_phase(
    simulation,
    nsteps,
    log_path,
    trajectory_path,
    final_state_path=None,
    log_period=1_000,
    trajectory_period=1_000,
    metadata_groups=None,
    classify_final=True,
    classification_kwargs=None,
    include_initial_frame=True,
    include_initial_log=True,
):
    """
    Run any evolved phase with one shared pattern:

    - HDF5 thermodynamic log
    - many-frame GSD trajectory
    - optional one-frame final GSD
    - optional phase classification on the final state

    This is the common runner for future cavitation_evolved and
    excitation_evolved workflows.
    """

    log_path = Path(log_path)
    trajectory_path = Path(trajectory_path)

    if final_state_path is not None:
        final_state_path = Path(final_state_path)

    logger_handle = start_hdf5_logger(
        simulation=simulation,
        log_path=log_path,
        log_period=log_period,
    )

    trajectory_handle = start_gsd_trajectory_writer(
        simulation=simulation,
        trajectory_path=trajectory_path,
        trajectory_period=trajectory_period,
        mode="wb",
    )

    simulation.run(
        0,
        write_at_start=include_initial_frame,
    )

    initial_log_row = None

    if include_initial_log:
        initial_log_row = _collect_initial_log_row(
            simulation=simulation,
            logger_handle=logger_handle,
        )

    simulation.run(int(nsteps))

    stop_gsd_trajectory_writer(
        simulation=simulation,
        writer_handle=trajectory_handle,
    )

    stop_hdf5_logger(
        simulation=simulation,
        logger_objects=logger_handle,
    )

    if include_initial_log and initial_log_row is not None:
        prepend_initial_log_row(
            log_path=log_path,
            initial_row=initial_log_row,
        )

    if final_state_path is not None:
        save_final_state(
            simulation=simulation,
            gsd_path=final_state_path,
        )

    if metadata_groups:
        metadata_helpers.write_metadata_groups(
            hdf5_path=log_path,
            groups=metadata_groups,
            mode="a",
            overwrite=True,
        )

    classification_result = None

    if classify_final and final_state_path is not None:
        classification_kwargs = classification_kwargs or {}
        classification_result = classify_final_state(
            state_path=final_state_path,
            log_path=log_path,
            **classification_kwargs,
        )

    return {
        "log_path": log_path,
        "trajectory_path": trajectory_path,
        "final_state_path": final_state_path,
        "classification_result": classification_result,
    }


def save_last_frame_as_gsd(
    trajectory_path,
    final_state_path,
    overwrite=False,
):
    """
    Save the last frame of a trajectory as a one-frame GSD file.
    """

    trajectory_path = Path(trajectory_path)
    final_state_path = Path(final_state_path)

    if final_state_path.exists() and not overwrite:
        return final_state_path

    final_state_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(trajectory_path), mode="r") as trajectory:
        frame = trajectory[-1]

    with gsd.hoomd.open(name=str(final_state_path), mode="w") as final_file:
        final_file.append(frame)

    return final_state_path
