# Logging_Helpers.py

from pathlib import Path

import hoomd
import h5py
import gsd.hoomd
import numpy as np

from .Project_Paths import THERMALIZED_STATES_V2_ROOT


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
    Write metadata attributes into an existing HDF5 log file.
    """

    # ============================================================
    # Prepare path
    # ============================================================
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    # ============================================================
    # Write metadata
    # ============================================================
    with h5py.File(log_path, mode="a") as hdf:
        metadata_group = hdf.require_group(group_name)

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, Path):
                value = str(value)

            metadata_group.attrs[key] = value

    print("Wrote HDF5 metadata")
    print("Log file:", log_path)
    print("Metadata group:", group_name)


# ============================================================
# Save final simulation state
# ============================================================

def save_final_state(
    simulation,
    gsd_path,
):
    """
    Save the current simulation state as a single-frame GSD file.
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

    V2 convention:
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
        "phase_separated": phase_separated,
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
    base_folder=THERMALIZED_STATES_V2_ROOT,
):
    """
    Build standard V2 paths for a logged simulation phase.

    Folder structure:

        Thermalized_States_v2/
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
    base_folder=THERMALIZED_STATES_V2_ROOT,
):
    """
    Run a simulation phase with HDF5 logging and save the final state.

    V2 path convention:
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
    # Check phase separation
    # ============================================================
    phase_separated = check_phase_separated(
        simulation=simulation,
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
        phase_separated=phase_separated,
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

    return paths


# ============================================================
# Read HDF5 log
# ============================================================

def read_hdf5_log(
    log_path,
):
    """
    Read an HDF5 log file into a nested dictionary.
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    data = {}

    def read_group(group, output):
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                output[key] = item[()]
            elif isinstance(item, h5py.Group):
                output[key] = {}
                read_group(item, output[key])

        if len(group.attrs) > 0:
            attrs = {}

            for key, value in group.attrs.items():
                if hasattr(value, "shape") and value.shape == ():
                    attrs[key] = value.item()
                else:
                    attrs[key] = value

            output["attrs"] = attrs

    with h5py.File(log_path, mode="r") as hdf:
        read_group(hdf, data)

    return data


# ============================================================
# Check for phase separation using voxel density
# ============================================================

def check_phase_separated(
    simulation,
    nbins=20,
    density_threshold=0.2,
    voxel_fraction_threshold=0.05,
):
    """
    Check whether a simulation appears phase separated using voxel densities.

    Default rule:
    - Divide the box into nbins x nbins x nbins voxels.
    - Compute density in each voxel.
    - If more than 5% of voxels have density below 0.2,
      return True.
    - Otherwise return False.
    """

    # ============================================================
    # Extract positions
    # ============================================================
    snapshot = simulation.state.get_snapshot()
    positions = snapshot.particles.position

    # ============================================================
    # Box information
    # ============================================================
    Lx = snapshot.configuration.box[0]
    Ly = snapshot.configuration.box[1]
    Lz = snapshot.configuration.box[2]

    bounds = [
        [-Lx / 2, Lx / 2],
        [-Ly / 2, Ly / 2],
        [-Lz / 2, Lz / 2],
    ]

    voxel_volume = (Lx / nbins) * (Ly / nbins) * (Lz / nbins)

    # ============================================================
    # Count particles in each voxel
    # ============================================================
    voxel_counts, _ = np.histogramdd(
        positions,
        bins=nbins,
        range=bounds,
    )

    voxel_densities = voxel_counts.ravel() / voxel_volume

    # ============================================================
    # Check low-density voxel fraction
    # ============================================================
    low_density_fraction = np.mean(
        voxel_densities < density_threshold
    )

    phase_separated = (
        low_density_fraction > voxel_fraction_threshold
    )

    return bool(phase_separated)