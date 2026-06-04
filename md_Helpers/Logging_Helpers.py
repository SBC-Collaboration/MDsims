#Logging_Helpers.py

from pathlib import Path
import hoomd
import h5py
import gsd.hoomd

from .Project_Paths import PROJECT_ROOT, SIMPLE_LATTICES_ROOT, THERMALIZED_STATES_ROOT






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

    This function:
    - creates a ThermodynamicQuantities compute
    - creates a HOOMD logger
    - logs timestep, TPS, temperature, pressure, pressure tensor, and energies
    - attaches an HDF5 writer to simulation.operations.writers

    It does NOT:
    - run the simulation
    - check whether files already exist
    - save the final state
    - write metadata

    Parameters
    ----------
    simulation : hoomd.Simulation
        Existing HOOMD simulation.

    log_path : str or pathlib.Path
        Path to the HDF5 log file.

    log_period : int
        Number of timesteps between log entries.

    Returns
    -------
    logger_objects : dict
        Dictionary containing the thermo compute, logger, writer,
        log path, and log period.
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

    This function:
    - removes the HDF5 writer from simulation.operations.writers
    - removes the thermo compute from simulation.operations.computes

    It does NOT:
    - save the final state
    - write metadata
    - delete or modify the HDF5 file
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
    print("Log file:", logger_objects["log_path"])






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

    This function:
    - opens the existing HDF5 file
    - creates a metadata group if needed
    - stores simple metadata values as HDF5 attributes

    It does NOT:
    - run the simulation
    - start or stop logging
    - save the final state
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
    target_rho=None,
    seed=None,
    dt=None,
    kT=None,
    epsilon_LJ=None,
    sigma_LJ=None,
    r_cut_LJ=None,
    r_on_LJ=None,
    lj_mode=None,
    log_period=None,
    nsteps=None,
):
    """
    Build a metadata dictionary for the current simulation state.

    This function:
    - computes actual density from the current simulation state
    - stores box size, particle count, phase name, and run parameters

    It does NOT:
    - write metadata to file
    - start or stop logging
    - save the final state
    """

    # ============================================================
    # Extract current simulation state information
    # ============================================================
    snapshot = simulation.state.get_snapshot()

    N = snapshot.particles.N
    Lx = snapshot.configuration.box[0]
    Ly = snapshot.configuration.box[1]
    Lz = snapshot.configuration.box[2]

    volume = Lx * Ly * Lz
    actual_rho = N / volume

    # ============================================================
    # Build metadata dictionary
    # ============================================================
    metadata = {
        "phase_name": phase_name,
        "N": N,
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz,
        "volume": volume,
        "actual_rho": actual_rho,
        "final_timestep": simulation.timestep,
    }

    # ============================================================
    # Optional user-provided metadata
    # ============================================================
    optional_metadata = {
        "target_rho": target_rho,
        "seed": seed,
        "dt": dt,
        "kT": kT,
        "epsilon_LJ": epsilon_LJ,
        "sigma_LJ": sigma_LJ,
        "r_cut_LJ": r_cut_LJ,
        "r_on_LJ": r_on_LJ,
        "lj_mode": lj_mode,
        "log_period": log_period,
        "nsteps": nsteps,
    }

    for key, value in optional_metadata.items():
        if value is not None:
            metadata[key] = value

    return metadata






# ============================================================
# Build phase file paths
# ============================================================

def get_phase_paths(
    BoxLength,
    rho,
    phase_name,
    base_folder=THERMALIZED_STATES_ROOT,
):
    """
    Build standard paths for a logged simulation phase.
    """

    BoxLength_str = f"{BoxLength:.1f}"
    rho_str = f"{rho:.2f}"

    folder = (
        Path(base_folder)
        / f"BoxLength_{BoxLength_str}"
        / f"rho_{rho_str}"
    )

    log_path = folder / f"{phase_name}_log.hdf5"
    state_path = folder / f"{phase_name}_final_state.gsd"

    return {
        "folder": folder,
        "log_path": log_path,
        "state_path": state_path,
        "phase_name": phase_name,
    }






# ============================================================
# Run logged phase
# ============================================================

def run_logged_phase(
    simulation,
    BoxLength,
    rho,
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
    lj_mode=None,
    base_folder=THERMALIZED_STATES_ROOT,
):
    """
    Run a simulation phase with HDF5 logging and save the final state.
    """

    paths = get_phase_paths(
        BoxLength=BoxLength,
        rho=rho,
        phase_name=phase_name,
        base_folder=base_folder,
    )

    logger_handle = start_hdf5_logger(
        simulation=simulation,
        log_path=paths["log_path"],
        log_period=log_period,
    )
    
    simulation.run(0)
    simulation.run(nsteps)

    stop_hdf5_logger(
        simulation=simulation,
        logger_objects=logger_handle,
    )

    metadata = build_simulation_metadata(
        simulation=simulation,
        phase_name=phase_name,
        target_rho=rho,
        seed=seed,
        dt=dt,
        kT=kT,
        epsilon_LJ=epsilon_LJ,
        sigma_LJ=sigma_LJ,
        r_cut_LJ=r_cut_LJ,
        r_on_LJ=r_on_LJ,
        lj_mode=lj_mode,
        log_period=log_period,
        nsteps=nsteps,
    )

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

    This function is meant to load logged data so plotting helpers
    can use it later.
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