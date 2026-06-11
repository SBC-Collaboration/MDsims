#Logging_Helpers.py

from pathlib import Path
import hoomd
import h5py
import gsd.hoomd
import numpy as np

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
    starting_state_path=None,
    phase_separated=None,
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
        "starting_state_path": starting_state_path,
        "phase_separated": phase_separated,
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
    kT,
    nsteps,
    phase_name,
    base_folder=THERMALIZED_STATES_ROOT,
):
    """
    Build standard paths for a logged simulation phase.

    Folder structure:

        base_folder/
            BoxLength_<L>/
                rho_<rho>/
                    kT_<kT>/
                        <phase_name>_nsteps_<nsteps>.gsd
                        <phase_name>_nsteps_<nsteps>_log.hdf5
    """

    BoxLength_str = f"{BoxLength:.1f}"
    rho_str = f"{rho:.2f}"
    kT_str = f"{kT:.2f}"

    folder = (
        Path(base_folder)
        / f"BoxLength_{BoxLength_str}"
        / f"rho_{rho_str}"
        / f"kT_{kT_str}"
    )

    log_path = folder / f"{phase_name}_nsteps_{nsteps}_log.hdf5"
    state_path = folder / f"{phase_name}_nsteps_{nsteps}.gsd"

    return {
        "folder": folder,
        "log_path": log_path,
        "state_path": state_path,
        "phase_name": phase_name,
        "nsteps": nsteps,
        "kT": kT,
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
    starting_state_path="unknown",
    base_folder=THERMALIZED_STATES_ROOT,
):
    """
    Run a simulation phase with HDF5 logging and save the final state.
    """

    paths = get_phase_paths(
        BoxLength=BoxLength,
        rho=rho,
        kT=kT,
        nsteps=nsteps,
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
    
    phase_separated = check_phase_separated(
        simulation=simulation,
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
        starting_state_path=starting_state_path,
        phase_separated=phase_separated,
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





# ============================================================
# Check for phase separation using voxel density
# ============================================================

def check_phase_separated(
    simulation,
    nbins=20,
    density_threshold=0.2,
    voxel_fraction_threshold=0.1,
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
    low_density_fraction = np.mean(voxel_densities < density_threshold)

    phase_separated = low_density_fraction > voxel_fraction_threshold

    return bool(phase_separated)







# ============================================================
# Get mean and std from the last N logged values
# ============================================================

def get_log_tail_stats(
    log_path=None,
    log=None,
    quantity="pressure",
    n_last=100,
    per_particle=False,
    N=None,
    ddof=1,
):
    """
    Compute the mean and standard deviation of the last n_last logged values.

    Important:
    - n_last means the last n_last LOGGER ENTRIES, not MD timesteps.
    - If log_period = 2000 and n_last = 100,
      this uses the last 200,000 MD steps.

    Examples
    --------
    pressure_stats = get_log_tail_stats(
        log_path=log_path,
        quantity="pressure",
        n_last=100,
    )

    pe_stats = get_log_tail_stats(
        log_path=log_path,
        quantity="PE_per_particle",
        n_last=100,
    )

    ke_stats = get_log_tail_stats(
        log_path=log_path,
        quantity="KE_per_particle",
        n_last=100,
    )
    """

    # ============================================================
    # Load log if needed
    # ============================================================
    if log is None:
        if log_path is None:
            raise ValueError("You must provide either log_path or log.")

        log = read_hdf5_log(log_path)

    # ============================================================
    # Quantity aliases
    # ============================================================
    quantity_aliases = {
        # Pressure
        "pressure": ("pressure", False),

        # Potential energy
        "PE": ("potential_energy", False),
        "pe": ("potential_energy", False),
        "potential": ("potential_energy", False),
        "potential_energy": ("potential_energy", False),

        # Potential energy per particle
        "PE_per_particle": ("potential_energy", True),
        "pe_per_particle": ("potential_energy", True),
        "potential_per_particle": ("potential_energy", True),
        "potential_energy_per_particle": ("potential_energy", True),

        # Kinetic energy
        "KE": ("kinetic_energy", False),
        "ke": ("kinetic_energy", False),
        "kinetic": ("kinetic_energy", False),
        "kinetic_energy": ("kinetic_energy", False),

        # Kinetic energy per particle
        "KE_per_particle": ("kinetic_energy", True),
        "ke_per_particle": ("kinetic_energy", True),
        "kinetic_per_particle": ("kinetic_energy", True),
        "kinetic_energy_per_particle": ("kinetic_energy", True),

        # Kinetic temperature, separate from kinetic energy
        "temperature": ("kinetic_temperature", False),
        "kinetic_temperature": ("kinetic_temperature", False),

        # Tensor, if you ever want it later
        "pressure_tensor": ("pressure_tensor", False),
    }

    if quantity not in quantity_aliases:
        raise ValueError(
            "Unknown quantity. Use one of: "
            f"{list(quantity_aliases.keys())}"
        )

    quantity_name, alias_per_particle = quantity_aliases[quantity]

    if alias_per_particle:
        per_particle = True

    # ============================================================
    # Extract timestep and thermodynamic quantity
    # ============================================================
    timestep = log["hoomd-data"]["Simulation"]["timestep"]

    values = (
        log["hoomd-data"]["md"]
           ["compute"]
           ["ThermodynamicQuantities"]
           [quantity_name]
    )

    timestep = np.asarray(timestep)
    values = np.asarray(values)

    # ============================================================
    # Divide by particle number if requested
    # ============================================================
    if per_particle:
        if N is None:
            try:
                N = log["metadata"]["attrs"]["N"]
            except KeyError:
                raise ValueError(
                    "N was not found in the metadata. "
                    "Pass N manually using N=..."
                )

        values = values / N

    # ============================================================
    # Keep only the last n_last logger entries
    # ============================================================
    tail_timesteps = timestep[-n_last:]
    tail_values = values[-n_last:]

    n_used = tail_values.shape[0]

    if n_used == 0:
        raise ValueError("No logged values were found.")

    # ============================================================
    # Compute mean and standard deviation
    # ============================================================
    mean_value = np.mean(tail_values, axis=0)

    if n_used > ddof:
        std_value = np.std(tail_values, axis=0, ddof=ddof)
    else:
        std_value = np.nan

    # ============================================================
    # Clean NumPy outputs
    # ============================================================
    def clean_output(x):
        x = np.asarray(x)

        if x.shape == ():
            return float(x)

        return x.tolist()

    stats = {
        "quantity": quantity_name,
        "per_particle": bool(per_particle),
        "N": int(N) if per_particle else None,
        "n_last_requested": int(n_last),
        "n_values_used": int(n_used),
        "first_timestep_used": int(tail_timesteps[0]),
        "last_timestep_used": int(tail_timesteps[-1]),
        "mean": clean_output(mean_value),
        "std": clean_output(std_value),
    }

    return stats