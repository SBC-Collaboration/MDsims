# runs.py

import math
from pathlib import Path

import hoomd
import h5py
import gsd.hoomd

from . import classification as ps
from . import metadata as metadata_helpers
from .classification import classify_final_state
from .paths import THERMALIZED_STATES_V3_ROOT, thermalized_run_paths

# ============================================================
# Default phase-separation settings
# ============================================================

DEFAULT_PHASE_SEP_NBINS = ps.DEFAULT_PHASE_SEP_NBINS
DEFAULT_PHASE_SEP_DENSITY_THRESHOLD = ps.DEFAULT_PHASE_SEP_DENSITY_THRESHOLD
DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD = ps.DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD


class NPHVolumeSafetyStop(RuntimeError):
    """Raised when an NPH box leaves its configured safe volume interval."""

    def __init__(
        self,
        *,
        reason,
        volume_ratio,
        timestep,
        lower_ratio,
        upper_ratio,
    ):
        self.reason = str(reason)
        self.volume_ratio = float(volume_ratio)
        self.timestep = int(timestep)
        self.lower_ratio = float(lower_ratio)
        self.upper_ratio = float(upper_ratio)
        super().__init__(
            "NPH safety stop before excessive memory growth: "
            f"reason={self.reason}, box volume ratio={self.volume_ratio:.6g} "
            f"at timestep {self.timestep}; allowed interval is "
            f"[{self.lower_ratio:.6g}, {self.upper_ratio:.6g}]. "
            "Increase tauS or inspect the pressure response before continuing."
        )
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

    outer_thermo = None
    outer_filter = getattr(simulation, "nph_outer_filter", None)
    if outer_filter is not None:
        outer_thermo = hoomd.md.compute.ThermodynamicQuantities(
            filter=outer_filter
        )
        simulation.operations.computes.append(outer_thermo)

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
            "volume",
        ],
    )

    if outer_thermo is not None:
        logger.add(
            outer_thermo,
            quantities=[
                "kinetic_temperature",
                "pressure",
                "pressure_tensor",
                "potential_energy",
                "kinetic_energy",
                "num_particles",
                "volume",
            ],
            user_name="OuterRegionThermodynamicQuantities",
        )

    for method in simulation.operations.integrator.methods:
        if type(method).__name__ == "ConstantPressure":
            logger.add(
                method,
                quantities=["barostat_energy"],
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
        "outer_thermo": outer_thermo,
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
    outer_thermo = logger_objects.get("outer_thermo")
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
    if (
        outer_thermo is not None
        and outer_thermo in simulation.operations.computes
    ):
        simulation.operations.computes.remove(outer_thermo)

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
    paths = thermalized_run_paths(
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

    Group attributes are returned under each group's "attrs" key.
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

        output["attrs"] = attrs

    with h5py.File(log_path, mode="r") as hdf:
        read_group(hdf, data)

    return data






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
    include_initial=True,
    box_volume_ratio_bounds=None,
    safety_check_period=100,
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
    trajectory_handle = None

    try:
        trajectory_handle = start_gsd_trajectory_writer(
            simulation=simulation,
            trajectory_path=trajectory_path,
            trajectory_period=trajectory_period,
            mode="wb",
        )
        simulation.run(0, write_at_start=include_initial)
        if box_volume_ratio_bounds is None:
            simulation.run(int(nsteps))
        else:
            lower_ratio, upper_ratio = map(float, box_volume_ratio_bounds)
            if not 0.0 < lower_ratio < 1.0 < upper_ratio:
                raise ValueError(
                    "box_volume_ratio_bounds must bracket 1.0"
                )
            check_period = int(safety_check_period)
            if check_period <= 0:
                raise ValueError("safety_check_period must be positive")

            initial_volume = float(simulation.state.box.volume)
            remaining = int(nsteps)
            while remaining:
                chunk = min(check_period, remaining)
                simulation.run(chunk)
                remaining -= chunk
                current_volume = float(simulation.state.box.volume)
                volume_ratio = current_volume / initial_volume
                if not math.isfinite(volume_ratio):
                    stop_reason = "nonfinite_volume"
                elif volume_ratio < lower_ratio:
                    stop_reason = "lower_volume_limit"
                elif volume_ratio > upper_ratio:
                    stop_reason = "upper_volume_limit"
                else:
                    stop_reason = None
                if stop_reason is not None:
                    raise NPHVolumeSafetyStop(
                        reason=stop_reason,
                        volume_ratio=volume_ratio,
                        timestep=simulation.timestep,
                        lower_ratio=lower_ratio,
                        upper_ratio=upper_ratio,
                    )
    finally:
        if trajectory_handle is not None:
            stop_gsd_trajectory_writer(
                simulation=simulation,
                writer_handle=trajectory_handle,
            )
        stop_hdf5_logger(
            simulation=simulation,
            logger_objects=logger_handle,
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
