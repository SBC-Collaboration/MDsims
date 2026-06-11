# Simulation_Helpers.py

from pathlib import Path

import hoomd
import gsd.hoomd

from . import Logging_Helpers as lh
from . import Create_Lattices as cl


# ============================================================
# Infer n_fcc_cells from particle count
# ============================================================

def infer_n_fcc_cells_from_N(N):
    """
    Infer n_fcc_cells from N assuming an FCC lattice:

        N = 4 * n_fcc_cells**3

    Returns None if N is not consistent with this form.
    """

    N = int(N)

    n_guess = int(round((N / 4.0) ** (1.0 / 3.0)))

    if 4 * n_guess**3 == N:
        return n_guess

    return None


# ============================================================
# Make HOOMD simulation
# ============================================================

def make_simulation(
    frame,
    target_rho=None,
    n_fcc_cells=None,
    seed=1,
    dt=0.005,
    kT=1.5,
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    buffer_LJ=0.4,
    lj_mode="xplor",
    r_on_LJ=2.0,
    starting_state_path="unknown",
):
    """
    Create a HOOMD simulation from a GSD frame.

    V2 density convention:
    - n_fcc_cells fixes the particle count
    - N = 4 * n_fcc_cells**3
    - target_rho fixes the box size
    - BoxLength is derived, not directly chosen
    """

    # ============================================================
    # Figure out appropriate device
    # ============================================================
    try:
        dev = hoomd.device.GPU()
        simulation = hoomd.Simulation(
            device=dev,
            seed=seed,
        )
        simulation.create_state_from_snapshot(frame)

        print("Using GPU device")

    except Exception as e:
        print("GPU initialization failed:")
        print(e)
        print("Falling back to CPU")

        dev = hoomd.device.CPU()
        simulation = hoomd.Simulation(
            device=dev,
            seed=seed,
        )
        simulation.create_state_from_snapshot(frame)

    print("Final device:", simulation.device)

    # ============================================================
    # Build integrator
    # ============================================================
    integrator = hoomd.md.Integrator(dt=dt)

    cell = hoomd.md.nlist.Cell(buffer=buffer_LJ)

    lj = hoomd.md.pair.LJ(
        nlist=cell,
        mode=lj_mode,
    )

    lj.params[("A", "A")] = dict(
        epsilon=epsilon_LJ,
        sigma=sigma_LJ,
    )

    lj.r_cut[("A", "A")] = r_cut_LJ

    if lj_mode == "xplor":
        lj.r_on[("A", "A")] = r_on_LJ

    integrator.forces.append(lj)

    nvt = hoomd.md.methods.ConstantVolume(
        filter=hoomd.filter.All(),
        thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT),
    )

    integrator.methods.append(nvt)

    simulation.operations.integrator = integrator

    # ============================================================
    # Extract frame information
    # ============================================================
    BoxLength = float(frame.configuration.box[0])
    N = int(frame.particles.N)
    volume = BoxLength**3
    actual_rho = N / volume

    if target_rho is None:
        target_rho = actual_rho

    if n_fcc_cells is None:
        n_fcc_cells = infer_n_fcc_cells_from_N(N)

    # ============================================================
    # Derived FCC metadata
    # ============================================================
    fcc_cell_size = None
    
    if n_fcc_cells is not None:
        n_fcc_cells = int(n_fcc_cells)
        fcc_cell_size = BoxLength / n_fcc_cells

    
    # ============================================================
    # Store metadata on simulation
    # ============================================================
    simulation.metadata = {
        "lattice_type": "fcc",
        "density_mode": "fixed_N_variable_L",

        "n_fcc_cells": n_fcc_cells,

        "N": N,
        "target_rho": target_rho,
        "actual_rho": actual_rho,

        "BoxLength": BoxLength,
        "volume": volume,
        "fcc_cell_size": fcc_cell_size,

        "seed": seed,
        "dt": dt,
        "kT": kT,

        "epsilon_LJ": epsilon_LJ,
        "sigma_LJ": sigma_LJ,
        "r_cut_LJ": r_cut_LJ,
        "buffer_LJ": buffer_LJ,
        "lj_mode": lj_mode,
        "r_on_LJ": r_on_LJ,

        "starting_state_path": starting_state_path,
    }

    return simulation


# ============================================================
# Thermalize and randomize
# ============================================================

def thermalize_and_randomize(
    simulation,
    kT=1.5,
    nsteps=10_000,
    log=False,
    phase_name="randomization",
    log_period=1_000,
):
    """
    Thermalize particle momenta and run the simulation.

    If log=True:
    - run with HDF5 logging
    - save final GSD state
    - write metadata into the HDF5 file

    If log=False:
    - just thermalize and run
    - no files are written
    """

    # ============================================================
    # Make sure metadata exists
    # ============================================================
    if not hasattr(simulation, "metadata"):
        raise ValueError(
            "simulation.metadata was not found. "
            "Create the simulation using sh.make_simulation(frame, ...)."
        )

    metadata = simulation.metadata

    # ============================================================
    # Update kT in metadata if user changes it here
    # ============================================================
    metadata["kT"] = kT

    # ============================================================
    # Thermalize particle momenta before running
    # ============================================================
    simulation.state.thermalize_particle_momenta(
        filter=hoomd.filter.All(),
        kT=kT,
    )

    # ============================================================
    # Run without logging
    # ============================================================
    if not log:
        simulation.run(0)
        simulation.run(nsteps)
        return simulation

    # ============================================================
    # Required V2 metadata
    # ============================================================
    if metadata.get("n_fcc_cells") is None:
        raise ValueError(
            "simulation.metadata['n_fcc_cells'] is missing. "
            "Pass n_fcc_cells to sh.make_simulation(...)."
        )

    # ============================================================
    # Run logged phase
    # ============================================================
    paths = lh.run_logged_phase(
        simulation=simulation,

        n_fcc_cells=metadata["n_fcc_cells"],
        target_rho=metadata["target_rho"],

        phase_name=phase_name,
        nsteps=nsteps,
        log_period=log_period,

        seed=metadata["seed"],
        dt=metadata["dt"],
        kT=kT,

        epsilon_LJ=metadata["epsilon_LJ"],
        sigma_LJ=metadata["sigma_LJ"],
        r_cut_LJ=metadata["r_cut_LJ"],
        r_on_LJ=metadata["r_on_LJ"],
        buffer_LJ=metadata["buffer_LJ"],
        lj_mode=metadata["lj_mode"],

        starting_state_path=metadata["starting_state_path"],
    )

    simulation.logged_paths = paths

    return simulation


# ============================================================
# Get or make thermalized state
# ============================================================

def get_or_make_thermalized_state(
    n_fcc_cells,
    target_rho,
    nsteps,
    kT=1.5,
    phase_name="randomization",
    log_period=1_000,
    seed=1,
    dt=0.005,
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    buffer_LJ=0.4,
    lj_mode="xplor",
    r_on_LJ=2.0,
    overwrite=False,
    overwrite_lattice=False,
):
    """
    Main reusable V2 database function.

    Workflow:
    1. Build expected thermalized-state paths.
    2. If the thermalized state already exists and overwrite=False,
       load and return it.
    3. Otherwise, load/create the FCC lattice.
    4. Create the HOOMD simulation.
    5. Thermalize particle momenta.
    6. Run logged randomization.
    7. Save final randomized state and metadata.

    V2 convention:
    - n_fcc_cells is the chosen system size
    - N = 4 * n_fcc_cells**3
    - target_rho is the requested density
    - BoxLength is derived
    """

    # ============================================================
    # Build expected thermalized-state paths
    # ============================================================
    paths = lh.get_phase_paths(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=nsteps,
        seed=seed,
        phase_name=phase_name,
    )

    state_path = paths["state_path"]
    log_path = paths["log_path"]

    # ============================================================
    # Return existing thermalized state if present
    # ============================================================
    if state_path.exists() and log_path.exists() and not overwrite:
        print("Loaded existing thermalized state:")
        print(state_path)

        with gsd.hoomd.open(
            name=str(state_path),
            mode="r",
        ) as f:
            frame = f[0]

        return {
            "frame": frame,
            "simulation": None,
            "paths": paths,
            "created_new": False,
        }

    # ============================================================
    # Create/load lattice
    # ============================================================
    frame = cl.make_lattice_frame(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        end_print=True,
        overwrite=overwrite_lattice,
    )

    lattice_path = cl.get_lattice_path(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
    )

    # ============================================================
    # Create simulation
    # ============================================================
    simulation = make_simulation(
        frame=frame,
        target_rho=target_rho,
        n_fcc_cells=n_fcc_cells,

        seed=seed,
        dt=dt,
        kT=kT,

        epsilon_LJ=epsilon_LJ,
        sigma_LJ=sigma_LJ,
        r_cut_LJ=r_cut_LJ,
        buffer_LJ=buffer_LJ,
        lj_mode=lj_mode,
        r_on_LJ=r_on_LJ,

        starting_state_path=str(lattice_path),
    )

    # ============================================================
    # Thermalize, randomize, log, and save
    # ============================================================
    simulation = thermalize_and_randomize(
        simulation=simulation,
        kT=kT,
        nsteps=nsteps,
        log=True,
        phase_name=phase_name,
        log_period=log_period,
    )

    # ============================================================
    # Load final randomized frame from saved GSD
    # ============================================================
    with gsd.hoomd.open(
        name=str(simulation.logged_paths["state_path"]),
        mode="r",
    ) as f:
        final_frame = f[0]

    return {
        "frame": final_frame,
        "simulation": simulation,
        "paths": simulation.logged_paths,
        "created_new": True,
    }