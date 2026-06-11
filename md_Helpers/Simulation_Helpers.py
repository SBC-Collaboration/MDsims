#Simulation_Helpers.py

from pathlib import Path
import hoomd
import gsd.hoomd

from . import Logging_Helpers as lh
from . import Create_Lattices as cl


def make_simulation(
    frame,
    target_rho=None,
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

    Important density convention:
    - target_rho is the density you asked for
    - actual_rho is N / V after integer lattice construction
    """

    # ============================================================
    # Figure out appropriate device
    # ============================================================
    try:
        dev = hoomd.device.GPU()
        simulation = hoomd.Simulation(device=dev, seed=seed)
        simulation.create_state_from_snapshot(frame)

        print("Using GPU device")

    except Exception as e:
        print("GPU initialization failed:")
        print(e)
        print("Falling back to CPU")

        dev = hoomd.device.CPU()
        simulation = hoomd.Simulation(device=dev, seed=seed)
        simulation.create_state_from_snapshot(frame)

    print("Final device:", simulation.device)

    # ============================================================
    # Build integrator
    # ============================================================
    integrator = hoomd.md.Integrator(dt=dt)

    cell = hoomd.md.nlist.Cell(buffer=buffer_LJ)

    lj = hoomd.md.pair.LJ(nlist=cell, mode=lj_mode)
    lj.params[("A", "A")] = dict(epsilon=epsilon_LJ, sigma=sigma_LJ)
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
    # Store metadata on simulation
    # ============================================================
    BoxLength = frame.configuration.box[0]
    N = frame.particles.N
    actual_rho = N / BoxLength**3

    if target_rho is None:
        target_rho = actual_rho

    simulation.metadata = {
        "BoxLength": BoxLength,
        "target_rho": target_rho,
        "actual_rho": actual_rho,
        "N": N,
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


def thermalize_and_randomize(
    simulation,
    kT=1.5,
    nsteps=10_000,
    log=False,
    phase_name="randomization",
    log_period=1_000,
):
    """
    Thermalize momenta and run simulation for nsteps.
    Optionally log and save the final state.
    """

    simulation.state.thermalize_particle_momenta(
        filter=hoomd.filter.All(),
        kT=kT,
    )

    if not log:
        simulation.run(0)
        simulation.run(nsteps)
        return simulation

    if not hasattr(simulation, "metadata"):
        raise ValueError(
            "simulation.metadata was not found. "
            "Create the simulation using sh.make_simulation(frame)."
        )

    metadata = simulation.metadata

    paths = lh.run_logged_phase(
        simulation=simulation,
        BoxLength=metadata["BoxLength"],
        rho=metadata["target_rho"],
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
        lj_mode=metadata["lj_mode"],
        starting_state_path=metadata["starting_state_path"],
    )

    simulation.logged_paths = paths

    return simulation


def get_or_make_thermalized_state(
    BoxLength,
    target_rho,
    nsteps,
    kT=1.5,
    lattice_type="fcc",
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
):
    """
    Main reusable database function.

    Workflow:
    1. Check whether thermalized state already exists.
    2. If yes, load and return it.
    3. If no, check/load/create the lattice.
    4. Make the simulation.
    5. Thermalize and save the logged state.

    Density convention:
    - target_rho: density requested by the user
    - actual_rho: N / V after lattice particle count is chosen
    """

    # ============================================================
    # Build expected thermalized-state paths using target rho
    # ============================================================
    paths = lh.get_phase_paths(
        BoxLength=BoxLength,
        rho=target_rho,
        kT=kT,
        nsteps=nsteps,
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

        with gsd.hoomd.open(name=str(state_path), mode="r") as f:
            frame = f[0]

        return {
            "frame": frame,
            "simulation": None,
            "paths": paths,
            "created_new": False,
        }

    # ============================================================
    # Create/load lattice using target rho
    # ============================================================
    frame, lattice_path = cl.make_lattice_frame(
        BoxLength=BoxLength,
        rho=target_rho,
        lattice_type=lattice_type,
        end_print=True,
        return_path=True,
    )

    # ============================================================
    # Create simulation, preserving target_rho separately
    # ============================================================
    simulation = make_simulation(
        frame=frame,
        target_rho=target_rho,
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
    # Thermalize and save
    # ============================================================
    simulation = thermalize_and_randomize(
        simulation=simulation,
        kT=kT,
        nsteps=nsteps,
        log=True,
        phase_name=phase_name,
        log_period=log_period,
    )

    return {
        "frame": frame,
        "simulation": simulation,
        "paths": simulation.logged_paths,
        "created_new": True,
    }