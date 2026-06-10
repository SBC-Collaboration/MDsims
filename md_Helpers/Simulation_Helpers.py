#Simulation_Helpers.py

import hoomd
from . import Logging_Helpers as lh


def make_simulation(
    frame,
    seed=1,
    dt=0.005,
    kT=1.5,
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    buffer_LJ=0.4,
    lj_mode="xplor",
    r_on_LJ=2.0,
    starting_state_path=None,
):
    """
    Create a HOOMD simulation that:
    - tries GPU first, then falls back to CPU
    - initializes from a frame
    - attaches integrator + LJ + thermostat
    """

    # ============================================================
    # Figure out appropriate device
    # ============================================================
    try:
        dev = hoomd.device.GPU()
        simulation = hoomd.Simulation(device=dev, seed=seed)
        simulation.create_state_from_snapshot(frame)

    except Exception as e:
        print("GPU initialization failed:")
        print(e)
        print("Falling back to CPU")

        dev = hoomd.device.CPU()
        simulation = hoomd.Simulation(device=dev, seed=seed)
        simulation.create_state_from_snapshot(frame)

    print("Starting Simulation --------------- Final device:", simulation.device)

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
    simulation.metadata = {
        "BoxLength": frame.configuration.box[0],
        "rho": frame.particles.N / frame.configuration.box[0]**3,
        "N": frame.particles.N,
        "seed": seed,
        "dt": dt,
        "kT": kT,
        "epsilon_LJ": epsilon_LJ,
        "sigma_LJ": sigma_LJ,
        "r_cut_LJ": r_cut_LJ,
        "buffer_LJ": buffer_LJ,
        "lj_mode": lj_mode,
        "r_on_LJ": r_on_LJ,
        "starting_state_path": str(starting_state_path) if starting_state_path is not None else "",
    }

    return simulation


def thermalize_and_randomize(
    simulation,
    nsteps=10_000,
    log=False,
    phase_name="randomization",
    log_period=1_000,
):
    """
    Thermalize momenta and run simulation for nsteps.

    Temperature is taken from simulation.metadata["kT"], which is set in
    make_simulation(). This avoids accidentally using one kT for the
    thermostat and another kT for the velocity draw.
    """

    if not hasattr(simulation, "metadata"):
        raise ValueError(
            "simulation.metadata was not found. "
            "Create the simulation using sh.make_simulation(frame)."
        )

    metadata = simulation.metadata
    kT = metadata["kT"]

    # ============================================================
    # Thermalize particle momenta using simulation kT
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
    # Run with logging and save final state
    # ============================================================
    paths = lh.run_logged_phase(
        simulation=simulation,
        BoxLength=metadata["BoxLength"],
        rho=metadata["rho"],
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
    )

    simulation.logged_paths = paths

    return simulation