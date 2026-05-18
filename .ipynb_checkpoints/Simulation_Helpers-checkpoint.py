import hoomd


def make_integrator(
    dt=0.005,
    kT=1.5,
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    buffer_LJ=0.4,
    lj_mode="shift",
    r_on_LJ=2.0,
):
    """
    Create a HOOMD integrator with:
    - LJ pair force
    - Cell neighbor list
    - NVT Bussi thermostat
    """

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

    return integrator


def thermalize_and_randomize(simulation, kT=1.5, nsteps=10_000):
    """
    Thermalize particle momenta and run the simulation
    for nsteps to randomize/equilibrate the initial state.
    """

    simulation.state.thermalize_particle_momenta(
        filter=hoomd.filter.All(),
        kT=kT,
    )

    simulation.run(0)
    simulation.run(nsteps)

    return simulation