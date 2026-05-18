import time
import numpy as np
import hoomd
import hoomd.md


def pick_method():
    """Pick an integration method that exists in this HOOMD build."""
    m = hoomd.md.methods
    candidates = [
        # common in some builds
        ("ConstantVolume", dict(filter=hoomd.filter.All())),
        ("ConstantPressure", dict(filter=hoomd.filter.All(), S=1.0)),
        ("Langevin", dict(filter=hoomd.filter.All(), kT=1.0)),
        ("Brownian", dict(filter=hoomd.filter.All(), kT=1.0)),
        # legacy names
        ("NVT", dict(filter=hoomd.filter.All(), kT=1.0, tau=1.0)),
        ("NVE", dict(filter=hoomd.filter.All())),
    ]
    for name, kwargs in candidates:
        if hasattr(m, name):
            return getattr(m, name)(**kwargs)

    avail = sorted([x for x in dir(m) if not x.startswith("_")])
    raise RuntimeError(f"No compatible method found. Available hoomd.md.methods: {avail}")


def make_snapshot(sim, N=20000, a=1.6):
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]

    if sim.device.communicator.rank == 0:
        n = int(np.ceil(N ** (1 / 3)))
        L = n * a
        snap.configuration.box = [L, L, L, 0, 0, 0]

        xs = (np.arange(n) + 0.5) * a - L / 2
        grid = np.array(np.meshgrid(xs, xs, xs, indexing="ij")).reshape(3, -1).T
        snap.particles.position[:] = grid[:N]

    return snap


def run(device, N=20000, steps=4000):
    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(make_snapshot(sim, N))

    # gentle starting velocities
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.1)

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=nl)
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    lj.r_cut[("A", "A")] = 2.5

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(pick_method())
    sim.operations.integrator = integrator

    # warmup (build neighbor lists / kernels)
    sim.run(1000)

    t0 = time.time()
    sim.run(steps)
    t1 = time.time()

    return steps / (t1 - t0)


print("HOOMD version:", hoomd.version.version)
print("GPU available:", hoomd.device.GPU.is_available())

# show which method we ended up using
_method = pick_method()
print("Using method:", type(_method).__name__)
print()

cpu_tps = run(hoomd.device.CPU())
print(f"CPU timesteps/s: {cpu_tps:.1f}")

if hoomd.device.GPU.is_available():
    gpu_tps = run(hoomd.device.GPU())
    print(f"GPU timesteps/s: {gpu_tps:.1f}")
    print(f"Speedup (GPU/CPU): {gpu_tps / cpu_tps:.2f}x")
