import time
import numpy as np

import hoomd
import hoomd.md


def make_lattice_snapshot(sim, N, a=1.6):
    """Create a simple-cubic lattice snapshot with no overlaps."""
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]

    if sim.device.communicator.rank == 0:
        n = int(np.ceil(N ** (1 / 3)))  # lattice cells per side
        L = n * a
        snap.configuration.box = [L, L, L, 0, 0, 0]

        xs = (np.arange(n) + 0.5) * a - L / 2
        grid = np.array(np.meshgrid(xs, xs, xs, indexing="ij")).reshape(3, -1).T
        snap.particles.position[:] = grid[:N]

    return snap


def run(device, N=20000, steps=3000):
    sim = hoomd.Simulation(device=device, seed=1)

    # Stable initialization: lattice + gentle thermalization
    snap = make_lattice_snapshot(sim, N=N, a=1.6)
    sim.create_state_from_snapshot(snap)

    # Give small random velocities so it's not a "dead" system
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.1)

    # Lennard-Jones force with a required buffer in your installs
    nl = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=nl)
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0)
    lj.r_cut[("A", "A")] = 2.5

    # Integrator + a method that exists in your HOOMD 5.3.1 builds
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)

    # Your build has ConstantVolume (confirmed by dir(hoomd.md.methods))
    integrator.methods.append(hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All()))
    sim.operations.integrator = integrator

    # Warmup to build neighbor lists / kernels
    sim.run(500)

    t0 = time.time()
    sim.run(steps)
    t1 = time.time()

    return steps / (t1 - t0)


def main():
    print("HOOMD version:", hoomd.version.version)
    print("GPU available:", hoomd.device.GPU.is_available())
    print()

    print("Running CPU benchmark...")
    cpu_tps = run(hoomd.device.CPU())
    print(f"CPU timesteps/s: {cpu_tps:.1f}")

    if hoomd.device.GPU.is_available():
        print("Running GPU benchmark...")
        gpu_tps = run(hoomd.device.GPU())
        print(f"GPU timesteps/s: {gpu_tps:.1f}")
        print(f"Speedup (GPU/CPU): {gpu_tps / cpu_tps:.2f}x")


if __name__ == "__main__":
    main()
