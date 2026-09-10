"""Plot HOOMD's Lennard-Jones cutoff treatments.

Run from the repository root with::

    python -m examples.plot_lj_modes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_DIRECTORY = Path("Plots/HOOMD/PairPotentials")


def _snapshot(hoomd, separation: float, box_length: float):
    snapshot = hoomd.Snapshot()
    snapshot.particles.N = 2
    snapshot.particles.types = ["A"]
    snapshot.particles.typeid[:] = [0, 0]
    snapshot.particles.position[:] = [
        [-separation / 2, 0, 0],
        [separation / 2, 0, 0],
    ]
    snapshot.configuration.box = [box_length] * 3 + [0, 0, 0]
    return snapshot


def measure_pair_interaction(pair_force, separations, box_length: float = 30.0):
    """Measure potential and signed radial force for a two-particle system."""

    import hoomd

    potentials = []
    radial_forces = []
    for separation in separations:
        simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
        simulation.create_state_from_snapshot(
            _snapshot(hoomd, float(separation), box_length)
        )
        integrator = hoomd.md.Integrator(dt=0.005, forces=[pair_force])
        simulation.operations.integrator = integrator
        thermodynamics = hoomd.md.compute.ThermodynamicQuantities(
            filter=hoomd.filter.All()
        )
        simulation.operations.computes.append(thermodynamics)
        simulation.run(0)
        potentials.append(float(thermodynamics.potential_energy))
        # Particle zero is left of the origin, so its outward direction is -x.
        radial_forces.append(-float(pair_force.forces[0][0]))
    return np.asarray(potentials), np.asarray(radial_forces)


def make_pair_forces(r_cut: float, r_on: float, buffer: float = 0.4):
    """Create the four HOOMD LJ variants used in the comparison."""

    import hoomd

    forces = {}
    for mode in ("none", "shift", "xplor"):
        force = hoomd.md.pair.LJ(
            nlist=hoomd.md.nlist.Cell(buffer=buffer),
            default_r_cut=r_cut,
            mode=mode,
        )
        force.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
        if mode == "xplor":
            force.r_on[("A", "A")] = r_on
        forces[f"LJ ({mode})"] = force

    force_shifted = hoomd.md.pair.ForceShiftedLJ(
        nlist=hoomd.md.nlist.Cell(buffer=buffer),
        default_r_cut=r_cut,
    )
    force_shifted.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
    forces["ForceShiftedLJ"] = force_shifted
    return forces


def create_plots(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    r_cut: float = 1.5,
    r_on: float = 1.35,
    samples: int = 700,
):
    """Measure all four cutoff modes and save potential, force, and combined plots."""

    import matplotlib.pyplot as plt

    if not 0 < r_on < r_cut:
        raise ValueError("r_on must satisfy 0 < r_on < r_cut")
    if samples < 2:
        raise ValueError("samples must be at least 2")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    separations = np.linspace(0.85, r_cut + 0.1, samples)
    results = {
        label: measure_pair_interaction(force, separations)
        for label, force in make_pair_forces(r_cut, r_on).items()
    }

    def decorate(axis, ylabel: str, ylim):
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.axvline(r_cut, color="red", linestyle="--", linewidth=1,
                     label=rf"$r_{{cut}}={r_cut}$")
        axis.axvline(r_on, color="gray", linestyle=":", linewidth=1,
                     label=rf"$r_{{on}}={r_on}$")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.grid(alpha=0.3)
        axis.legend()

    outputs = []
    for index, (name, ylabel, ylim, filename) in enumerate([
        ("potential", r"$U(r)$", (-1.25, 2.0), "lj_cutoff_mode_comparison.png"),
        ("force", r"$F(r)=-dU/dr$", (-6, 20), "lj_cutoff_force_comparison.png"),
    ]):
        figure, axis = plt.subplots(figsize=(9, 6))
        for label, values in results.items():
            axis.plot(separations, values[index], label=label)
        decorate(axis, ylabel, ylim)
        axis.set_xlabel(r"$r/\sigma$")
        axis.set_xlim(separations.min(), separations.max())
        axis.set_title(f"HOOMD Lennard-Jones cutoff {name}s")
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        outputs.append(path)

    figure, (potential_axis, force_axis) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True
    )
    for label, (potential, force) in results.items():
        potential_axis.plot(separations, potential, label=label)
        force_axis.plot(separations, force, label=label)
    decorate(potential_axis, r"$U(r)$", (-1.25, 2.0))
    decorate(force_axis, r"$F(r)=-dU/dr$", (-6, 15))
    potential_axis.set_title("HOOMD Lennard-Jones cutoff treatments")
    force_axis.set_xlabel(r"$r/\sigma$")
    force_axis.set_xlim(separations.min(), separations.max())
    figure.tight_layout()
    path = output_directory / "lj_cutoff_potential_force_comparison.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--r-cut", type=float, default=1.5)
    parser.add_argument("--r-on", type=float, default=1.35)
    parser.add_argument("--samples", type=int, default=700)
    arguments = parser.parse_args()
    for output in create_plots(
        arguments.output_directory,
        arguments.r_cut,
        arguments.r_on,
        arguments.samples,
    ):
        print(output)


if __name__ == "__main__":
    main()
