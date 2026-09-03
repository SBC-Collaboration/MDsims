"""FCC lattice construction shared by every simulation stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FCC_METHOD_VERSION = "fcc_parity_v1"


@dataclass(frozen=True)
class FCCLattice:
    positions: np.ndarray
    n_cells: int
    n_particles: int
    target_density: float
    actual_density: float
    box_length: float
    volume: float


def build_fcc_lattice(n_cells: int, density: float) -> FCCLattice:
    """Build the same parity/checkerboard FCC lattice used in V3."""

    n_cells = int(n_cells)
    density = float(density)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if density <= 0:
        raise ValueError("density must be positive")

    n_particles = 4 * n_cells**3
    volume = n_particles / density
    box_length = volume ** (1.0 / 3.0)
    n_grid = 2 * n_cells

    indices = np.indices((n_grid, n_grid, n_grid)).reshape(3, -1).T
    indices = indices[np.sum(indices, axis=1) % 2 == 0]
    spacing = box_length / n_grid
    positions = indices.astype(np.float64) * spacing - box_length / 2.0

    if len(positions) != n_particles:
        raise RuntimeError(
            f"FCC construction produced {len(positions)} particles; "
            f"expected {n_particles}"
        )

    actual_density = n_particles / box_length**3
    return FCCLattice(
        positions=positions,
        n_cells=n_cells,
        n_particles=n_particles,
        target_density=density,
        actual_density=actual_density,
        box_length=box_length,
        volume=volume,
    )


def make_gsd_frame(lattice: FCCLattice, particle_type: str = "A"):
    """Convert an FCC lattice to a GSD frame without importing GSD globally."""

    import gsd.hoomd

    frame = gsd.hoomd.Frame()
    frame.configuration.step = 0
    frame.configuration.box = [
        lattice.box_length,
        lattice.box_length,
        lattice.box_length,
        0.0,
        0.0,
        0.0,
    ]
    frame.particles.N = lattice.n_particles
    frame.particles.types = [str(particle_type)]
    frame.particles.position = lattice.positions
    frame.particles.typeid = np.zeros(lattice.n_particles, dtype=np.uint32)
    return frame

