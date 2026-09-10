"""Render one FCC unit cell with a distinct color for each basis particle.

Run from the repository root with::

    python -m examples.render_single_fcc_cell
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from md_Helpers.lattices import build_fcc_lattice


DEFAULT_OUTPUT = Path("Plots/FCC/single_fcc_cell.png")


def basis_indices(positions: np.ndarray, box_length: float) -> np.ndarray:
    """Return the FCC basis index (0-3) for each one-cell lattice position."""

    grid_indices = np.rint((positions + box_length / 2) / (box_length / 2)).astype(int)
    parity = np.mod(grid_indices, 2)
    basis = {(0, 0, 0): 0, (0, 1, 1): 1, (1, 0, 1): 2, (1, 1, 0): 3}
    return np.asarray([basis[tuple(value)] for value in parity], dtype=int)


def render_single_cell(
    output: Path = DEFAULT_OUTPUT,
    density: float = 0.3,
    width: int = 700,
    height: int = 700,
    samples: int = 512,
):
    """Build and render a colored, boxed FCC unit cell to a PNG file."""

    try:
        import fresnel
    except ImportError as error:
        raise ImportError(
            "FCC rendering requires fresnel (install it from conda-forge)."
        ) from error
    import matplotlib.pyplot as plt

    lattice = build_fcc_lattice(n_cells=1, density=density)
    colors = plt.colormaps["turbo"](np.linspace(0.08, 0.92, 4))[:, :3]

    device = fresnel.Device()
    scene = fresnel.Scene(device)
    particles = fresnel.geometry.Sphere(
        scene, N=lattice.n_particles, radius=0.12 * lattice.box_length
    )
    particles.position[:] = lattice.positions
    particles.material = fresnel.material.Material(
        roughness=0.45, primitive_color_mix=1.0
    )
    particles.color[:] = fresnel.color.linear(
        colors[basis_indices(lattice.positions, lattice.box_length)]
    )
    particles.outline_width = 0.03
    box = [lattice.box_length] * 3 + [0, 0, 0]
    fresnel.geometry.Box(scene, box, box_radius=0.012 * lattice.box_length)

    azimuth = math.radians(-55)
    elevation = math.radians(22)
    distance = 2.5 * lattice.box_length
    scene.camera = fresnel.camera.Orthographic(
        position=(
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ),
        look_at=(0, 0, 0),
        up=(0, 0, 1),
        height=1.55 * lattice.box_length,
    )
    scene.lights = fresnel.light.lightbox()
    scene.background_color = (1, 1, 1)
    scene.background_alpha = 1
    image = fresnel.tracer.Path(device=device, w=width, h=height).sample(
        scene, samples=samples
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image._repr_png_())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=512)
    arguments = parser.parse_args()
    print(render_single_cell(arguments.output, arguments.density, samples=arguments.samples))


if __name__ == "__main__":
    main()
