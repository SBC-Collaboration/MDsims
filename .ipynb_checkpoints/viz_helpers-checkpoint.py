#viz_helpers.py

import math
import os
import warnings

import fresnel
import IPython
import packaging.version
import numpy as np
import matplotlib.pyplot as plt

device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=300, h=300)

FRESNEL_MIN_VERSION = packaging.version.parse("0.13.0")
FRESNEL_MAX_VERSION = packaging.version.parse("0.14.0")


def render(snapshot):
    if (
        "version" not in dir(fresnel)
        or packaging.version.parse(fresnel.version.version) < FRESNEL_MIN_VERSION
        or packaging.version.parse(fresnel.version.version) >= FRESNEL_MAX_VERSION
    ):
        warnings.warn(
            f"Unsupported fresnel version {fresnel.version.version} - expect errors."
        )
    L = snapshot.configuration.box[0]
    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.Sphere(
        scene, N=len(snapshot.particles.position), radius=0.5
    )
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([252 / 255, 209 / 255, 1 / 255]), roughness=0.5
    )
    geometry.position[:] = snapshot.particles.position[:]
    geometry.outline_width = 0.04
    fresnel.geometry.Box(scene, [L, L, L, 0, 0, 0], box_radius=0.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8), theta=math.pi),
        fresnel.light.Light(
            direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3
        ),
    ]
    scene.camera = fresnel.camera.Orthographic(
        position=(L * 2, L, L * 2), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 1.4 + 1
    )
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)
    samples = 2000
    if "CI" in os.environ:
        samples = 100
    return IPython.display.Image(tracer.sample(scene, samples=samples)._repr_png_())







def plot_voxel_histogram(sim, nbins=10, use_density=False):
    """
    Compute voxel occupancy histogram and plot it with integer-centered bins.

    Parameters:
    - sim: HOOMD simulation
    - nbins: number of bins per dimension (voxel resolution)
    - use_density: if True, plots density instead of raw counts
    """

    # ============================================================
    # Extract positions
    # ============================================================
    snap = sim.state.get_snapshot()
    positions = snap.particles.position

    # Box size (assuming cubic)
    L = sim.state.box.Lx
    bounds = [[-L/2, L/2]] * 3
    voxel_volume = (L / nbins)**3

    # ============================================================
    # Compute voxel histogram
    # ============================================================
    hist, _ = np.histogramdd(
        positions,
        bins=nbins,
        range=bounds
    )



    # Flatten to 1D
    data = hist.ravel()

    # ============================================================
    # Build integer-centered bins
    # ============================================================
    min_val = int(np.floor(data.min()))
    max_val = int(np.ceil(data.max()))

    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    hist2 = np.histogram(data, bins=bins)

    
    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=(6, 4))
    plt.stairs(hist2[0], edges=hist2[1]/voxel_volume, edgecolor='black')

    # plt.xticks(np.arange(min_val, max_val + 1))
    plt.xlabel("Particles per voxel" if not use_density else "Density per voxel")
    plt.ylabel("Count")
    plt.title(f"Voxel distribution (nbins={nbins})")

    plt.show()

    return bins









def plot_xy_particles(sim, point_size=1, alpha=0.7):
    """
    Make a rasterized scatterplot of all particles in the x-y plane.

    Parameters:
    - sim: HOOMD simulation
    - point_size: marker size for each particle
    - alpha: marker transparency
    """

    # ============================================================
    # Extract positions
    # ============================================================
    snap = sim.state.get_snapshot()
    positions = snap.particles.position

    x = positions[:, 0]
    y = positions[:, 1]

    # ============================================================
    # Get box dimensions
    # ============================================================
    box = sim.state.box
    Lx = box.Lx
    Ly = box.Ly

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=(6, 6))

    plt.scatter(
        x,
        y,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    plt.xlim(-Lx / 2, Lx / 2)
    plt.ylim(-Ly / 2, Ly / 2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle positions in x-y plane")

    plt.gca().set_aspect("equal")

    plt.show()





def plot_xy_slice(sim, fraction=0.25, point_size=1, alpha=0.7):
    """
    Plot an x-y slice through the middle of the box.

    Parameters:
    - sim: HOOMD simulation
    - fraction: fraction of box thickness to include in z
                (0 < fraction <= 1)
    - point_size: marker size
    - alpha: transparency
    """

    # ============================================================
    # Validate input
    # ============================================================
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    # ============================================================
    # Extract positions
    # ============================================================
    snap = sim.state.get_snapshot()
    positions = snap.particles.position

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # ============================================================
    # Determine z slice thickness
    # ============================================================
    box = sim.state.box

    Lx = box.Lx
    Ly = box.Ly
    Lz = box.Lz

    half_thickness = 0.5 * fraction * Lz

    # ============================================================
    # Keep only particles near z = 0
    # ============================================================
    mask = np.abs(z) <= half_thickness

    x_slice = x[mask]
    y_slice = y[mask]

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=(6, 6))

    plt.scatter(
        x_slice,
        y_slice,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    plt.xlim(-Lx / 2, Lx / 2)
    plt.ylim(-Ly / 2, Ly / 2)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(f"Middle {100*fraction:.0f}% z-slice")

    plt.gca().set_aspect("equal")

    plt.show()