# viz_helpers.py

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


# ============================================================
# Convert input object to snapshot/frame-like object
# ============================================================

def _get_snapshot(obj):
    """
    Return a snapshot/frame-like object from any supported input.

    Supported inputs:
    - HOOMD simulation: sim
    - HOOMD state: sim.state
    - HOOMD snapshot: sim.state.get_snapshot()
    - GSD frame: frame

    The returned object must have:
    - obj.configuration.box
    - obj.particles.position
    """

    # ============================================================
    # HOOMD Simulation
    # ============================================================
    if hasattr(obj, "state") and hasattr(obj.state, "get_snapshot"):
        return obj.state.get_snapshot()

    # ============================================================
    # HOOMD State
    # ============================================================
    if hasattr(obj, "get_snapshot"):
        return obj.get_snapshot()

    # ============================================================
    # GSD frame or HOOMD snapshot
    # ============================================================
    if hasattr(obj, "configuration") and hasattr(obj, "particles"):
        return obj

    raise TypeError(
        "Input must be a HOOMD simulation, HOOMD state, "
        "HOOMD snapshot, or GSD frame."
    )


# ============================================================
# Extract positions and box lengths
# ============================================================

def _get_positions_and_box(obj):
    """
    Extract particle positions and box lengths from any supported object.
    """

    snapshot = _get_snapshot(obj)

    positions = np.asarray(
        snapshot.particles.position,
        dtype=np.float64,
    )

    box = np.asarray(
        snapshot.configuration.box,
        dtype=np.float64,
    )

    Lx = float(box[0])
    Ly = float(box[1])
    Lz = float(box[2])

    return positions, Lx, Ly, Lz, snapshot


def render(obj):
    """
    Render a GSD frame, HOOMD snapshot, or HOOMD simulation.

    This keeps the original V1 render settings as closely as possible.
    """

    # ============================================================
    # Accept simulation, state, frame, or snapshot
    # ============================================================
    if hasattr(obj, "state") and hasattr(obj.state, "get_snapshot"):
        snapshot = obj.state.get_snapshot()
    elif hasattr(obj, "get_snapshot"):
        snapshot = obj.get_snapshot()
    else:
        snapshot = obj

    # ============================================================
    # Original V1 render code
    # ============================================================
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
        scene,
        N=len(snapshot.particles.position),
        radius=0.5,
    )

    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([252 / 255, 209 / 255, 1 / 255]),
        roughness=0.5,
    )

    geometry.position[:] = snapshot.particles.position[:]
    geometry.outline_width = 0.04

    fresnel.geometry.Box(
        scene,
        [L, L, L, 0, 0, 0],
        box_radius=0.02,
    )

    scene.lights = [
        fresnel.light.Light(
            direction=(0, 0, 1),
            color=(0.8, 0.8, 0.8),
            theta=math.pi,
        ),
        fresnel.light.Light(
            direction=(1, 1, 1),
            color=(1.1, 1.1, 1.1),
            theta=math.pi / 3,
        ),
    ]

    scene.camera = fresnel.camera.Orthographic(
        position=(L * 2, L, L * 2),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
        height=L * 1.4 + 1,
    )

    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)

    samples = 2000

    if "CI" in os.environ:
        samples = 100

    return IPython.display.Image(
        tracer.sample(scene, samples=samples)._repr_png_()
    )


# ============================================================
# Compute voxel densities
# ============================================================

def compute_voxel_densities(
    obj,
    nbins=10,
):
    """
    Compute voxel densities for a simulation, snapshot, or frame.

    Returns
    -------
    voxel_densities : np.ndarray
        Flattened array of voxel densities.

    voxel_counts : np.ndarray
        Flattened array of particle counts per voxel.

    voxel_volume : float
        Volume of one voxel.
    """

    # ============================================================
    # Extract positions and box
    # ============================================================
    positions, Lx, Ly, Lz, snapshot = _get_positions_and_box(obj)

    bounds = [
        [-Lx / 2, Lx / 2],
        [-Ly / 2, Ly / 2],
        [-Lz / 2, Lz / 2],
    ]

    voxel_volume = (Lx / nbins) * (Ly / nbins) * (Lz / nbins)

    # ============================================================
    # Count particles in each voxel
    # ============================================================
    voxel_counts, _ = np.histogramdd(
        positions,
        bins=nbins,
        range=bounds,
    )

    voxel_counts = voxel_counts.ravel()

    voxel_densities = voxel_counts / voxel_volume

    return voxel_densities, voxel_counts, voxel_volume


# ============================================================
# Plot voxel density histogram
# ============================================================

def plot_voxel_histogram(
    obj,
    nbins=10,
):
    """
    Plot voxel density histogram.

    The x-axis is always voxel density:

        voxel_density = particles_in_voxel / voxel_volume

    Input can be:
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame

    Returns
    -------
    hist2 : tuple
        hist2[0] = number of voxels in each density bin
        hist2[1] = density bin edges
    """

    # ============================================================
    # Compute voxel densities
    # ============================================================
    voxel_densities, voxel_counts, voxel_volume = compute_voxel_densities(
        obj,
        nbins=nbins,
    )

    # ============================================================
    # Build density-bin edges from integer particle-count bins
    # ============================================================
    min_count = int(np.floor(voxel_counts.min()))
    max_count = int(np.ceil(voxel_counts.max()))

    count_edges = np.arange(
        min_count - 0.5,
        max_count + 1.5,
        1,
    )

    density_edges = count_edges / voxel_volume

    hist_y, hist_x_edges = np.histogram(
        voxel_densities,
        bins=density_edges,
    )

    hist2 = (hist_y, hist_x_edges)

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=(6, 4))

    plt.stairs(
        hist2[0],
        edges=hist2[1],
        edgecolor="black",
    )

    plt.xlabel("Voxel density")
    plt.ylabel("Number of voxels")
    plt.title(f"Voxel density distribution (nbins={nbins})")

    plt.show()

    return hist2


# ============================================================
# Plot all particles in x-y plane
# ============================================================

def plot_xy_particles(
    obj,
    point_size=1,
    alpha=0.7,
):
    """
    Make a rasterized scatterplot of all particles in the x-y plane.

    Input can be:
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame
    """

    # ============================================================
    # Extract positions and box
    # ============================================================
    positions, Lx, Ly, Lz, snapshot = _get_positions_and_box(obj)

    x = positions[:, 0]
    y = positions[:, 1]

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


# ============================================================
# Plot x-y slice
# ============================================================

def plot_xy_slice(
    obj,
    fraction=0.25,
    point_size=1,
    alpha=0.7,
):
    """
    Plot an x-y slice through the middle of the box.

    Input can be:
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame

    Parameters
    ----------
    fraction : float
        Fraction of box thickness to include in z.
        Must satisfy 0 < fraction <= 1.
    """

    # ============================================================
    # Validate input
    # ============================================================
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    # ============================================================
    # Extract positions and box
    # ============================================================
    positions, Lx, Ly, Lz, snapshot = _get_positions_and_box(obj)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

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

    plt.title(f"Middle {100 * fraction:.0f}% z-slice")

    plt.gca().set_aspect("equal")

    plt.show()


# ============================================================
# Plot pressure log
# ============================================================

def plot_pressure_log(
    log,
    figsize=(8, 5),
):
    """
    Plot pressure vs timestep from a log dictionary returned by
    Logging_Helpers.read_hdf5_log().
    """

    # ============================================================
    # Extract data
    # ============================================================
    timestep = log["hoomd-data"]["Simulation"]["timestep"]

    pressure = (
        log["hoomd-data"]["md"]
           ["compute"]
           ["ThermodynamicQuantities"]
           ["pressure"]
    )

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=figsize)

    plt.plot(timestep, pressure)

    plt.xlabel("Timestep")
    plt.ylabel("Pressure")
    plt.title("Pressure vs Timestep")

    plt.grid(alpha=0.3)

    plt.show()


# ============================================================
# Plot logged thermodynamic quantity
# ============================================================

def plot_log_quantity(
    log,
    quantity,
    figsize=(8, 5),
):
    """
    Plot a ThermodynamicQuantities quantity against timestep.

    Examples
    --------
    vh.plot_log_quantity(log, "pressure")

    vh.plot_log_quantity(log, "kinetic_temperature")

    vh.plot_log_quantity(log, "potential_energy")

    vh.plot_log_quantity(log, "kinetic_energy")
    """

    # ============================================================
    # Extract data
    # ============================================================
    timestep = log["hoomd-data"]["Simulation"]["timestep"]

    values = (
        log["hoomd-data"]["md"]
           ["compute"]
           ["ThermodynamicQuantities"]
           [quantity]
    )

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=figsize)

    plt.plot(timestep, values)

    plt.xlabel("Timestep")
    plt.ylabel(quantity.replace("_", " ").title())
    plt.title(quantity.replace("_", " ").title())

    plt.grid(alpha=0.3)

    plt.show()