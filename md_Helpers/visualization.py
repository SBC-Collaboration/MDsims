# visualization.py

import math
import os
import warnings

import fresnel
import IPython
import packaging.version
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=300, h=300)

FRESNEL_MIN_VERSION = packaging.version.parse("0.13.0")
FRESNEL_MAX_VERSION = packaging.version.parse("0.14.0")


# ============================================================
# Convert input to snapshot/frame
# ============================================================

def _as_snapshot(obj):
    """
    Accept:
    - result dictionary from sh.get_or_make_thermalized_state(...)
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame

    Return something with:
    - configuration.box
    - particles.position
    """

    # ============================================================
    # Result dictionary from get_or_make_thermalized_state
    # ============================================================
    if isinstance(obj, dict):
        if "frame" in obj and obj["frame"] is not None:
            return _as_snapshot(obj["frame"])

        if "simulation" in obj and obj["simulation"] is not None:
            return _as_snapshot(obj["simulation"])

        raise TypeError(
            "Result dictionary does not contain a usable frame or simulation."
        )

    # ============================================================
    # None check
    # ============================================================
    if obj is None:
        raise TypeError(
            "Cannot convert None to snapshot. If this came from "
            "result['simulation'], the state was probably loaded from disk. "
            "Use result or result['frame'] instead."
        )

    # ============================================================
    # HOOMD simulation
    # ============================================================
    if hasattr(obj, "state") and hasattr(obj.state, "get_snapshot"):
        return obj.state.get_snapshot()

    # ============================================================
    # HOOMD state
    # ============================================================
    if hasattr(obj, "get_snapshot"):
        return obj.get_snapshot()

    # ============================================================
    # GSD frame or HOOMD snapshot
    # ============================================================
    if hasattr(obj, "configuration") and hasattr(obj, "particles"):
        return obj

    raise TypeError(
        "Expected a result dictionary, HOOMD simulation, "
        "HOOMD state, HOOMD snapshot, or GSD frame."
    )


# ============================================================
# Extract positions and box lengths
# ============================================================

def _get_positions_and_box(obj):
    """
    Extract particle positions and box lengths from any supported object.
    """

    snapshot = _as_snapshot(obj)

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


# ============================================================
# Render system
# ============================================================

def render(obj):
    """
    Render a result dictionary, HOOMD simulation, HOOMD state,
    HOOMD snapshot, or GSD frame.

    Visual settings are intentionally kept the same as the original V1 render.
    """

    snapshot = _as_snapshot(obj)

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
    Compute voxel densities.

    Input can be:
    - result dictionary from sh.get_or_make_thermalized_state(...)
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame

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

    # ============================================================
    # Convert counts to voxel densities
    # ============================================================
    voxel_densities = voxel_counts / voxel_volume

    return voxel_densities, voxel_counts, voxel_volume


# ============================================================
# Plot voxel density histogram
# ============================================================

def plot_voxel_histogram(
    obj,
    nbins=10,
    fit=True,
):
    """
    Plot voxel density histogram.

    The x-axis is always voxel density:

        voxel_density = particles_in_voxel / voxel_volume

    Input can be:
    - result dictionary from sh.get_or_make_thermalized_state(...)
    - HOOMD simulation
    - HOOMD state
    - HOOMD snapshot
    - GSD frame

    Parameters
    ----------
    fit : bool
        If True, overlay a Gaussian fit to the voxel-density distribution.

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
    # Gaussian fit to voxel densities
    # ============================================================
    density_mean = float(np.mean(voxel_densities))
    density_std = float(np.std(voxel_densities, ddof=1))

    x_fit = np.linspace(
        hist_x_edges[0],
        hist_x_edges[-1],
        500,
    )

    bin_width = hist_x_edges[1] - hist_x_edges[0]
    n_voxels = len(voxel_densities)

    if density_std > 0:
        gaussian_pdf = (
            1.0
            / (density_std * np.sqrt(2.0 * np.pi))
            * np.exp(
                -0.5 * ((x_fit - density_mean) / density_std)**2
            )
        )

        gaussian_counts = gaussian_pdf * n_voxels * bin_width
    else:
        gaussian_counts = np.zeros_like(x_fit)

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=(6, 4))

    plt.stairs(
        hist2[0],
        edges=hist2[1],
        edgecolor="black",
        label="Voxel densities",
    )

    if fit and density_std > 0:
        plt.plot(
            x_fit,
            gaussian_counts,
            linestyle="--",
            linewidth=2,
            label=(
                f"Gaussian fit\n"
                f"mean = {density_mean:.4f}, "
                f"std = {density_std:.4f}"
            ),
        )

    plt.axvline(
        density_mean,
        linestyle=":",
        linewidth=2,
        label=f"mean density = {density_mean:.4f}",
    )

    plt.xlabel("Voxel density")
    plt.ylabel("Number of voxels")
    plt.title(f"Voxel density distribution (nbins={nbins})")

    plt.legend()
    plt.show()

    print("Voxel density summary")
    print("=" * 60)
    print(f"nbins:              {nbins}")
    print(f"number of voxels:   {n_voxels}")
    print(f"voxel volume:       {voxel_volume}")
    print(f"mean density:       {density_mean}")
    print(f"std density:        {density_std}")
    print(f"min density:        {voxel_densities.min()}")
    print(f"max density:        {voxel_densities.max()}")
    print("=" * 60)

    return {
    "hist": hist2,
    "hist_y": hist2[0],
    "hist_x_edges": hist2[1],
    "gaussian_mean": density_mean,
    "gaussian_std": density_std,
    "voxel_densities": voxel_densities,
    "voxel_counts": voxel_counts,
    "voxel_volume": voxel_volume,
    }


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
    - result dictionary from sh.get_or_make_thermalized_state(...)
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
    fraction=0.05,
    point_size=1,
    alpha=0.7,
):
    """
    Plot an x-y slice through the middle of the box.

    Input can be:
    - result dictionary from sh.get_or_make_thermalized_state(...)
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
# Plot cavitation x-y slice
# ============================================================

def plot_cavitation_xy_slice(
    obj,
    fraction=0.05,
    point_size=1,
    alpha=0.7,
    show_bubble=True,
):
    """
    Plot a thin x-y slice through a cavitation bubble.

    Input should usually be the result dictionary returned by
    cavitation.get_or_create_cavitation_state(...) or
    cavitation.get_or_create_cavitation(...).
    """

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    info = {}

    if isinstance(obj, dict):
        info = dict(obj.get("creation_info", {}))

        if not info and "initial_result" in obj:
            info = dict(
                obj.get("initial_result", {}).get("creation_info", {})
            )

    bubble_center = info.get("bubble_center", None)

    if bubble_center is not None:
        bubble_center = np.asarray(
            bubble_center,
            dtype=np.float64,
        )

    center_x = float(info.get(
        "bubble_center_x",
        0.0 if bubble_center is None else bubble_center[0],
    ))
    center_y = float(info.get(
        "bubble_center_y",
        0.0 if bubble_center is None else bubble_center[1],
    ))
    center_z = float(info.get(
        "bubble_center_z",
        0.0 if bubble_center is None else bubble_center[2],
    ))

    bubble_radius = info.get("bubble_radius", None)

    positions, Lx, Ly, Lz, snapshot = _get_positions_and_box(obj)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    half_thickness = 0.5 * fraction * Lz
    mask = np.abs(z - center_z) <= half_thickness

    plt.figure(figsize=(6, 6))

    plt.scatter(
        x[mask],
        y[mask],
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    if show_bubble and bubble_radius is not None:
        circle = plt.Circle(
            (center_x, center_y),
            float(bubble_radius),
            fill=False,
            linewidth=2,
            color="red",
            linestyle="--",
        )
        plt.gca().add_patch(circle)

    plt.xlim(-Lx / 2, Lx / 2)
    plt.ylim(-Ly / 2, Ly / 2)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Cavitation x-y slice at z={center_z:.3f} "
        f"({100 * fraction:.1f}% box thickness)"
    )

    plt.gca().set_aspect("equal")
    plt.show()


# ============================================================
# Animate x-y trajectory slice
# ============================================================

def animate_xy_slice_trajectory(
    trajectory_path,
    fraction=0.10,
    stride=1,
    max_frames=100,
    point_size=1,
    alpha=0.7,
    interval=120,
    center_z=0.0,
):
    """
    Animate a thin x-y slice from a many-frame GSD trajectory.

    This is useful for cavitation because a slice shows the bubble interior;
    a full 3D render mostly shows particles on the outside of the box.
    """

    import gsd.hoomd

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    trajectory_path = os.fspath(trajectory_path)

    frames = []

    with gsd.hoomd.open(
        name=trajectory_path,
        mode="r",
    ) as trajectory:
        total_frames = len(trajectory)

        frame_indices = range(0, total_frames, int(stride))

        if max_frames is not None:
            frame_indices = list(frame_indices)[:int(max_frames)]

        for frame_index in frame_indices:
            frame = trajectory[frame_index]

            positions = np.asarray(
                frame.particles.position,
                dtype=np.float64,
            )

            box = np.asarray(
                frame.configuration.box,
                dtype=np.float64,
            )

            Lx = float(box[0])
            Ly = float(box[1])
            Lz = float(box[2])

            half_thickness = 0.5 * float(fraction) * Lz
            z = positions[:, 2]
            mask = np.abs(z - center_z) <= half_thickness

            frames.append({
                "frame_index": int(frame_index),
                "step": int(frame.configuration.step),
                "xy": positions[mask][:, :2],
                "Lx": Lx,
                "Ly": Ly,
            })

    if not frames:
        raise ValueError(f"No frames found in trajectory: {trajectory_path}")

    Lx = frames[0]["Lx"]
    Ly = frames[0]["Ly"]

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        [],
        [],
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    ax.set_xlim(-Lx / 2, Lx / 2)
    ax.set_ylim(-Ly / 2, Ly / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    def update(frame_data):
        scatter.set_offsets(frame_data["xy"])
        ax.set_title(
            f"x-y slice | frame {frame_data['frame_index']} | "
            f"step {frame_data['step']}"
        )
        return (scatter,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=True,
    )

    plt.close(fig)

    return IPython.display.HTML(anim.to_jshtml())


# ============================================================
# Plot cavitation measurements
# ============================================================

def plot_cavitation_measurements(
    measurements,
    figsize=(10, 8),
):
    """
    Plot the main diagnostics from cavitation.measure_cavitation_trajectory().
    """

    if "timestep" not in measurements:
        raise ValueError("measurements must contain a 'timestep' column")

    panels = [
        ("bubble_radius_estimate", "Bubble Radius Estimate"),
        ("void_fraction_estimate", "Void Fraction Estimate"),
        ("pressure", "Pressure"),
        ("PE_per_particle", "PE/N"),
    ]

    available_panels = [
        panel
        for panel in panels
        if panel[0] in measurements
    ]

    if not available_panels:
        raise ValueError(
            "No recognized cavitation measurement columns were found."
        )

    n_panels = len(available_panels)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=figsize,
        sharex=True,
    )

    if n_panels == 1:
        axes = [axes]

    for axis, (column, label) in zip(axes, available_panels):
        axis.plot(
            measurements["timestep"],
            measurements[column],
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep")

    plt.tight_layout()
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
    runs.read_hdf5_log().
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

    Special behavior:
    - "kinetic_energy" plots KE/N
    - "potential_energy" plots PE/N

    The calling keys stay the same:

        vh.plot_log_quantity(log, "pressure")
        vh.plot_log_quantity(log, "kinetic_temperature")
        vh.plot_log_quantity(log, "potential_energy")
        vh.plot_log_quantity(log, "kinetic_energy")
    """

    # ============================================================
    # Extract timestep and thermodynamic data
    # ============================================================
    timestep = log["hoomd-data"]["Simulation"]["timestep"]

    thermo = (
        log["hoomd-data"]["md"]
           ["compute"]
           ["ThermodynamicQuantities"]
    )

    values = np.asarray(
        thermo[quantity],
        dtype=float,
    )

    # ============================================================
    # Convert total energies to per-particle energies
    # ============================================================
    if quantity in ["kinetic_energy", "potential_energy"]:
        metadata = (
            log.get("metadata", {})
               .get("state", {})
               .get("attrs", {})
        )

        if "N" not in metadata:
            metadata = (
                log.get("metadata", {})
                   .get("attrs", {})
            )

        N = int(metadata["N"])

        values = values / N

        if quantity == "kinetic_energy":
            y_label = "KE/N"
            title = "KE/N vs Timestep"

        elif quantity == "potential_energy":
            y_label = "PE/N"
            title = "PE/N vs Timestep"

    else:
        y_label = quantity.replace("_", " ").title()
        title = f"{y_label} vs Timestep"

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=figsize)

    plt.plot(timestep, values)

    plt.xlabel("Timestep")
    plt.ylabel(y_label)
    plt.title(title)

    plt.grid(alpha=0.3)

    plt.show()
