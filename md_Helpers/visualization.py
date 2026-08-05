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

from .spatial import (
    as_snapshot as _as_snapshot,
    compute_voxel_densities,
    positions_and_box,
)

device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=300, h=300)

FRESNEL_MIN_VERSION = packaging.version.parse("0.13.0")
FRESNEL_MAX_VERSION = packaging.version.parse("0.14.0")


def _get_positions_and_box(obj):
    positions, box_lengths, snapshot = positions_and_box(obj, wrap=True)
    return positions, *map(float, box_lengths), snapshot


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


def plot_voxel_mixture_fit(
    fit,
    x_axis="density",
    show_residuals=True,
    figsize=(8, 7),
):
    """Plot an observed voxel histogram and its three-component fit."""

    if x_axis == "density":
        x = fit["density_axis"]
        xlabel = "Voxel density"
    elif x_axis == "count":
        x = fit["count_axis"]
        xlabel = "Particles per voxel"
    else:
        raise ValueError("x_axis must be 'density' or 'count'")

    if show_residuals:
        fig, (axis, residual_axis) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, axis = plt.subplots(figsize=figsize)
        residual_axis = None

    axis.step(
        x,
        fit["observed_counts"],
        where="mid",
        color="black",
        linewidth=1.5,
        label="Observed",
    )
    axis.plot(x, fit["model_counts"], linewidth=2.5, label="Total fit")
    axis.plot(x, fit["gas_counts"], linestyle="--", label="Gas Poisson")
    axis.plot(x, fit["liquid_counts"], linestyle="--", label="Liquid Gaussian")
    axis.plot(
        x,
        fit["interface_counts"],
        linestyle="--",
        label="Interface integral",
    )
    axis.set_ylabel("Number of voxels")
    axis.set_title(
        f"Voxel mixture fit at step {fit.get('timestep', 'unknown')}"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    if residual_axis is not None:
        residuals = fit["observed_counts"] - fit["model_counts"]
        residual_axis.axhline(0.0, color="black", linewidth=1)
        residual_axis.step(x, residuals, where="mid")
        residual_axis.set_ylabel("Data - fit")
        residual_axis.grid(alpha=0.25)
        residual_axis.set_xlabel(xlabel)
    else:
        axis.set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()


# ============================================================
# Plot trajectory voxel density histograms
# ============================================================

def _get_trajectory_path(obj):
    """
    Accept a trajectory path or an evolved-run result dictionary.
    """

    if isinstance(obj, dict):
        paths = obj.get("paths", {})

        if "trajectory_path" in paths:
            return os.fspath(paths["trajectory_path"])

        run_result = obj.get("run_result", {})

        if "trajectory_path" in run_result:
            return os.fspath(run_result["trajectory_path"])

    return os.fspath(obj)


def _is_two_segment_trajectory(obj):
    if not isinstance(obj, dict):
        return False
    paths = obj.get("paths", obj)
    return "segment_1" in paths and "segment_2" in paths


def _select_trajectory_frame_indices(
    total_frames,
    last_n=5,
    skip=4,
    frame_indices=None,
):
    """
    Pick trajectory frames for comparison.

    Default selects the last `last_n` frames, stepping backward by `skip`.
    Example with last_n=5 and skip=4:
        [..., final-16, final-12, final-8, final-4, final]
    """

    if frame_indices is not None:
        selected = [
            int(index)
            for index in frame_indices
        ]

        return [
            index if index >= 0 else total_frames + index
            for index in selected
        ]

    last_index = total_frames - 1

    selected = [
        last_index - int(skip) * offset
        for offset in range(int(last_n))
    ]

    selected = [
        index
        for index in selected
        if index >= 0
    ]

    return sorted(selected)


def plot_trajectory_voxel_histograms(
    trajectory,
    nbins=10,
    last_n=5,
    skip=4,
    frame_indices=None,
    histogram_bins="shared_counts",
    n_density_bins=40,
    normalize=False,
    alpha=0.8,
    figsize=(7, 5),
):
    """
    Plot voxel-density histograms from multiple trajectory frames together.

    Parameters
    ----------
    trajectory : path-like or dict
        Path to a GSD trajectory, or an evolved-run result dictionary with
        paths["trajectory_path"].

    nbins : int
        Number of voxel divisions per box dimension.

    last_n : int
        Number of frames to plot when frame_indices is not provided.

    skip : int
        Spacing between selected frames when walking backward from the final
        frame. skip=4 means final, final-4, final-8, ...

    frame_indices : list[int], optional
        Exact frame indices to plot. Negative indices are accepted.

    histogram_bins : {"shared_counts", "linear_density"}
        shared_counts uses integer particle-count bin edges converted into
        density using the first frame's voxel volume. This is best when the box
        volume is fixed, as in current cavitation runs.

        linear_density uses evenly spaced density bins across all selected
        frames.

    normalize : bool
        If True, plot fraction of voxels instead of number of voxels.

    Returns
    -------
    dict
        Selected frame indices, summary table, common bin edges, and histogram
        values.
    """

    import gsd.hoomd

    trajectory_path = _get_trajectory_path(trajectory)

    frame_data = []

    with gsd.hoomd.open(
        name=trajectory_path,
        mode="r",
    ) as gsd_trajectory:
        selected_indices = _select_trajectory_frame_indices(
            total_frames=len(gsd_trajectory),
            last_n=last_n,
            skip=skip,
            frame_indices=frame_indices,
        )

        if not selected_indices:
            raise ValueError("No trajectory frames were selected.")

        for frame_index in selected_indices:
            frame = gsd_trajectory[frame_index]

            voxel_densities, voxel_counts, voxel_volume = (
                compute_voxel_densities(
                    frame,
                    nbins=nbins,
                )
            )

            frame_data.append({
                "frame_index": int(frame_index),
                "timestep": int(frame.configuration.step),
                "voxel_densities": voxel_densities,
                "voxel_counts": voxel_counts,
                "voxel_volume": float(voxel_volume),
                "mean_density": float(np.mean(voxel_densities)),
                "std_density": float(np.std(voxel_densities, ddof=1)),
                "min_density": float(np.min(voxel_densities)),
                "max_density": float(np.max(voxel_densities)),
                "low_zero_voxels": int(np.sum(voxel_counts == 0)),
            })

    all_densities = np.concatenate([
        item["voxel_densities"]
        for item in frame_data
    ])

    if histogram_bins == "shared_counts":
        voxel_volume = frame_data[0]["voxel_volume"]

        min_count = min(
            int(np.floor(np.min(item["voxel_counts"])))
            for item in frame_data
        )
        max_count = max(
            int(np.ceil(np.max(item["voxel_counts"])))
            for item in frame_data
        )

        count_edges = np.arange(
            min_count - 0.5,
            max_count + 1.5,
            1,
        )

        density_edges = count_edges / voxel_volume

    elif histogram_bins == "linear_density":
        density_edges = np.linspace(
            float(np.min(all_densities)),
            float(np.max(all_densities)),
            int(n_density_bins) + 1,
        )

    else:
        raise ValueError(
            "histogram_bins must be 'shared_counts' or 'linear_density'."
        )

    histograms = []

    plt.figure(figsize=figsize)

    for item in frame_data:
        hist_y, hist_x_edges = np.histogram(
            item["voxel_densities"],
            bins=density_edges,
        )

        if normalize:
            hist_y = hist_y / np.sum(hist_y)
            y_label = "Fraction of voxels"
        else:
            y_label = "Number of voxels"

        histograms.append({
            "frame_index": item["frame_index"],
            "timestep": item["timestep"],
            "hist_y": hist_y,
            "hist_x_edges": hist_x_edges,
        })

        plt.stairs(
            hist_y,
            edges=hist_x_edges,
            linewidth=1.8,
            alpha=alpha,
            label=(
                f"frame {item['frame_index']} "
                f"(step {item['timestep']})"
            ),
        )

    plt.xlabel("Voxel density")
    plt.ylabel(y_label)
    plt.title(
        f"Trajectory voxel-density histograms "
        f"(voxel grid {nbins}x{nbins}x{nbins})"
    )
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    summary = [
        {
            "frame_index": item["frame_index"],
            "timestep": item["timestep"],
            "mean_density": item["mean_density"],
            "std_density": item["std_density"],
            "min_density": item["min_density"],
            "max_density": item["max_density"],
            "empty_voxels": item["low_zero_voxels"],
        }
        for item in frame_data
    ]

    return {
        "trajectory_path": trajectory_path,
        "selected_frame_indices": selected_indices,
        "summary": summary,
        "density_edges": density_edges,
        "histograms": histograms,
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

    bubble_radius = info.get(
        "radius",
        info.get("bubble_radius"),
    )

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


def _hot_spike_creation_info(obj):
    info = {}

    if isinstance(obj, dict):
        info = dict(obj.get("creation_info", {}))

        if not info and "initial_result" in obj:
            info = dict(
                obj.get("initial_result", {}).get("creation_info", {})
            )

    return info


# ============================================================
# Plot hot-spike x-y slice
# ============================================================

def plot_hot_spike_xy_slice(
    obj,
    fraction=0.05,
    point_size=1,
    alpha=0.7,
    show_radius=True,
):
    """
    Plot a thin x-y slice through the hot-spike center.

    Input should usually be the result dictionary returned by
    hot_spike.get_or_create_hot_spike_state(...) or
    hot_spike.get_or_create_hot_spike(...).
    """

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    info = _hot_spike_creation_info(obj)

    spike_center = info.get("spike_center", None)
    if spike_center is not None:
        spike_center = np.asarray(
            spike_center,
            dtype=np.float64,
        )

    center_x = float(info.get(
        "spike_center_x",
        0.0 if spike_center is None else spike_center[0],
    ))
    center_y = float(info.get(
        "spike_center_y",
        0.0 if spike_center is None else spike_center[1],
    ))
    center_z = float(info.get(
        "spike_center_z",
        0.0 if spike_center is None else spike_center[2],
    ))

    radius = info.get("radius", None)

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

    if show_radius and radius is not None:
        circle = plt.Circle(
            (center_x, center_y),
            float(radius),
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
        f"Hot-spike x-y slice at z={center_z:.3f} "
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


def animate_hot_spike_xy_slice_trajectory(
    result,
    fraction=0.10,
    stride=1,
    max_frames=100,
    point_size=1,
    alpha=0.7,
    interval=120,
    show_radius=True,
):
    """
    Animate the hot-spike trajectory x-y slice with the initial radius overlaid.
    """

    import gsd.hoomd

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    info = _hot_spike_creation_info(result)

    center = np.array(
        [
            info.get("spike_center_x", 0.0),
            info.get("spike_center_y", 0.0),
            info.get("spike_center_z", 0.0),
        ],
        dtype=np.float64,
    )
    radius = info.get("radius", None)

    frames = []

    if _is_two_segment_trajectory(result):
        from .excitation_evolution import iter_stitched_trajectory

        frame_items = iter_stitched_trajectory(
            result,
            stride=stride,
            max_frames=max_frames,
        )
    else:
        trajectory_path = _get_trajectory_path(result)

        def single_trajectory_items():
            with gsd.hoomd.open(
                name=trajectory_path,
                mode="r",
            ) as trajectory:
                frame_indices = range(0, len(trajectory), int(stride))
                if max_frames is not None:
                    frame_indices = list(frame_indices)[:int(max_frames)]
                for output_index, frame_index in enumerate(frame_indices):
                    frame = trajectory[frame_index]
                    yield {
                        "frame": frame,
                        "frame_index": output_index,
                        "segment_index": 1,
                        "timestep": int(frame.configuration.step),
                        "elapsed_time": np.nan,
                    }

        frame_items = single_trajectory_items()

    for item in frame_items:
        frame = item["frame"]
        positions = np.asarray(
            frame.particles.position,
            dtype=np.float64,
        )
        box_lengths = np.asarray(
            frame.configuration.box[:3],
            dtype=np.float64,
        )
        positions = (
            (positions + 0.5 * box_lengths) % box_lengths
            - 0.5 * box_lengths
        )
        dz = positions[:, 2] - center[2]
        dz -= box_lengths[2] * np.round(dz / box_lengths[2])
        mask = np.abs(dz) <= 0.5 * float(fraction) * box_lengths[2]

        frames.append({
            "frame_index": int(item["frame_index"]),
            "step": int(item["timestep"]),
            "segment_index": int(item["segment_index"]),
            "elapsed_time": float(item["elapsed_time"]),
            "xy": positions[mask][:, :2],
            "box_lengths": box_lengths,
        })

    if not frames:
        raise ValueError("No frames found in the hot-spike trajectory.")

    Lx, Ly = frames[0]["box_lengths"][:2]

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        [],
        [],
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )
    artists = [scatter]

    circle = None
    if show_radius and radius is not None:
        circle = plt.Circle(
            center[:2],
            float(radius),
            fill=False,
            linewidth=2,
            color="red",
            linestyle="--",
        )
        ax.add_patch(circle)
        artists.append(circle)

    ax.set_xlim(-Lx / 2, Lx / 2)
    ax.set_ylim(-Ly / 2, Ly / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    def update(frame_data):
        scatter.set_offsets(frame_data["xy"])
        if np.isfinite(frame_data["elapsed_time"]):
            time_label = (
                f" | segment {frame_data['segment_index']}"
                f" | time {frame_data['elapsed_time']:.4f}"
            )
        else:
            time_label = ""
        ax.set_title(
            f"Hot-spike x-y slice | frame {frame_data['frame_index']} | "
            f"step {frame_data['step']}{time_label}"
        )
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=True,
    )

    plt.close(fig)

    return IPython.display.HTML(anim.to_jshtml())


def animate_masked_nph_hot_spike_xy_slice_trajectory(
    result,
    fraction=0.10,
    stride=1,
    max_frames=100,
    point_size=1,
    alpha=0.7,
    interval=120,
    particle_stride=1,
    masked_color="tab:orange",
    inner_color="tab:blue",
    show_mask_boundary=True,
    show_radius=True,
    show_box=True,
):
    """
    Animate a masked NPH hot-spike trajectory with its changing box.

    The particle mask is a fixed set of tags selected after excitation. The
    box, mask boundary, spike center, and spike radius are scaled using each
    frame's box lengths. ``stride`` and ``max_frames`` behave identically to
    :func:`animate_hot_spike_xy_slice_trajectory`. ``particle_stride`` can
    reduce rendering cost without changing which trajectory frames are used.
    """

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    if int(particle_stride) <= 0:
        raise ValueError("particle_stride must be positive")
    if not isinstance(result, dict) or "initial_result" not in result:
        raise ValueError(
            "Expected the result returned by get_or_create_hot_spike()."
        )

    from .excitation_evolution import (
        build_outer_pressure_mask,
        iter_stitched_trajectory,
    )

    paths = result.get("paths", {})
    diameter_fraction = paths.get("outer_mask_diameter_fraction", 0.75)
    if diameter_fraction is None:
        raise ValueError(
            "This result does not use an outer NPH particle mask."
        )

    mask_info = build_outer_pressure_mask(
        result["initial_result"],
        diameter_fraction=diameter_fraction,
    )
    creation_info = _hot_spike_creation_info(result)
    reference_box = np.asarray(
        result["initial_result"]["source_result"]["frame"]
              .configuration.box[:3],
        dtype=np.float64,
    )
    reference_center = np.asarray(mask_info["center"], dtype=np.float64)
    reference_mask_radius = float(mask_info["radius"])
    reference_spike_radius = creation_info.get("radius")

    particle_count = int(result["initial_result"]["frame"].particles.N)
    outer_membership = np.zeros(particle_count, dtype=bool)
    outer_membership[mask_info["outer_tags"].astype(np.int64)] = True
    displayed_particles = (
        np.arange(particle_count, dtype=np.int64) % int(particle_stride) == 0
    )

    frames = []
    for item in iter_stitched_trajectory(
        result,
        stride=stride,
        max_frames=max_frames,
    ):
        frame = item["frame"]
        positions = np.asarray(frame.particles.position, dtype=np.float64)
        if positions.shape[0] != particle_count:
            raise ValueError(
                "Particle count changed, so fixed mask tags cannot be applied."
            )

        box_lengths = np.asarray(
            frame.configuration.box[:3],
            dtype=np.float64,
        )
        positions = (
            (positions + 0.5 * box_lengths) % box_lengths
            - 0.5 * box_lengths
        )

        scale = box_lengths / reference_box
        center = reference_center * scale
        dz = positions[:, 2] - center[2]
        dz -= box_lengths[2] * np.round(dz / box_lengths[2])
        slice_mask = (
            np.abs(dz) <= 0.5 * float(fraction) * box_lengths[2]
        )
        outer_visible = slice_mask & outer_membership & displayed_particles
        inner_visible = slice_mask & ~outer_membership & displayed_particles
        isotropic_scale = float(np.min(scale))

        frames.append({
            "frame_index": int(item["frame_index"]),
            "step": int(item["timestep"]),
            "segment_index": int(item["segment_index"]),
            "elapsed_time": float(item["elapsed_time"]),
            "outer_xy": positions[outer_visible, :2],
            "inner_xy": positions[inner_visible, :2],
            "box_lengths": box_lengths,
            "center_xy": center[:2],
            "mask_radius": reference_mask_radius * isotropic_scale,
            "spike_radius": (
                None
                if reference_spike_radius is None
                else float(reference_spike_radius) * isotropic_scale
            ),
            "volume": float(np.prod(box_lengths)),
            "density": float(particle_count / np.prod(box_lengths)),
        })

    if not frames:
        raise ValueError("No frames found in the masked NPH trajectory.")

    max_lx = max(frame["box_lengths"][0] for frame in frames)
    max_ly = max(frame["box_lengths"][1] for frame in frames)
    axis_half_width = 0.525 * max(max_lx, max_ly)

    fig, ax = plt.subplots(figsize=(7, 7))
    inner_scatter = ax.scatter(
        [],
        [],
        s=point_size,
        alpha=alpha,
        color=inner_color,
        label="Inner / unmasked",
        rasterized=True,
    )
    outer_scatter = ax.scatter(
        [],
        [],
        s=point_size,
        alpha=alpha,
        color=masked_color,
        label="Outer pressure mask",
        rasterized=True,
    )
    artists = [inner_scatter, outer_scatter]

    box_patch = None
    if show_box:
        box_patch = plt.Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            fill=False,
            linewidth=2,
            color="black",
            label="Simulation box",
        )
        ax.add_patch(box_patch)
        artists.append(box_patch)

    mask_circle = None
    if show_mask_boundary:
        mask_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            fill=False,
            linewidth=2,
            linestyle="--",
            color=masked_color,
            label="Scaled mask boundary",
        )
        ax.add_patch(mask_circle)
        artists.append(mask_circle)

    spike_circle = None
    if show_radius and reference_spike_radius is not None:
        spike_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            fill=False,
            linewidth=2,
            linestyle=":",
            color="red",
            label="Excitation radius",
        )
        ax.add_patch(spike_circle)
        artists.append(spike_circle)

    title = ax.set_title("")
    artists.append(title)
    ax.set_xlim(-axis_half_width, axis_half_width)
    ax.set_ylim(-axis_half_width, axis_half_width)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=4)

    initial_volume = frames[0]["volume"]

    def update(frame_data):
        inner_scatter.set_offsets(frame_data["inner_xy"])
        outer_scatter.set_offsets(frame_data["outer_xy"])
        lx, ly = frame_data["box_lengths"][:2]

        if box_patch is not None:
            box_patch.set_xy((-0.5 * lx, -0.5 * ly))
            box_patch.set_width(lx)
            box_patch.set_height(ly)
        if mask_circle is not None:
            mask_circle.center = frame_data["center_xy"]
            mask_circle.set_radius(frame_data["mask_radius"])
        if spike_circle is not None:
            spike_circle.center = frame_data["center_xy"]
            spike_circle.set_radius(frame_data["spike_radius"])

        volume_change = 100.0 * (
            frame_data["volume"] / initial_volume - 1.0
        )
        time_label = (
            f" | time {frame_data['elapsed_time']:.4f}"
            if np.isfinite(frame_data["elapsed_time"])
            else ""
        )
        title.set_text(
            f"Masked NPH | frame {frame_data['frame_index']}"
            f" | segment {frame_data['segment_index']}"
            f" | step {frame_data['step']}{time_label}\n"
            f"Lx={lx:.4f}, Ly={ly:.4f}"
            f" | volume change={volume_change:+.3f}%"
            f" | density={frame_data['density']:.5f}"
        )
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=True,
    )
    plt.close(fig)
    return IPython.display.HTML(anim.to_jshtml())


def _periodic_weighted_center(points, weights, box_lengths):
    """Return a weighted center for periodic coordinates in [-L/2, L/2)."""

    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    box_lengths = np.asarray(box_lengths, dtype=float)

    center = []
    for axis in range(3):
        angles = 2.0 * np.pi * (
            (points[:, axis] + 0.5 * box_lengths[axis])
            / box_lengths[axis]
        )
        sin_mean = np.sum(weights * np.sin(angles))
        cos_mean = np.sum(weights * np.cos(angles))
        angle = np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)
        coordinate = angle * box_lengths[axis] / (2.0 * np.pi)
        center.append(coordinate - 0.5 * box_lengths[axis])

    return np.asarray(center, dtype=float)


def _estimate_low_density_center_from_frame(frame, nbins):
    """Estimate bubble center from low-density voxels in one frame."""

    positions, box_lengths, _ = positions_and_box(frame, wrap=True)
    bounds = [
        [-box_length / 2.0, box_length / 2.0]
        for box_length in box_lengths
    ]
    counts, edges = np.histogramdd(
        positions,
        bins=int(nbins),
        range=bounds,
    )
    flat_counts = counts.ravel()

    center_axes = [
        0.5 * (axis_edges[:-1] + axis_edges[1:])
        for axis_edges in edges
    ]
    centers = np.stack(
        np.meshgrid(*center_axes, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)

    low_cut = np.percentile(flat_counts, 20)
    liquid_count = np.percentile(flat_counts, 75)
    low_mask = flat_counts <= low_cut
    weights = np.clip(liquid_count - flat_counts[low_mask], 0.0, None)

    if weights.size == 0 or np.sum(weights) <= 0.0:
        min_index = int(np.argmin(flat_counts))
        return centers[min_index], {
            "center_method": "minimum_count_voxel",
            "center_voxel_count": float(flat_counts[min_index]),
        }

    return _periodic_weighted_center(
        centers[low_mask],
        weights,
        box_lengths,
    ), {
        "center_method": "weighted_low_density_voxels",
        "center_low_count_percentile": 20.0,
        "center_liquid_count_percentile": 75.0,
        "center_n_low_voxels": int(np.sum(low_mask)),
    }


def fit_and_animate_final_bubble(
    trajectory,
    nbins=12,
    nframes=5,
    skip=5,
    tail_fraction=0.5,
    interface_void_fraction=0.5,
    interface_points=40,
    max_iterations=500,
    slice_fraction=0.10,
    bubble_center=None,
    point_size=1,
    alpha=0.7,
    interval=120,
    show_histogram=True,
    show_residuals=True,
    phase_separation_kwargs=None,
):
    """Fit a pooled tail histogram and overlay its radius for a quick check.

    The histogram fit uses ``nframes`` frames separated by ``skip`` from the
    final ``tail_fraction`` of the trajectory.  The video independently shows
    the final 50 consecutive frames, or the entire trajectory when it contains
    fewer than 50 frames.  The fitted radius uses
    ``gas_weight + 0.5 * interface_weight`` by default.  If ``bubble_center``
    is not supplied, the circle center is estimated from low-density voxels in
    the final frame.

    Returns a dictionary containing ``fit`` (the per-frame-average smoothed
    histogram and size estimate), ``has_bubble``, and ``animation`` (notebook
    HTML, or ``None`` when the final state rethermalized).
    """

    import gsd.hoomd
    from .classification import compute_voxel_fraction_phase_separation
    from .voxel_fit import fit_trajectory_tail_voxel_histogram

    phase_separation_kwargs = dict(phase_separation_kwargs or {})

    if slice_fraction <= 0.0 or slice_fraction > 1.0:
        raise ValueError("slice_fraction must satisfy 0 < value <= 1")

    trajectory_path = _get_trajectory_path(trajectory)

    bubble_center_source = "explicit"
    center_diagnostics = {}

    if bubble_center is None:
        creation_info = {}
        if isinstance(trajectory, dict):
            creation_info = dict(trajectory.get("creation_info", {}))
            if not creation_info:
                creation_info = dict(
                    trajectory.get("initial_result", {}).get(
                        "creation_info",
                        {},
                    )
                )
        constructed_center = creation_info.get("bubble_center", [
            creation_info.get("bubble_center_x", 0.0),
            creation_info.get("bubble_center_y", 0.0),
            creation_info.get("bubble_center_z", 0.0),
        ])
        bubble_center = constructed_center
        bubble_center_source = "final_frame_low_density_voxels"
    else:
        constructed_center = bubble_center

    bubble_center = np.asarray(bubble_center, dtype=float)
    if bubble_center.shape != (3,):
        raise ValueError("bubble_center must contain three coordinates")

    fit = fit_trajectory_tail_voxel_histogram(
        trajectory_path=trajectory_path,
        voxel_nbins=nbins,
        nframes=nframes,
        skip=skip,
        tail_fraction=tail_fraction,
        interface_void_fraction=interface_void_fraction,
        interface_points=interface_points,
        max_iterations=max_iterations,
    )

    if show_histogram:
        plot_voxel_mixture_fit(
            fit,
            x_axis="density",
            show_residuals=show_residuals,
        )

    frames = []
    with gsd.hoomd.open(name=trajectory_path, mode="r") as gsd_trajectory:
        final_frame = gsd_trajectory[-1]
        phase_separation = compute_voxel_fraction_phase_separation(
            final_frame,
            nbins=nbins,
            **phase_separation_kwargs,
        )
        has_bubble = bool(phase_separation["phase_separated"])

        if not has_bubble:
            return {
                "fit": fit,
                "animation": None,
                "has_bubble": False,
                "outcome": "rethermalized",
                "message": "Final state is not phase separated; no bubble found.",
                "phase_separation": phase_separation,
                "fit_frame_indices": fit["frame_indices"],
                "frame_indices": [],
                "bubble_radius": None,
                "bubble_center": None,
                "bubble_center_source": None,
                "bubble_center_diagnostics": {},
                "bubble_volume": None,
                "bubble_volume_fraction": None,
            }

        if bubble_center_source == "final_frame_low_density_voxels":
            try:
                bubble_center, center_diagnostics = (
                    _estimate_low_density_center_from_frame(
                        gsd_trajectory[-1],
                        nbins=nbins,
                    )
                )
            except Exception as error:
                bubble_center = np.asarray(constructed_center, dtype=float)
                bubble_center_source = "constructed_center_fallback"
                center_diagnostics = {"center_error": repr(error)}

        first_video_frame = max(0, len(gsd_trajectory) - 50)
        video_frame_indices = list(range(
            first_video_frame,
            len(gsd_trajectory),
        ))
        for frame_index in video_frame_indices:
            frame = gsd_trajectory[int(frame_index)]
            positions = np.asarray(frame.particles.position, dtype=float)
            box_lengths = np.asarray(frame.configuration.box[:3], dtype=float)
            positions = (
                (positions + 0.5 * box_lengths) % box_lengths
                - 0.5 * box_lengths
            )
            dz = positions[:, 2] - bubble_center[2]
            dz -= box_lengths[2] * np.round(dz / box_lengths[2])
            mask = np.abs(dz) <= 0.5 * float(slice_fraction) * box_lengths[2]
            frames.append({
                "frame_index": int(frame_index),
                "step": int(frame.configuration.step),
                "xy": positions[mask, :2],
                "box_lengths": box_lengths,
            })

    first = frames[0]
    Lx, Ly = first["box_lengths"][:2]
    radius = float(fit["bubble_radius_estimate"])
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        [],
        [],
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )
    circle = plt.Circle(
        bubble_center[:2],
        radius,
        fill=False,
        linewidth=2,
        color="red",
        linestyle="--",
    )
    ax.add_patch(circle)

    ax.set_xlim(-Lx / 2.0, Lx / 2.0)
    ax.set_ylim(-Ly / 2.0, Ly / 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    def update(frame_data):
        scatter.set_offsets(frame_data["xy"])
        ax.set_title(
            f"Final bubble fit: R={radius:.3f} | "
            f"frame {frame_data['frame_index']} | step {frame_data['step']}"
        )
        return scatter, circle

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=True,
    )
    html_animation = IPython.display.HTML(anim.to_jshtml())
    plt.close(fig)

    return {
        "fit": fit,
        "animation": html_animation,
        "has_bubble": True,
        "outcome": "phase_separated",
        "phase_separation": phase_separation,
        "fit_frame_indices": fit["frame_indices"],
        "video_frame_indices": video_frame_indices,
        "frame_indices": video_frame_indices,
        "bubble_radius": radius,
        "bubble_center": bubble_center,
        "bubble_center_source": bubble_center_source,
        "bubble_center_diagnostics": center_diagnostics,
        "bubble_volume": fit["bubble_volume_estimate"],
        "bubble_volume_fraction": fit["bubble_volume_fraction"],
    }


# ============================================================
# Plot cavitation measurements
# ============================================================

def plot_cavitation_measurements(
    measurements,
    figsize=(10, 8),
):
    """
    Plot diagnostics from cavitation_analysis.measure_cavitation_trajectory().
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
    if "stitched" in log and "elapsed_time" in log["stitched"]:
        x_values = log["stitched"]["elapsed_time"]
        x_label = "Elapsed Physical Time"
        title = "Pressure vs Elapsed Physical Time"
    else:
        x_values = log["hoomd-data"]["Simulation"]["timestep"]
        x_label = "Timestep"
        title = "Pressure vs Timestep"

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

    plt.plot(x_values, pressure)

    plt.xlabel(x_label)
    plt.ylabel("Pressure")
    plt.title(title)

    plt.grid(alpha=0.3)

    plt.show()


# ============================================================
# Plot logged thermodynamic quantity
# ============================================================

def plot_log_quantity(
    log,
    quantity,
    figsize=(8, 5),
    show_segment_boundary=True,
):
    """
    Plot a ThermodynamicQuantities quantity against timestep.

    Special behavior:
    - "kinetic_energy" plots KE/N
    - "potential_energy" plots PE/N
    - stitched two-segment logs mark the segment boundary

    The calling keys stay the same:

        vh.plot_log_quantity(log, "pressure")
        vh.plot_log_quantity(log, "kinetic_temperature")
        vh.plot_log_quantity(log, "potential_energy")
        vh.plot_log_quantity(log, "kinetic_energy")
    """

    # ============================================================
    # Extract timestep and thermodynamic data
    # ============================================================
    if "stitched" in log and "elapsed_time" in log["stitched"]:
        x_values = log["stitched"]["elapsed_time"]
        x_label = "Elapsed Physical Time"
        x_name = "Elapsed Physical Time"
    else:
        x_values = log["hoomd-data"]["Simulation"]["timestep"]
        x_label = "Timestep"
        x_name = "Timestep"

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
            title = f"KE/N vs {x_name}"

        elif quantity == "potential_energy":
            y_label = "PE/N"
            title = f"PE/N vs {x_name}"

    else:
        y_label = quantity.replace("_", " ").title()
        title = f"{y_label} vs {x_name}"

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=figsize)

    plt.plot(x_values, values)

    if (
        show_segment_boundary
        and "stitched" in log
        and "segment_boundary_time" in log["stitched"]
    ):
        boundary_time = float(log["stitched"]["segment_boundary_time"])
        plt.axvline(
            boundary_time,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"Segment boundary (t={boundary_time:g})",
        )
        plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.grid(alpha=0.3)

    plt.show()


def plot_ke_and_fastest_particle_speed(
    result,
    max_time=500.0,
    figsize=(10, 6),
    show_segment_boundary=True,
):
    """
    Plot KE/N and the fastest saved particle speed on separate y-axes.

    The fastest speed is evaluated from the GSD frame at every stitched log
    point through ``max_time``. Log rows and trajectory frames are matched by
    both segment index and timestep, so the change in timestep size does not
    introduce an alignment ambiguity.
    """

    from .excitation_evolution import (
        iter_stitched_trajectory,
        read_stitched_log,
    )

    max_time = float(max_time)
    if max_time < 0:
        raise ValueError("max_time must be nonnegative")

    log = read_stitched_log(result)
    stitched = log["stitched"]
    elapsed_time = np.asarray(stitched["elapsed_time"], dtype=np.float64)
    timesteps = np.asarray(stitched["timestep"], dtype=np.int64)
    segment_indices = np.asarray(
        stitched["segment_index"],
        dtype=np.int8,
    )

    selected = elapsed_time <= max_time
    elapsed_time = elapsed_time[selected]
    timesteps = timesteps[selected]
    segment_indices = segment_indices[selected]
    if elapsed_time.size == 0:
        raise ValueError(f"No log points were found through t={max_time:g}")

    thermo = (
        log["hoomd-data"]["md"]
           ["compute"]
           ["ThermodynamicQuantities"]
    )
    kinetic_energy = np.asarray(
        thermo["kinetic_energy"],
        dtype=np.float64,
    )[selected]

    metadata = (
        log.get("metadata", {})
           .get("state", {})
           .get("attrs", {})
    )
    if "N" not in metadata:
        metadata = log.get("metadata", {}).get("attrs", {})
    particle_count = int(metadata["N"])
    ke_per_particle = kinetic_energy / particle_count

    requested_keys = [
        (int(segment_index), int(timestep))
        for segment_index, timestep in zip(segment_indices, timesteps)
    ]
    needed_keys = set(requested_keys)
    speed_by_key = {}
    fastest_index_by_key = {}

    for item in iter_stitched_trajectory(result):
        if item["elapsed_time"] > max_time:
            break
        key = (int(item["segment_index"]), int(item["timestep"]))
        if key not in needed_keys:
            continue

        velocities = np.asarray(
            item["frame"].particles.velocity,
            dtype=np.float64,
        )
        speed_squared = np.einsum(
            "ij,ij->i",
            velocities,
            velocities,
        )
        fastest_index = int(np.argmax(speed_squared))
        fastest_index_by_key[key] = fastest_index
        speed_by_key[key] = float(np.sqrt(speed_squared[fastest_index]))

    missing_keys = [key for key in requested_keys if key not in speed_by_key]
    if missing_keys:
        preview = ", ".join(map(str, missing_keys[:5]))
        raise ValueError(
            "No matching trajectory frame was found for "
            f"{len(missing_keys)} log point(s): {preview}. "
            "The log_period and trajectory_period may differ."
        )

    fastest_speeds = np.asarray(
        [speed_by_key[key] for key in requested_keys],
        dtype=np.float64,
    )
    fastest_indices = np.asarray(
        [fastest_index_by_key[key] for key in requested_keys],
        dtype=np.int64,
    )

    fig, ke_axis = plt.subplots(figsize=figsize)
    speed_axis = ke_axis.twinx()

    ke_line = ke_axis.plot(
        elapsed_time,
        ke_per_particle,
        color="tab:blue",
        label="KE/N",
    )[0]
    speed_line = speed_axis.plot(
        elapsed_time,
        fastest_speeds,
        color="tab:orange",
        label="Fastest particle speed",
    )[0]

    legend_artists = [ke_line, speed_line]
    if show_segment_boundary:
        boundary_time = float(stitched["segment_boundary_time"])
        boundary_line = ke_axis.axvline(
            boundary_time,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"Segment boundary (t={boundary_time:g})",
        )
        legend_artists.append(boundary_line)

    ke_axis.set_xlim(0, max_time)
    ke_axis.set_xlabel("Elapsed Physical Time")
    ke_axis.set_ylabel("KE/N", color="tab:blue")
    speed_axis.set_ylabel("Fastest Particle Speed", color="tab:orange")
    ke_axis.tick_params(axis="y", labelcolor="tab:blue")
    speed_axis.tick_params(axis="y", labelcolor="tab:orange")
    ke_axis.set_title(
        "KE/N and Fastest Particle Speed vs Elapsed Physical Time"
    )
    ke_axis.grid(alpha=0.3)
    ke_axis.legend(
        legend_artists,
        [artist.get_label() for artist in legend_artists],
        loc="best",
    )
    fig.tight_layout()
    plt.show()

    return {
        "figure": fig,
        "ke_axis": ke_axis,
        "speed_axis": speed_axis,
        "elapsed_time": elapsed_time,
        "timestep": timesteps,
        "segment_index": segment_indices,
        "ke_per_particle": ke_per_particle,
        "fastest_particle_speed": fastest_speeds,
        "fastest_particle_index": fastest_indices,
        "segment_boundary_time": float(stitched["segment_boundary_time"]),
    }
