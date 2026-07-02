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
):
    """Fit a pooled tail histogram and overlay its radius for a quick check.

    The selected frames come from the final ``tail_fraction`` of the
    trajectory, walking backward from the final frame in increments of
    ``skip``.  The fitted radius uses ``gas_weight + 0.5 * interface_weight``
    by default.  The circle stays at the constructed bubble center (or the
    explicitly supplied ``bubble_center``); this is intended as a preliminary
    visual scale check, not bubble tracking.

    Returns a dictionary containing ``fit`` (the per-frame-average smoothed
    histogram and size estimate) and ``animation`` (notebook HTML).
    """

    import gsd.hoomd
    from .voxel_fit import fit_trajectory_tail_voxel_histogram

    if slice_fraction <= 0.0 or slice_fraction > 1.0:
        raise ValueError("slice_fraction must satisfy 0 < value <= 1")

    trajectory_path = _get_trajectory_path(trajectory)

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
        bubble_center = creation_info.get("bubble_center", [
            creation_info.get("bubble_center_x", 0.0),
            creation_info.get("bubble_center_y", 0.0),
            creation_info.get("bubble_center_z", 0.0),
        ])
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
        for frame_index in fit["frame_indices"]:
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
        "frame_indices": fit["frame_indices"],
        "bubble_radius": radius,
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
