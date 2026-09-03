"""Notebook visualizations for saved V4 GSD frames and HDF5 logs."""

from __future__ import annotations

import io
import math
import os
from typing import Sequence

import numpy as np


def _render_array(frame, samples: int = 2_000):
    """Render one GSD frame with the same yellow-sphere style used in V3."""

    try:
        import fresnel
    except ImportError as error:
        raise ImportError(
            "3D rendering requires fresnel. Install it in the notebook environment "
            "with: mamba install -c conda-forge fresnel"
        ) from error

    device = fresnel.Device()
    tracer = fresnel.tracer.Path(device=device, w=300, h=300)
    box = np.asarray(frame.configuration.box, dtype=float)
    lengths = box[:3]
    camera_length = float(np.max(lengths))
    positions = np.asarray(frame.particles.position, dtype=float)

    scene = fresnel.Scene(device)
    geometry = fresnel.geometry.Sphere(scene, N=len(positions), radius=0.5)
    geometry.material = fresnel.material.Material(
        color=fresnel.color.linear([252 / 255, 209 / 255, 1 / 255]),
        roughness=0.5,
    )
    geometry.position[:] = positions
    geometry.outline_width = 0.04
    fresnel.geometry.Box(scene, box, box_radius=0.02)
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
        position=(camera_length * 2, camera_length, camera_length * 2),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
        height=camera_length * 1.4 + 1,
    )
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)
    if "CI" in os.environ:
        samples = min(int(samples), 100)
    # Keep Fresnel's ImageArray intact. Some Fresnel versions expose direct
    # slicing and _repr_png_(), but np.asarray(ImageArray) produces a scalar
    # object array instead of its pixels.
    return tracer.sample(scene, samples=int(samples))


def render_frame(frame, samples: int = 2_000):
    """Return an inline PNG for one GSD frame."""

    try:
        from IPython.display import Image
        from PIL import Image as PILImage
    except ImportError as error:
        raise ImportError("Rendering requires IPython and Pillow") from error
    rendered = _render_array(frame, samples=samples)
    if hasattr(rendered, "_repr_png_"):
        return Image(rendered._repr_png_())
    rgb = np.asarray(rendered[:, :, :3], dtype=np.uint8)
    return Image(PILImage.fromarray(rgb)._repr_png_())


def render_frames_movie(
    frames: Sequence,
    duration: int = 200,
    samples: int = 500,
):
    """Return an inline animated GIF for a sequence of GSD frames."""

    try:
        from IPython.display import Image
        from PIL import Image as PILImage
    except ImportError as error:
        raise ImportError("Movie rendering requires IPython and Pillow") from error
    frames = list(frames)
    if not frames:
        raise ValueError("frames cannot be empty")
    images = []
    for frame in frames:
        rendered = _render_array(frame, samples=samples)
        rgb = np.asarray(rendered[:, :, :3], dtype=np.uint8)
        images.append(PILImage.fromarray(rgb))
    first = images[0].convert("P", palette=PILImage.Palette.ADAPTIVE)
    remaining = [image.quantize(palette=first) for image in images[1:]]
    buffer = io.BytesIO()
    first.save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=remaining,
        duration=int(duration),
        loop=0,
    )
    return Image(data=buffer.getvalue())


def _slice_data(frame, z: float, thickness: float | None, fraction: float):
    positions = np.asarray(frame.particles.position, dtype=float)
    box = np.asarray(frame.configuration.box, dtype=float)
    lx, ly, lz = box[:3]
    if thickness is None:
        if not 0 < float(fraction) <= 1:
            raise ValueError("fraction must satisfy 0 < fraction <= 1")
        thickness = float(fraction) * lz
    if float(thickness) <= 0 or float(thickness) > lz:
        raise ValueError("thickness must be positive and no larger than Lz")
    # Minimum-image distance makes slices work correctly at a periodic boundary.
    dz = (positions[:, 2] - float(z) + lz / 2) % lz - lz / 2
    selected = positions[np.abs(dz) <= float(thickness) / 2]
    return selected, (float(lx), float(ly), float(lz)), float(thickness)


def plot_xy_frames(
    frames: Sequence,
    frame_labels: Sequence[int] | None = None,
    z: float = 0.0,
    thickness: float | None = None,
    fraction: float = 0.05,
    point_size: float = 1.0,
    alpha: float = 0.7,
):
    """Plot one or more x-y slices using the V3 scatter-plot style."""

    import matplotlib.pyplot as plt

    frames = list(frames)
    if not frames:
        raise ValueError("frames cannot be empty")
    labels = list(frame_labels) if frame_labels is not None else list(range(len(frames)))
    columns = min(3, len(frames))
    rows = int(np.ceil(len(frames) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, frame, label in zip(axes, frames, labels):
        selected, (lx, ly, _), actual_thickness = _slice_data(
            frame, z, thickness, fraction
        )
        axis.scatter(
            selected[:, 0] if len(selected) else [],
            selected[:, 1] if len(selected) else [],
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )
        axis.set(xlim=(-lx / 2, lx / 2), ylim=(-ly / 2, ly / 2))
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(
            f"Frame {label}: z={z:g}, thickness={actual_thickness:g}"
        )
    for axis in axes[len(frames):]:
        axis.set_visible(False)
    figure.tight_layout()
    plt.show()
    return figure


def animate_xy_frames(
    frames: Sequence,
    frame_labels: Sequence[int] | None = None,
    z: float = 0.0,
    thickness: float | None = None,
    fraction: float = 0.05,
    point_size: float = 1.0,
    alpha: float = 0.7,
    interval: int = 200,
):
    """Return a JavaScript-backed inline movie of x-y slices."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    frames = list(frames)
    if not frames:
        raise ValueError("frames cannot be empty")
    labels = list(frame_labels) if frame_labels is not None else list(range(len(frames)))
    figure, axis = plt.subplots(figsize=(6, 6))
    scatter = axis.scatter([], [], s=point_size, alpha=alpha)

    def update(index):
        selected, (lx, ly, _), actual_thickness = _slice_data(
            frames[index], z, thickness, fraction
        )
        scatter.set_offsets(selected[:, :2] if len(selected) else np.empty((0, 2)))
        axis.set(xlim=(-lx / 2, lx / 2), ylim=(-ly / 2, ly / 2))
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(
            f"Frame {labels[index]}: z={z:g}, thickness={actual_thickness:g}"
        )
        return (scatter,)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=int(interval),
        blit=False,
    )
    plt.close(figure)
    return HTML(animation.to_jshtml())


def plot_phase_histogram(fit: dict, title: str | None = None):
    """Plot an averaged voxel histogram and any available mixture components."""

    import matplotlib.pyplot as plt

    x = np.asarray(fit["density_axis"], dtype=float)
    observed = np.asarray(fit["observed_counts"], dtype=float)
    figure, axis = plt.subplots(figsize=(8, 5))
    individual = fit.get("individual_histograms")
    if individual is not None:
        for index, histogram in enumerate(np.atleast_2d(individual)):
            axis.step(
                x,
                histogram,
                where="mid",
                color="0.75",
                alpha=0.55,
                label="Selected frames" if index == 0 else None,
            )
    axis.step(x, observed, where="mid", color="black", linewidth=2, label="Average")
    curves = [
        ("gas_counts", "Gas", "--"),
        ("liquid_counts", "Liquid", "--"),
        ("interface_counts", "Interface", "--"),
        ("model_counts", "Total fit", "-"),
    ]
    for key, label, style in curves:
        if key in fit and fit[key] is not None:
            axis.plot(x, np.asarray(fit[key]), style, linewidth=2, label=label)
    axis.set_xlabel("Voxel density")
    axis.set_ylabel("Average number of voxels")
    axis.set_title(title or "Averaged voxel histogram and phase fit")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure


def plot_log_dataframe(
    dataframe,
    quantities,
    x="run_step",
    skip_initial_by_quantity: dict[str, int] | None = None,
):
    """Plot selected synchronized HDF5 log quantities."""

    import matplotlib.pyplot as plt

    quantities = [quantities] if isinstance(quantities, str) else list(quantities)
    if not quantities:
        raise ValueError("quantities cannot be empty")
    missing = [name for name in [x, *quantities] if name not in dataframe.columns]
    if missing:
        raise KeyError(f"Log columns are unavailable: {missing}")
    figure, axes = plt.subplots(
        len(quantities),
        1,
        figsize=(9, 2.8 * len(quantities)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    skip_initial_by_quantity = skip_initial_by_quantity or {}
    for axis, quantity in zip(axes, quantities):
        skip = int(skip_initial_by_quantity.get(quantity, 0))
        if skip < 0:
            raise ValueError("Initial log points to skip cannot be negative")
        plotted = dataframe.iloc[skip:]
        axis.plot(plotted[x], plotted[quantity])
        axis.set_ylabel(quantity)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel(x)
    figure.tight_layout()
    plt.show()
    return figure
