import numpy as np


def as_snapshot(obj):
    """Return a snapshot-like object from a result, simulation, state, or frame."""

    if isinstance(obj, dict):
        if obj.get("frame") is not None:
            return as_snapshot(obj["frame"])
        if obj.get("simulation") is not None:
            return as_snapshot(obj["simulation"])
        raise TypeError("Result dictionary has no usable frame or simulation.")

    if obj is None:
        raise TypeError("Cannot convert None to a snapshot or frame.")

    if hasattr(obj, "state") and hasattr(obj.state, "get_snapshot"):
        return obj.state.get_snapshot()

    if hasattr(obj, "get_snapshot"):
        return obj.get_snapshot()

    if hasattr(obj, "configuration") and hasattr(obj, "particles"):
        return obj

    raise TypeError(
        "Expected a result dictionary, simulation, state, snapshot, or GSD frame."
    )


def wrap_positions(positions, box_lengths):
    """Wrap positions into the primary periodic box, [-L/2, L/2)."""

    positions = np.asarray(positions, dtype=np.float64)
    box_lengths = np.asarray(box_lengths, dtype=np.float64)

    return (
        (positions + 0.5 * box_lengths)
        % box_lengths
        - 0.5 * box_lengths
    )


def positions_and_box(obj, wrap=True):
    """Return positions, three box lengths, and the snapshot-like object."""

    snapshot = as_snapshot(obj)
    positions = np.asarray(snapshot.particles.position, dtype=np.float64)
    box = np.asarray(snapshot.configuration.box, dtype=np.float64)
    box_lengths = box[:3].astype(np.float64, copy=True)

    if wrap:
        positions = wrap_positions(positions, box_lengths)

    return positions, box_lengths, snapshot


def periodic_distances(positions, center, box_lengths):
    """Return minimum-image distances from positions to one center."""

    positions = np.asarray(positions, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    box_lengths = np.asarray(box_lengths, dtype=np.float64)

    displacements = positions - center
    displacements -= box_lengths * np.round(displacements / box_lengths)
    return np.linalg.norm(displacements, axis=1)


def compute_voxel_densities(obj, nbins=10):
    """Return flattened voxel densities, counts, and one-voxel volume."""

    nbins = int(nbins)
    if nbins <= 0:
        raise ValueError("nbins must be positive")

    positions, box_lengths, _ = positions_and_box(obj, wrap=True)
    bounds = [
        [-box_length / 2.0, box_length / 2.0]
        for box_length in box_lengths
    ]
    voxel_volume = float(np.prod(box_lengths / nbins))
    voxel_counts, _ = np.histogramdd(
        positions,
        bins=nbins,
        range=bounds,
    )
    voxel_counts = voxel_counts.ravel()
    voxel_densities = voxel_counts / voxel_volume
    return voxel_densities, voxel_counts, voxel_volume
