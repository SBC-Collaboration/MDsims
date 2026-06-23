# cavitation.py

from pathlib import Path

import numpy as np
import gsd.hoomd

from .paths import CAVITATION_STATES_ROOT
from . import simulation as sh
from . import runs as lh




# ============================================================
# Formatting helpers
# ============================================================

def _format_float(
    value,
    decimals=3,
):
    """
    Format floats consistently for folder names.

    Example:
        0.8  -> "0.800"
        0.10 -> "0.100"
    """

    return f"{float(value):.{decimals}f}"


# ============================================================
# Build cavitation state path
# ============================================================

def get_cavitation_state_path(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius_fraction,
    source_seed=1,
    source_phase_name="randomization",
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
    base_folder=CAVITATION_STATES_ROOT,
):
    """
    Build the standard path for an initial cavitated GSD state.

    Bubble-radius convention:
        bubble_radius = radius_fraction * (BoxLength / 2)

    This only builds the path. It does not check whether the file exists.
    """

    n_cells_str = f"{int(n_fcc_cells)}"
    rho_str = _format_float(target_rho)
    kT_str = _format_float(kT)
    source_nsteps_str = f"{int(source_nsteps)}"
    source_seed_str = f"{int(source_seed)}"
    radius_str = _format_float(radius_fraction)

    if random_location:
        if bubble_center is not None:
            raise ValueError(
                "Use either random_location=True or bubble_center, not both."
            )

        center_folder = f"random_center_bubble_seed_{int(bubble_seed)}"

    else:
        if bubble_center is None:
            center_folder = "center_box"

        else:
            bubble_center = np.asarray(
                bubble_center,
                dtype=np.float64,
            )

            if bubble_center.shape != (3,):
                raise ValueError("bubble_center must have shape (3,)")

            center_folder = (
                f"center_x_{_format_float(bubble_center[0])}"
                f"_y_{_format_float(bubble_center[1])}"
                f"_z_{_format_float(bubble_center[2])}"
            )

    folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{n_cells_str}"
        / f"source_rho_{rho_str}"
        / f"kT_{kT_str}"
        / f"source_nsteps_{source_nsteps_str}"
        / f"source_seed_{source_seed_str}"
        / f"source_phase_{source_phase_name}"
        / f"radius_fraction_of_half_box_{radius_str}"
        / center_folder
    )

    state_path = folder / "cavitation.gsd"

    return state_path


# ============================================================
# Load source randomized state
# ============================================================

def get_source_randomization_result(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    source_seed=1,
    source_phase_name="randomization",
    source_log_period=1_000,
    overwrite_source=False,
    require_existing_source=True,
):
    """
    Get the source post-thermalized/randomized state.

    By default, this requires the source state to already exist.

    This prevents cavitation from accidentally launching a long
    thermalization/randomization run when the requested source does not exist.
    """

    # ============================================================
    # Build expected source paths
    # ============================================================
    source_paths = lh.get_phase_paths(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=source_nsteps,
        seed=source_seed,
        phase_name=source_phase_name,
    )

    source_state_path = Path(source_paths["state_path"])
    source_log_path = Path(source_paths["log_path"])

    # ============================================================
    # Require source to already exist
    # ============================================================
    if require_existing_source:
        missing_paths = []

        if not source_state_path.exists():
            missing_paths.append(source_state_path)

        if not source_log_path.exists():
            missing_paths.append(source_log_path)

        if len(missing_paths) > 0:
            print("No source thermalized state found for the specified values.")
            print("=" * 70)
            print("n_fcc_cells       =", n_fcc_cells)
            print("target_rho        =", target_rho)
            print("kT                =", kT)
            print("source_nsteps     =", source_nsteps)
            print("source_seed       =", source_seed)
            print("source_phase_name =", source_phase_name)
            print()
            print("Expected source state:")
            print(source_state_path)
            print()
            print("Expected source log:")
            print(source_log_path)
            print("=" * 70)

            raise FileNotFoundError(
                "No source thermalized state exists for the specified values. "
                "Run/create the source randomization state first."
            )

    # ============================================================
    # Load source using the normal database helper
    # ============================================================
    result = sh.get_or_make_thermalized_state(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=source_nsteps,
        phase_name=source_phase_name,
        log_period=source_log_period,
        seed=source_seed,
        overwrite=overwrite_source,
    )

    return result


# ============================================================
# Load frame from GSD
# ============================================================

def load_frame_from_gsd(
    state_path,
    frame_index=-1,
):
    """
    Load a GSD frame from disk.
    """

    state_path = Path(state_path)

    if not state_path.exists():
        raise FileNotFoundError(f"GSD file does not exist: {state_path}")

    with gsd.hoomd.open(
        name=str(state_path),
        mode="r",
    ) as traj:
        frame = traj[frame_index]

    return frame


# ============================================================
# Save frame to GSD
# ============================================================

def save_frame_to_gsd(
    frame,
    state_path,
    overwrite=False,
):
    """
    Save a single GSD frame.
    """

    state_path = Path(state_path)

    if state_path.exists() and not overwrite:
        raise FileExistsError(
            f"Cavitation state already exists: {state_path}"
        )

    state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with gsd.hoomd.open(
        name=str(state_path),
        mode="w",
    ) as traj:
        traj.append(frame)

    return state_path


# ============================================================
# Choose bubble center
# ============================================================

def choose_bubble_center(
    frame,
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
):
    """
    Choose the bubble center.

    Defaults:
    - centered at the box center, which is (0, 0, 0)
    - random_location=True chooses a uniform random point in the box

    Periodic wrapping is handled later by the minimum-image distance.
    """

    box = np.asarray(
        frame.configuration.box,
        dtype=np.float64,
    )

    Lx = float(box[0])
    Ly = float(box[1])
    Lz = float(box[2])

    box_lengths = np.array(
        [Lx, Ly, Lz],
        dtype=np.float64,
    )

    if random_location:
        if bubble_center is not None:
            raise ValueError(
                "Use either random_location=True or bubble_center, not both."
            )

        rng = np.random.default_rng(
            int(bubble_seed)
        )

        center = rng.uniform(
            low=-0.5 * box_lengths,
            high=0.5 * box_lengths,
        )

    else:
        if bubble_center is None:
            center = np.array(
                [0.0, 0.0, 0.0],
                dtype=np.float64,
            )

        else:
            center = np.asarray(
                bubble_center,
                dtype=np.float64,
            )

            if center.shape != (3,):
                raise ValueError("bubble_center must have shape (3,)")

    return center


# ============================================================
# Minimum-image distances
# ============================================================

def compute_periodic_distances_from_center(
    positions,
    bubble_center,
    box_lengths,
):
    """
    Compute distances from bubble_center using periodic minimum image.

    This lets the bubble wrap correctly across periodic boundaries.
    """

    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    bubble_center = np.asarray(
        bubble_center,
        dtype=np.float64,
    )

    box_lengths = np.asarray(
        box_lengths,
        dtype=np.float64,
    )

    displacements = positions - bubble_center

    displacements = (
        displacements
        - box_lengths * np.round(displacements / box_lengths)
    )

    distances = np.linalg.norm(
        displacements,
        axis=1,
    )

    return distances


# ============================================================
# Copy masked particle fields
# ============================================================

def _copy_masked_particle_fields(
    source_frame,
    new_frame,
    keep_mask,
):
    """
    Copy particle fields from source_frame to new_frame using keep_mask.

    This preserves any per-particle arrays that exist in the source GSD,
    including velocities if they were saved.

    Current project states usually have position and typeid saved.
    """

    source_particles = source_frame.particles
    new_particles = new_frame.particles

    n_before = int(source_particles.N)

    particle_fields = [
        "position",
        "typeid",
        "velocity",
        "mass",
        "charge",
        "diameter",
        "body",
        "image",
        "orientation",
        "moment_inertia",
        "angular_momentum",
    ]

    copied_fields = []

    for field_name in particle_fields:
        try:
            value = getattr(source_particles, field_name)
        except Exception:
            continue

        if value is None:
            continue

        array = np.asarray(value)

        if array.ndim == 0:
            continue

        if array.shape[0] != n_before:
            continue

        setattr(
            new_particles,
            field_name,
            array[keep_mask].copy(),
        )

        copied_fields.append(field_name)

    if "position" not in copied_fields:
        raise RuntimeError(
            "Could not copy particle positions from source frame."
        )

    if "typeid" not in copied_fields:
        new_particles.typeid = np.zeros(
            int(new_particles.N),
            dtype=np.uint32,
        )

    return copied_fields


# ============================================================
# Make cavitated frame from existing frame
# ============================================================

def make_cavitated_frame_from_frame(
    frame,
    radius_fraction,
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
    return_info=False,
):
    """
    Create a cavitated GSD frame by removing particles inside one sphere.

    Bubble-radius convention:
        bubble_radius = radius_fraction * (BoxLength / 2)

    The box is not resized.
    Therefore:
        rho_after = N_after / BoxLength**3

    Particles are removed if:
        periodic_distance_to_center <= bubble_radius
    """

    radius_fraction = float(radius_fraction)

    if radius_fraction <= 0:
        raise ValueError("radius_fraction must be positive")

    # ============================================================
    # Source box and particle data
    # ============================================================
    source_box = np.asarray(
        frame.configuration.box,
        dtype=np.float64,
    )

    Lx = float(source_box[0])
    Ly = float(source_box[1])
    Lz = float(source_box[2])

    box_lengths = np.array(
        [Lx, Ly, Lz],
        dtype=np.float64,
    )

    BoxLength = Lx

    positions = np.asarray(
        frame.particles.position,
        dtype=np.float64,
    )

    N_before = int(frame.particles.N)
    volume = Lx * Ly * Lz
    rho_before = N_before / volume

    # ============================================================
    # Bubble geometry
    # ============================================================
    center = choose_bubble_center(
        frame=frame,
        random_location=random_location,
        bubble_seed=bubble_seed,
        bubble_center=bubble_center,
    )

    bubble_radius = radius_fraction * (BoxLength / 2.0)

    distances = compute_periodic_distances_from_center(
        positions=positions,
        bubble_center=center,
        box_lengths=box_lengths,
    )

    remove_mask = distances <= bubble_radius
    keep_mask = ~remove_mask

    N_removed = int(np.sum(remove_mask))
    N_after = int(np.sum(keep_mask))
    rho_after = N_after / volume

    if N_after <= 0:
        raise RuntimeError(
            "Cavitation removed all particles. "
            "Use a smaller radius_fraction."
        )

    # ============================================================
    # Build new GSD frame
    # ============================================================
    new_frame = gsd.hoomd.Frame()

    new_frame.configuration.step = int(frame.configuration.step)
    new_frame.configuration.box = list(source_box)

    new_frame.particles.N = N_after

    try:
        new_frame.particles.types = list(frame.particles.types)
    except Exception:
        new_frame.particles.types = ["A"]

    copied_fields = _copy_masked_particle_fields(
        source_frame=frame,
        new_frame=new_frame,
        keep_mask=keep_mask,
    )

    info = {
        "bubble_method": "remove_particles_in_sphere",
        "radius_fraction": radius_fraction,
        "radius_definition": "bubble_radius = radius_fraction * (BoxLength / 2)",
        "bubble_radius": float(bubble_radius),
        "bubble_center": center.copy(),
        "random_location": bool(random_location),
        "bubble_seed": int(bubble_seed),

        "BoxLength": float(BoxLength),
        "volume": float(volume),

        "N_before": int(N_before),
        "N_after": int(N_after),
        "particles_removed": int(N_removed),
        "particle_fraction_removed": float(N_removed / N_before),

        "rho_before": float(rho_before),
        "rho_after": float(rho_after),

        "periodic_distance": True,
        "copied_particle_fields": copied_fields,
    }

    if return_info:
        return new_frame, info

    return new_frame


# ============================================================
# Make or load cavitated state
# ============================================================

def make_or_load_cavitated_state(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius_fraction,
    source_seed=1,
    source_phase_name="randomization",
    source_log_period=1_000,
    random_location=False,
    bubble_seed=1,
    bubble_center=None,
    overwrite=False,
    overwrite_source=False,
    require_existing_source=True,
    base_folder=CAVITATION_STATES_ROOT,
    return_info=False,
):
    """
    Main cavitation database helper.

    Workflow:
    1. Use sh.get_or_make_thermalized_state(...) to get the source
       randomized state.
    2. Build the expected Cavitation_States path.
    3. If cavitation.gsd exists and overwrite=False, load it.
    4. Otherwise, remove particles inside the bubble and save cavitation.gsd.
    5. Return the cavitated GSD frame.

    This does not run any dynamics and does not write logs.
    """

    # ============================================================
    # Get source randomized state using existing workflow
    # ============================================================
    source_result = get_source_randomization_result(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
        source_log_period=source_log_period,
        overwrite_source=overwrite_source,
        require_existing_source=require_existing_source,
    )

    source_frame = source_result["frame"]
    source_paths = source_result["paths"]

    source_state_path = Path(source_paths["state_path"])
    source_log_path = Path(source_paths["log_path"])

    # ============================================================
    # Build cavitation path
    # ============================================================
    cavitation_state_path = get_cavitation_state_path(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        radius_fraction=radius_fraction,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
        random_location=random_location,
        bubble_seed=bubble_seed,
        bubble_center=bubble_center,
        base_folder=base_folder,
    )

    # ============================================================
    # Load existing cavitation state if present
    # ============================================================
    if cavitation_state_path.exists() and not overwrite:
        print("Loaded existing cavitation state:")
        print(cavitation_state_path)

        cavitated_frame = load_frame_from_gsd(
            cavitation_state_path,
        )

        info = {
            "created_new": False,
            "cavitation_state_path": str(cavitation_state_path),
            "source_state_path": str(source_state_path),
            "source_log_path": str(source_log_path),
            "N_after": int(cavitated_frame.particles.N),
            "BoxLength": float(cavitated_frame.configuration.box[0]),
            "rho_after": (
                int(cavitated_frame.particles.N)
                / float(cavitated_frame.configuration.box[0])**3
            ),
        }

        if return_info:
            return cavitated_frame, info

        return cavitated_frame

    # ============================================================
    # Create new cavitation frame
    # ============================================================
    cavitated_frame, info = make_cavitated_frame_from_frame(
        frame=source_frame,
        radius_fraction=radius_fraction,
        random_location=random_location,
        bubble_seed=bubble_seed,
        bubble_center=bubble_center,
        return_info=True,
    )

    save_frame_to_gsd(
        frame=cavitated_frame,
        state_path=cavitation_state_path,
        overwrite=overwrite,
    )

    info["created_new"] = True
    info["phase_name"] = "cavitation"

    info["n_fcc_cells"] = int(n_fcc_cells)
    info["source_target_rho"] = float(target_rho)
    info["source_kT"] = float(kT)
    info["source_nsteps"] = int(source_nsteps)
    info["source_seed"] = int(source_seed)
    info["source_phase_name"] = source_phase_name

    info["source_state_path"] = str(source_state_path)
    info["source_log_path"] = str(source_log_path)
    info["cavitation_state_path"] = str(cavitation_state_path)

    print("Created new cavitation state")
    print("=" * 70)
    print("source_state_path:", source_state_path)
    print("cavitation_state_path:", cavitation_state_path)
    print("radius_fraction:", info["radius_fraction"])
    print("bubble_radius:", info["bubble_radius"])
    print("bubble_center:", info["bubble_center"])
    print("N_before:", info["N_before"])
    print("N_after:", info["N_after"])
    print("particles_removed:", info["particles_removed"])
    print("rho_before:", info["rho_before"])
    print("rho_after:", info["rho_after"])
    print("=" * 70)

    if return_info:
        return cavitated_frame, info

    return cavitated_frame
