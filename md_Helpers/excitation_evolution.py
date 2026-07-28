"""Two-segment excitation evolution, stitching, and legacy archiving."""

from pathlib import Path
import shutil

import gsd.hoomd
import h5py
import numpy as np

from . import metadata as metadata_helpers
from . import runs as run_helpers
from . import simulation as simulation_helpers
from .paths import (
    EXCITATION_EVOLVED_V3_LEGACY_ROOT,
    EXCITATION_EVOLVED_V3_ROOT,
    excitation_evolved_paths,
)
from .run_logs import simulation_progress


DEFAULT_DT1 = 0.0005
DEFAULT_NSTEPS1 = 200_000
EVOLUTION_FORMAT = "two_segment_dt_v1"
TIMESTEP_DATASET = "hoomd-data/Simulation/timestep"
FORMAT_MARKER_NAME = ".two_segment_dt_v1"
_VALIDATED_ROOTS = set()


def _as_paths(result_or_paths):
    if isinstance(result_or_paths, (str, Path)):
        manifest_path = Path(result_or_paths)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Evolution manifest does not exist: {manifest_path}"
            )
        with h5py.File(manifest_path, mode="r") as hdf:
            run = {
                key: metadata_helpers.clean_read_value(value)
                for key, value in hdf["metadata/run"].attrs.items()
            }
            path_attrs = {
                key: metadata_helpers.clean_read_value(value)
                for key, value in hdf["metadata/paths"].attrs.items()
            }
            segment_attrs = {}
            for segment_index in [1, 2]:
                group = hdf[
                    f"metadata/segments/segment_{segment_index}"
                ]
                segment_attrs[segment_index] = {
                    key: metadata_helpers.clean_read_value(value)
                    for key, value in group.attrs.items()
                }

        def make_segment(segment_index):
            prefix = f"segment_{segment_index}"
            attrs = segment_attrs[segment_index]
            return {
                "segment_index": segment_index,
                "folder": Path(path_attrs[f"{prefix}_log_path"]).parent,
                "dt": float(attrs["dt"]),
                "nsteps": int(attrs["nsteps"]),
                "trajectory_path": Path(
                    path_attrs[f"{prefix}_trajectory_path"]
                ),
                "final_state_path": Path(
                    path_attrs[f"{prefix}_final_state_path"]
                ),
                "log_path": Path(path_attrs[f"{prefix}_log_path"]),
                "state_kind": "excitation_evolved_segment",
            }

        segment_1 = make_segment(1)
        segment_2 = make_segment(2)
        return {
            "folder": manifest_path.parent,
            "manifest_path": manifest_path,
            "segment_1": segment_1,
            "segment_2": segment_2,
            "segment_paths": [segment_1, segment_2],
            "trajectory_paths": [
                segment_1["trajectory_path"],
                segment_2["trajectory_path"],
            ],
            "log_paths": [
                segment_1["log_path"],
                segment_2["log_path"],
            ],
            "trajectory_path": segment_2["trajectory_path"],
            "final_state_path": segment_2["final_state_path"],
            "log_path": segment_2["log_path"],
            "dt1": float(segment_1["dt"]),
            "nsteps1": int(segment_1["nsteps"]),
            "dt2": float(segment_2["dt"]),
            "nsteps2": int(segment_2["nsteps"]),
            "total_nsteps": int(run["total_nsteps"]),
            "total_physical_time": float(run["total_physical_time"]),
            "state_kind": "excitation_evolved",
            "evolution_format": run["evolution_format"],
        }

    if not isinstance(result_or_paths, dict):
        raise TypeError(
            "Expected a manifest path, result dictionary, or paths dictionary."
        )

    paths = result_or_paths.get("paths", result_or_paths)
    if "segment_1" not in paths or "segment_2" not in paths:
        raise ValueError(
            "The supplied object does not contain two-segment excitation paths."
        )
    return paths


def _segment_is_complete(segment_paths):
    return all(
        Path(segment_paths[key]).exists()
        for key in ["trajectory_path", "final_state_path", "log_path"]
    )


def ensure_two_segment_root(base_folder=EXCITATION_EVOLVED_V3_ROOT):
    """
    Ensure the active root does not still contain legacy single-dt results.

    A marker avoids rescanning the result tree during parameter sweeps.
    """

    base_folder = Path(base_folder)
    resolved = str(base_folder.resolve())
    if resolved in _VALIDATED_ROOTS:
        return base_folder

    marker_path = base_folder / FORMAT_MARKER_NAME
    if marker_path.exists():
        _VALIDATED_ROOTS.add(resolved)
        return base_folder

    if base_folder.exists():
        for log_path in base_folder.rglob("excitation_log.hdf5"):
            if log_path.parent.name not in {"segment_1", "segment_2"}:
                raise RuntimeError(
                    "Excitation_Evolved_v3 still contains legacy single-dt "
                    "results. Run archive_legacy_excitation_evolved() before "
                    "creating two-segment results. First legacy log found: "
                    f"{log_path}"
                )

    base_folder.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(f"{EVOLUTION_FORMAT}\n", encoding="utf-8")
    _VALIDATED_ROOTS.add(resolved)
    return base_folder


def excitation_paths_from_manifest(manifest_path):
    """Reconstruct a two-segment paths dictionary from its saved manifest."""

    return _as_paths(manifest_path)


def _all_outputs_complete(paths):
    return (
        Path(paths["manifest_path"]).exists()
        and _segment_is_complete(paths["segment_1"])
        and _segment_is_complete(paths["segment_2"])
    )


def _frame_step(path, frame_index=-1):
    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        if len(trajectory) == 0:
            raise ValueError(f"GSD file contains no frames: {path}")
        return int(trajectory[frame_index].configuration.step)


def _load_frame(path, frame_index=-1):
    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        if len(trajectory) == 0:
            raise ValueError(f"GSD file contains no frames: {path}")
        return trajectory[frame_index]


def _manifest_groups(
    paths,
    status,
    evolve_seed,
    segment_timing=None,
    common_metadata=None,
):
    segment_timing = segment_timing or {}
    common_metadata = common_metadata or {}
    segment_1 = paths["segment_1"]
    segment_2 = paths["segment_2"]

    run = {
        "run_kind": "hot_spike_two_segment_evolution",
        "phase_name": "hot_spike",
        "evolution_format": EVOLUTION_FORMAT,
        "status": str(status),
        "n_segments": 2,
        "seed": int(evolve_seed),
        "dt1": float(paths["dt1"]),
        "nsteps1": int(paths["nsteps1"]),
        "dt2": float(paths["dt2"]),
        "nsteps2": int(paths["nsteps2"]),
        "total_nsteps": int(paths["total_nsteps"]),
        "total_physical_time": float(paths["total_physical_time"]),
        "physical_time_units": "reduced_lj_time",
    }

    path_attrs = {
        "manifest_path": str(paths["manifest_path"]),
        "segment_1_log_path": str(segment_1["log_path"]),
        "segment_1_trajectory_path": str(segment_1["trajectory_path"]),
        "segment_1_final_state_path": str(segment_1["final_state_path"]),
        "segment_2_log_path": str(segment_2["log_path"]),
        "segment_2_trajectory_path": str(segment_2["trajectory_path"]),
        "segment_2_final_state_path": str(segment_2["final_state_path"]),
        "final_state_path": str(paths["final_state_path"]),
    }

    groups = {
        "metadata/run": run,
        "metadata/paths": path_attrs,
        "metadata/segments/segment_1": {
            "segment_index": 1,
            "dt": float(segment_1["dt"]),
            "nsteps": int(segment_1["nsteps"]),
            "physical_time": (
                float(segment_1["dt"]) * int(segment_1["nsteps"])
            ),
            **segment_timing.get(1, {}),
        },
        "metadata/segments/segment_2": {
            "segment_index": 2,
            "dt": float(segment_2["dt"]),
            "nsteps": int(segment_2["nsteps"]),
            "physical_time": (
                float(segment_2["dt"]) * int(segment_2["nsteps"])
            ),
            **segment_timing.get(2, {}),
        },
    }

    for group_path in [
        "metadata/state",
        "metadata/source",
        "metadata/creation",
        "metadata/lj",
    ]:
        if common_metadata.get(group_path):
            groups[group_path] = common_metadata[group_path]

    return groups


def write_evolution_manifest(
    paths,
    status,
    evolve_seed,
    segment_timing=None,
    common_metadata=None,
):
    """Write or update the authoritative two-segment evolution manifest."""

    paths = _as_paths(paths)
    groups = _manifest_groups(
        paths=paths,
        status=status,
        evolve_seed=evolve_seed,
        segment_timing=segment_timing,
        common_metadata=common_metadata,
    )
    metadata_helpers.write_metadata_groups(
        hdf5_path=paths["manifest_path"],
        groups=groups,
        mode="a",
        overwrite=True,
    )
    metadata_helpers.clear_attrs(paths["manifest_path"], "metadata")
    return Path(paths["manifest_path"])


def read_evolution_manifest(manifest_or_result):
    """Read the manifest into a dictionary keyed by metadata group."""

    if isinstance(manifest_or_result, dict):
        manifest_path = _as_paths(manifest_or_result)["manifest_path"]
    else:
        manifest_path = Path(manifest_or_result)

    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Evolution manifest does not exist: {manifest_path}")

    groups = {}
    with h5py.File(manifest_path, mode="r") as hdf:
        if "metadata" not in hdf:
            return groups

        def collect(name, obj):
            if isinstance(obj, h5py.Group) and obj.attrs:
                groups[name] = {
                    key: metadata_helpers.clean_read_value(value)
                    for key, value in obj.attrs.items()
                }

        hdf["metadata"].visititems(
            lambda name, obj: collect(f"metadata/{name}", obj)
        )
    return groups


def validate_segment_continuity(result_or_paths):
    """Verify timestep, box, positions, and velocities at the boundary."""

    paths = _as_paths(result_or_paths)
    segment_1_final = _load_frame(
        paths["segment_1"]["final_state_path"]
    )
    segment_2_initial = _load_frame(
        paths["segment_2"]["trajectory_path"],
        frame_index=0,
    )
    segment_1_final_step = int(segment_1_final.configuration.step)
    segment_2_initial_step = int(segment_2_initial.configuration.step)
    if segment_1_final_step != segment_2_initial_step:
        raise ValueError(
            "Excitation segments are discontinuous: segment 1 ends at "
            f"{segment_1_final_step}, but segment 2 begins at "
            f"{segment_2_initial_step}."
        )

    if int(segment_1_final.particles.N) != int(segment_2_initial.particles.N):
        raise ValueError("Particle count changes at the segment boundary.")

    comparisons = {
        "box": (
            np.asarray(segment_1_final.configuration.box),
            np.asarray(segment_2_initial.configuration.box),
        ),
        "positions": (
            np.asarray(segment_1_final.particles.position),
            np.asarray(segment_2_initial.particles.position),
        ),
        "velocities": (
            np.asarray(segment_1_final.particles.velocity),
            np.asarray(segment_2_initial.particles.velocity),
        ),
    }
    max_differences = {}
    for name, (left, right) in comparisons.items():
        if left.shape != right.shape or not np.allclose(
            left,
            right,
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ValueError(
                f"{name.capitalize()} are discontinuous at the segment boundary."
            )
        max_differences[f"max_{name}_difference"] = (
            float(np.max(np.abs(left - right))) if left.size else 0.0
        )

    return {
        "segment_1_final_timestep": segment_1_final_step,
        "segment_2_initial_timestep": segment_2_initial_step,
        "continuous": True,
        **max_differences,
    }


def _segment_metadata_groups(
    hot_spike_helpers,
    simulation,
    initial_result,
    segment_paths,
    segment_index,
    paths,
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    source_seed,
    evolve_seed,
    log_period,
    trajectory_period,
    lj_kwargs,
):
    groups = hot_spike_helpers._build_evolution_metadata_groups(
        simulation=simulation,
        initial_result=initial_result,
        evolved_paths=segment_paths,
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        source_kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        evolve_nsteps=segment_paths["nsteps"],
        evolve_seed=evolve_seed,
        log_period=log_period,
        trajectory_period=trajectory_period,
        lj_kwargs=lj_kwargs,
    )
    groups["metadata/run"].update({
        "run_kind": "hot_spike_evolution_segment",
        "evolution_format": EVOLUTION_FORMAT,
        "segment_index": int(segment_index),
        "n_segments": 2,
        "segment_physical_time": (
            float(segment_paths["dt"]) * int(segment_paths["nsteps"])
        ),
    })
    groups["metadata/paths"].update({
        "manifest_path": str(paths["manifest_path"]),
        "overall_final_state_path": str(paths["final_state_path"]),
    })
    return groups


def get_or_create_two_segment_hot_spike(
    n_fcc_cells,
    target_rho,
    kT,
    source_nsteps,
    radius,
    injected_energy,
    dt2,
    nsteps2,
    method="velocity_rescale_com",
    source_seed=1,
    evolve_seed=1,
    dt1=DEFAULT_DT1,
    nsteps1=DEFAULT_NSTEPS1,
    source_log_period=1_000,
    log_period=1_000,
    trajectory_period=1_000,
    random_location=False,
    overwrite=False,
    overwrite_initial=False,
    overwrite_source=False,
    create_source_if_missing=True,
    reject_phase_separated_source=True,
    base_folder=EXCITATION_EVOLVED_V3_ROOT,
):
    """Run or load a hot-spike NVE evolution with exactly two timestep sizes."""

    # Local import avoids a module cycle while keeping initial-state ownership
    # in hot_spike.py.
    from . import hot_spike as hot_spike_helpers
    from . import classification as classification_helpers

    ensure_two_segment_root(base_folder)

    initial_result = hot_spike_helpers.get_or_create_hot_spike_state(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        radius=radius,
        injected_energy=injected_energy,
        method=method,
        source_seed=source_seed,
        source_log_period=source_log_period,
        random_location=random_location,
        overwrite=overwrite_initial,
        overwrite_source=overwrite_source,
        create_source_if_missing=create_source_if_missing,
        reject_phase_separated_source=reject_phase_separated_source,
    )

    source_metadata_seed = int(source_seed)
    if initial_result["frame"] is not None:
        source_metadata_seed = hot_spike_helpers._source_seed_from_metadata(
            source_result=initial_result["source_result"],
            fallback_seed=source_seed,
        )

    paths = excitation_evolved_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=target_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        method=method,
        radius=radius,
        energy=injected_energy,
        dt1=dt1,
        nsteps1=nsteps1,
        dt2=dt2,
        nsteps2=nsteps2,
        evolve_seed=evolve_seed,
        center=None,
        random_location=random_location,
        excitation_seed=source_metadata_seed,
        base_folder=base_folder,
    )

    if initial_result["frame"] is None:
        return {
            "frame": None,
            "paths": paths,
            "initial_result": initial_result,
            "created_new": False,
            "status": initial_result.get("status", "missing_source"),
        }

    if _all_outputs_complete(paths) and not overwrite:
        continuity = validate_segment_continuity(paths)
        segment_timing = {}
        for segment_index in [1, 2]:
            segment = paths[f"segment_{segment_index}"]
            segment_timing[segment_index] = {
                "start_timestep": _frame_step(
                    segment["trajectory_path"],
                    frame_index=0,
                ),
                "final_timestep": _frame_step(segment["final_state_path"]),
                "status": "loaded",
            }
        write_evolution_manifest(
            paths=paths,
            status="complete",
            evolve_seed=evolve_seed,
            segment_timing=segment_timing,
        )
        metadata_helpers.write_metadata_groups(
            paths["manifest_path"],
            {"metadata/continuity": continuity},
            mode="a",
            overwrite=True,
        )
        final_frame = _load_frame(paths["final_state_path"])
        classification_result = classification_helpers.read_phase_method_attrs(
            paths["log_path"],
            "voxel",
        )
        return {
            "frame": final_frame,
            "paths": paths,
            "initial_result": initial_result,
            "classification_result": classification_result,
            "continuity": continuity,
            "created_new": False,
            "status": "loaded_two_segment_evolution",
        }

    lj_kwargs = hot_spike_helpers._source_lj_kwargs(
        initial_result["source_result"]
    )
    common_metadata = {}
    segment_timing = {}
    segment_results = []
    current_frame = initial_result["frame"]
    active_simulation = None
    upstream_changed = bool(overwrite)

    write_evolution_manifest(
        paths=paths,
        status="running",
        evolve_seed=evolve_seed,
    )

    for segment_index in [1, 2]:
        segment_paths = paths[f"segment_{segment_index}"]
        segment_complete = _segment_is_complete(segment_paths)

        if segment_complete and not upstream_changed:
            current_frame = _load_frame(segment_paths["final_state_path"])
            active_simulation = None
            start_timestep = _frame_step(
                segment_paths["trajectory_path"],
                frame_index=0,
            )
            final_timestep = int(current_frame.configuration.step)
            segment_timing[segment_index] = {
                "start_timestep": start_timestep,
                "final_timestep": final_timestep,
                "status": "loaded",
            }
            segment_results.append({
                "segment_index": segment_index,
                "paths": segment_paths,
                "created_new": False,
                "status": "loaded",
            })
            continue

        continued_live = segment_index == 2 and active_simulation is not None
        if continued_live:
            simulation = active_simulation
            simulation.operations.integrator.dt = float(segment_paths["dt"])
            if hasattr(simulation, "metadata"):
                simulation.metadata["dt"] = float(segment_paths["dt"])
        else:
            simulation = simulation_helpers.make_simulation(
                frame=current_frame,
                target_rho=target_rho,
                n_fcc_cells=n_fcc_cells,
                seed=evolve_seed,
                dt=segment_paths["dt"],
                kT=kT,
                ensemble="NVE",
                starting_state_path=(
                    str(initial_result["paths"]["state_path"])
                    if segment_index == 1
                    else str(paths["segment_1"]["final_state_path"])
                ),
                **lj_kwargs,
            )
        integrator_metadata = hot_spike_helpers.infer_integrator_metadata(
            simulation
        )
        if integrator_metadata["ensemble"] != "NVE":
            raise RuntimeError(
                "Hot-spike evolution must be NVE, but actual ensemble was "
                f"{integrator_metadata['ensemble']}."
            )

        start_timestep = int(simulation.timestep)
        metadata_groups = _segment_metadata_groups(
            hot_spike_helpers=hot_spike_helpers,
            simulation=simulation,
            initial_result=initial_result,
            segment_paths=segment_paths,
            segment_index=segment_index,
            paths=paths,
            n_fcc_cells=n_fcc_cells,
            target_rho=target_rho,
            kT=kT,
            source_nsteps=source_nsteps,
            source_seed=source_seed,
            evolve_seed=evolve_seed,
            log_period=log_period,
            trajectory_period=trajectory_period,
            lj_kwargs=lj_kwargs,
        )
        metadata_groups["metadata/run"]["continuation_mode"] = (
            "live_integrator_dt_change"
            if continued_live
            else "created_from_saved_or_initial_frame"
        )
        if not common_metadata:
            common_metadata = {
                key: value
                for key, value in metadata_groups.items()
                if key in {
                    "metadata/state",
                    "metadata/source",
                    "metadata/creation",
                    "metadata/lj",
                }
            }

        with simulation_progress(
            f"Excitation segment {segment_index}",
            ncells=n_fcc_cells,
            rho=target_rho,
            Source_kT=kT,
            nsteps=segment_paths["nsteps"],
        ):
            run_result = run_helpers.run_logged_trajectory_phase(
                simulation=simulation,
                nsteps=segment_paths["nsteps"],
                log_path=segment_paths["log_path"],
                trajectory_path=segment_paths["trajectory_path"],
                final_state_path=segment_paths["final_state_path"],
                log_period=log_period,
                trajectory_period=trajectory_period,
                metadata_groups=metadata_groups,
                classify_final=(segment_index == 2),
                classification_kwargs=None,
            )

        current_frame = _load_frame(segment_paths["final_state_path"])
        active_simulation = simulation
        final_timestep = int(current_frame.configuration.step)
        segment_timing[segment_index] = {
            "start_timestep": start_timestep,
            "final_timestep": final_timestep,
            "status": "created",
            "continuation_mode": metadata_groups["metadata/run"][
                "continuation_mode"
            ],
        }
        segment_results.append({
            "segment_index": segment_index,
            "paths": segment_paths,
            "run_result": run_result,
            "created_new": True,
            "status": "created",
        })
        upstream_changed = True
        write_evolution_manifest(
            paths=paths,
            status=f"segment_{segment_index}_complete",
            evolve_seed=evolve_seed,
            segment_timing=segment_timing,
            common_metadata=common_metadata,
        )

    continuity = validate_segment_continuity(paths)
    write_evolution_manifest(
        paths=paths,
        status="complete",
        evolve_seed=evolve_seed,
        segment_timing=segment_timing,
        common_metadata=common_metadata,
    )
    metadata_helpers.write_metadata_groups(
        paths["manifest_path"],
        {"metadata/continuity": continuity},
        mode="a",
        overwrite=True,
    )
    classification_result = classification_helpers.read_phase_method_attrs(
        paths["log_path"],
        "voxel",
    )

    return {
        "frame": current_frame,
        "paths": paths,
        "initial_result": initial_result,
        "segment_results": segment_results,
        "run_result": segment_results[-1].get("run_result", {}),
        "classification_result": classification_result,
        "continuity": continuity,
        "created_new": any(item["created_new"] for item in segment_results),
        "status": "created_two_segment_evolution",
    }


def _manifest_segment_timing(paths):
    manifest_path = Path(paths["manifest_path"])
    if manifest_path.exists():
        groups = read_evolution_manifest(manifest_path)
    else:
        groups = {}

    timing = {}
    for segment_index in [1, 2]:
        segment = paths[f"segment_{segment_index}"]
        attrs = groups.get(
            f"metadata/segments/segment_{segment_index}",
            {},
        )
        start_timestep = attrs.get("start_timestep")
        if start_timestep is None:
            start_timestep = _frame_step(
                segment["trajectory_path"],
                frame_index=0,
            )
        timing[segment_index] = {
            "dt": float(segment["dt"]),
            "nsteps": int(segment["nsteps"]),
            "start_timestep": int(start_timestep),
        }
    return timing


def iter_stitched_trajectory(result_or_paths, stride=1, max_frames=None):
    """
    Yield both GSD trajectories as one sequence with piecewise physical time.

    Each yielded item contains ``frame``, ``frame_index``, ``segment_index``,
    ``timestep``, and ``elapsed_time``. The duplicate boundary frame is removed.
    """

    paths = _as_paths(result_or_paths)
    timing = _manifest_segment_timing(paths)
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")

    global_index = 0
    yielded = 0
    previous_timestep = None
    time_offset = 0.0

    for segment_index in [1, 2]:
        segment = paths[f"segment_{segment_index}"]
        segment_timing = timing[segment_index]
        with gsd.hoomd.open(
            name=str(segment["trajectory_path"]),
            mode="r",
        ) as trajectory:
            for local_index in range(len(trajectory)):
                frame = trajectory[local_index]
                timestep = int(frame.configuration.step)
                if (
                    segment_index == 2
                    and local_index == 0
                    and timestep == previous_timestep
                ):
                    continue

                selected = global_index % stride == 0
                global_index += 1
                previous_timestep = timestep
                if not selected:
                    continue

                elapsed_time = time_offset + (
                    timestep - segment_timing["start_timestep"]
                ) * segment_timing["dt"]
                yield {
                    "frame": frame,
                    "frame_index": yielded,
                    "local_frame_index": local_index,
                    "segment_index": segment_index,
                    "timestep": timestep,
                    "elapsed_time": float(elapsed_time),
                }
                yielded += 1
                if max_frames is not None and yielded >= int(max_frames):
                    return

        time_offset += (
            segment_timing["nsteps"] * segment_timing["dt"]
        )


def load_stitched_trajectory_frames(
    result_or_paths,
    stride=1,
    max_frames=None,
    sample_by_physical_time=True,
):
    """
    Load selected frames spanning the complete two-segment trajectory.

    When ``max_frames`` limits the result, the default selection is uniform in
    piecewise physical time so animations cover both segments at a steady
    physical-time pace.
    """

    paths = _as_paths(result_or_paths)
    timing = _manifest_segment_timing(paths)
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")

    rows = []
    previous_timestep = None
    global_index = 0
    time_offset = 0.0

    for segment_index in [1, 2]:
        segment = paths[f"segment_{segment_index}"]
        segment_timing = timing[segment_index]
        with gsd.hoomd.open(
            name=str(segment["trajectory_path"]),
            mode="r",
        ) as trajectory:
            for local_index in range(len(trajectory)):
                timestep = int(
                    trajectory[local_index].configuration.step
                )
                if (
                    segment_index == 2
                    and local_index == 0
                    and timestep == previous_timestep
                ):
                    continue
                previous_timestep = timestep

                if global_index % stride == 0:
                    rows.append({
                        "segment_index": segment_index,
                        "local_frame_index": local_index,
                        "timestep": timestep,
                        "elapsed_time": float(
                            time_offset
                            + (
                                timestep
                                - segment_timing["start_timestep"]
                            ) * segment_timing["dt"]
                        ),
                    })
                global_index += 1
        time_offset += (
            segment_timing["nsteps"] * segment_timing["dt"]
        )

    if max_frames is not None and len(rows) > int(max_frames):
        max_frames = int(max_frames)
        if max_frames <= 0:
            return []
        if sample_by_physical_time:
            elapsed = np.asarray(
                [row["elapsed_time"] for row in rows],
                dtype=np.float64,
            )
            targets = np.linspace(elapsed[0], elapsed[-1], max_frames)
            right = np.searchsorted(elapsed, targets, side="left")
            right = np.clip(right, 0, len(rows) - 1)
            left = np.clip(right - 1, 0, len(rows) - 1)
            choose_left = (
                np.abs(elapsed[left] - targets)
                <= np.abs(elapsed[right] - targets)
            )
            selected = np.where(choose_left, left, right)
            selected = np.unique(selected)
        else:
            selected = np.linspace(
                0,
                len(rows) - 1,
                max_frames,
                dtype=int,
            )
        rows = [rows[index] for index in selected]

    loaded = []
    for segment_index in [1, 2]:
        selected_rows = [
            row for row in rows
            if row["segment_index"] == segment_index
        ]
        if not selected_rows:
            continue
        segment = paths[f"segment_{segment_index}"]
        with gsd.hoomd.open(
            name=str(segment["trajectory_path"]),
            mode="r",
        ) as trajectory:
            for row in selected_rows:
                loaded.append({
                    **row,
                    "frame": trajectory[row["local_frame_index"]],
                })

    loaded.sort(key=lambda item: item["elapsed_time"])
    for frame_index, item in enumerate(loaded):
        item["frame_index"] = frame_index
    return loaded


def write_stitched_trajectory(
    result_or_paths,
    output_path=None,
    overwrite=False,
):
    """Materialize a duplicate-free combined GSD for legacy readers."""

    paths = _as_paths(result_or_paths)
    output_path = Path(
        output_path or (Path(paths["folder"]) / "excitation_trajectory_stitched.gsd")
    )
    source_paths = {
        Path(path).resolve()
        for path in paths["trajectory_paths"]
    }
    if output_path.resolve() in source_paths:
        raise ValueError("Stitched trajectory output cannot replace a segment.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Stitched trajectory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(output_path), mode="w") as output:
        for item in iter_stitched_trajectory(paths):
            output.append(item["frame"])
    return output_path


def _collect_datasets(group, prefix="", output=None):
    output = output if output is not None else {}
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            output[path] = np.asarray(item)
        elif isinstance(item, h5py.Group):
            _collect_datasets(item, path, output)
    return output


def _nested_set(container, path, value):
    parts = path.split("/")
    current = container
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def read_stitched_log(result_or_paths):
    """
    Read both segment logs as one nested log dictionary.

    The usual HOOMD datasets are concatenated. ``stitched`` adds aligned
    ``elapsed_time``, ``segment_index``, and ``source_row_index`` arrays.
    """

    paths = _as_paths(result_or_paths)
    timing = _manifest_segment_timing(paths)
    segment_datasets = []
    timesteps = []

    for segment_index in [1, 2]:
        log_path = Path(paths[f"segment_{segment_index}"]["log_path"])
        with h5py.File(log_path, mode="r") as hdf:
            datasets = _collect_datasets(hdf)
        if TIMESTEP_DATASET not in datasets:
            raise KeyError(f"{log_path} is missing {TIMESTEP_DATASET}")
        steps = np.asarray(datasets[TIMESTEP_DATASET], dtype=np.int64)
        segment_datasets.append(datasets)
        timesteps.append(steps)

    keep_1 = np.ones(timesteps[0].shape[0], dtype=bool)
    keep_2 = np.ones(timesteps[1].shape[0], dtype=bool)
    if timesteps[0].size and timesteps[1].size:
        if timesteps[1][0] == timesteps[0][-1]:
            keep_2[0] = False

    keeps = [keep_1, keep_2]
    common_paths = set(segment_datasets[0]).intersection(segment_datasets[1])
    combined = {}

    for dataset_path in sorted(common_paths):
        pieces = []
        valid = True
        for datasets, steps, keep in zip(segment_datasets, timesteps, keeps):
            values = np.asarray(datasets[dataset_path])
            if values.ndim == 0 or values.shape[0] != steps.shape[0]:
                valid = False
                break
            pieces.append(values[keep])
        if valid:
            combined[dataset_path] = np.concatenate(pieces, axis=0)

    kept_steps = [
        steps[keep]
        for steps, keep in zip(timesteps, keeps)
    ]
    elapsed_pieces = []
    segment_pieces = []
    row_pieces = []
    time_offset = 0.0
    for segment_index, (steps, keep) in enumerate(
        zip(timesteps, keeps),
        start=1,
    ):
        kept = steps[keep]
        segment_timing = timing[segment_index]
        elapsed_pieces.append(
            time_offset
            + (
                kept - segment_timing["start_timestep"]
            ) * segment_timing["dt"]
        )
        segment_pieces.append(
            np.full(kept.shape, segment_index, dtype=np.int8)
        )
        row_pieces.append(np.flatnonzero(keep).astype(np.int64))
        time_offset += (
            segment_timing["nsteps"] * segment_timing["dt"]
        )

    output = {}
    for dataset_path, values in combined.items():
        _nested_set(output, dataset_path, values)
    output["stitched"] = {
        "timestep": np.concatenate(kept_steps),
        "elapsed_time": np.concatenate(elapsed_pieces),
        "segment_index": np.concatenate(segment_pieces),
        "source_row_index": np.concatenate(row_pieces),
        "dt1": float(paths["dt1"]),
        "dt2": float(paths["dt2"]),
        "boundary_duplicate_removed": bool(not keep_2[0])
        if keep_2.size
        else False,
    }
    manifest_groups = read_evolution_manifest(paths["manifest_path"])
    for group_path, attrs in manifest_groups.items():
        _nested_set(output, group_path, {"attrs": attrs})
    output.setdefault("metadata", {})["attrs"] = {
        "evolution_format": EVOLUTION_FORMAT,
        "manifest_path": str(paths["manifest_path"]),
        "total_physical_time": float(paths["total_physical_time"]),
    }
    return output


def write_stitched_log(result_or_paths, output_path=None, overwrite=False):
    """Materialize the stitched log into an analysis-friendly HDF5 file."""

    paths = _as_paths(result_or_paths)
    output_path = Path(
        output_path or (Path(paths["folder"]) / "excitation_log_stitched.hdf5")
    )
    source_paths = {
        Path(path).resolve()
        for path in paths["log_paths"]
    }
    if output_path.resolve() in source_paths:
        raise ValueError("Stitched log output cannot replace a segment.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Stitched log already exists: {output_path}")

    stitched = read_stitched_log(paths)
    datasets = {}

    def collect(container, prefix=""):
        for key, value in container.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                if key in {"metadata", "attrs"}:
                    continue
                collect(value, path)
            elif isinstance(value, np.ndarray):
                datasets[path] = value

    collect(stitched)
    metadata_helpers.write_datasets(
        hdf5_path=output_path,
        datasets=datasets,
        mode="w",
        overwrite=True,
    )
    manifest_groups = read_evolution_manifest(paths["manifest_path"])
    manifest_groups.setdefault("metadata/run", {}).update(
        stitched["metadata"]["attrs"]
    )
    manifest_groups.setdefault("metadata/paths", {}).update({
        "manifest_path": str(paths["manifest_path"]),
        "segment_1_log_path": str(paths["segment_1"]["log_path"]),
        "segment_2_log_path": str(paths["segment_2"]["log_path"]),
    })
    metadata_helpers.write_metadata_groups(
        hdf5_path=output_path,
        groups=manifest_groups,
        mode="a",
        overwrite=True,
    )
    return output_path


def archive_legacy_excitation_evolved(
    source_root=EXCITATION_EVOLVED_V3_ROOT,
    archive_root=EXCITATION_EVOLVED_V3_LEGACY_ROOT,
    dry_run=True,
):
    """
    Move the old single-dt root aside so V3 can use the two-segment format.

    The operation refuses to merge with or overwrite an existing archive.
    """

    source_root = Path(source_root)
    archive_root = Path(archive_root)
    result = {
        "source_root": source_root,
        "archive_root": archive_root,
        "dry_run": bool(dry_run),
        "moved": False,
    }

    if not source_root.exists():
        result["status"] = "source_missing"
        return result
    if (source_root / FORMAT_MARKER_NAME).exists():
        raise RuntimeError(
            "The source root is already marked as the two-segment V3 format; "
            "it will not be archived as legacy data."
        )
    if archive_root.exists():
        raise FileExistsError(
            f"Legacy archive destination already exists: {archive_root}"
        )
    if source_root.resolve() == archive_root.resolve():
        raise ValueError("Source and archive roots must be different")

    file_count = sum(1 for path in source_root.rglob("*") if path.is_file())
    result["file_count"] = file_count
    if dry_run:
        result["status"] = "would_move"
        return result

    archive_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_root), str(archive_root))
    source_root.mkdir(parents=True, exist_ok=False)
    (source_root / FORMAT_MARKER_NAME).write_text(
        f"{EVOLUTION_FORMAT}\n",
        encoding="utf-8",
    )
    result["moved"] = True
    result["status"] = "moved_and_created_empty_v3_root"
    return result
