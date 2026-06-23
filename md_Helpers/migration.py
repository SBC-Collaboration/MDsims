from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from .paths import (
    SIMPLE_LATTICES_V2_ROOT,
    THERMALIZED_STATES_V2_ROOT,
    CAVITATION_STATES_ROOT,
)
from . import paths as path_helpers
from . import metadata as metadata_helpers


LATTICE_RE = re.compile(
    r"FCC/n_cells_(?P<n_fcc_cells>\d+)/"
    r"rho_(?P<rho>[0-9.]+)/"
    r"lattice\.gsd$"
)

THERMALIZED_RE = re.compile(
    r"FCC/n_cells_(?P<n_fcc_cells>\d+)/"
    r"rho_(?P<rho>[0-9.]+)/"
    r"kT_(?P<kT>[0-9.]+)/"
    r"nsteps_(?P<nsteps>\d+)/"
    r"seed_(?P<seed>\d+)/"
    r"(?P<phase_name>.+)_log\.hdf5$"
)

CAVITATION_RE = re.compile(
    r"FCC/n_cells_(?P<n_fcc_cells>\d+)/"
    r"source_rho_(?P<rho>[0-9.]+)/"
    r"kT_(?P<kT>[0-9.]+)/"
    r"source_nsteps_(?P<source_nsteps>\d+)/"
    r"source_seed_(?P<source_seed>\d+)/"
    r"source_phase_(?P<source_phase_name>[^/]+)/"
    r"radius_fraction(?:_of_half_box)?_(?P<radius_fraction>[0-9.]+)/"
    r"(?P<center_label>[^/]+)/"
    r"cavitation\.gsd$"
)

CENTER_RE = re.compile(
    r"center_x_(?P<x>-?[0-9.]+)_y_(?P<y>-?[0-9.]+)_z_(?P<z>-?[0-9.]+)$"
)


def _relative_posix(path, root):
    return Path(path).relative_to(root).as_posix()


def _require_copy_mode(link_mode):
    if link_mode != "copy":
        raise ValueError(
            "V3 migration is copy-only. Use link_mode='copy'."
        )


def copy_migrated_file(
    source_path,
    destination_path,
    overwrite=False,
):
    """
    Copy one existing saved file into the V3 layout.
    """

    source_path = Path(source_path)
    destination_path = Path(destination_path)

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        if not overwrite:
            return {
                "action": "skipped_exists",
                "source_path": source_path,
                "destination_path": destination_path,
            }

        destination_path.unlink()

    shutil.copy2(source_path, destination_path)

    return {
        "action": "copied",
        "source_path": source_path,
        "destination_path": destination_path,
    }


def parse_thermalized_log_path(log_path, root=THERMALIZED_STATES_V2_ROOT):
    rel = _relative_posix(log_path, root)
    match = THERMALIZED_RE.match(rel)

    if match is None:
        return None

    values = match.groupdict()
    values["n_fcc_cells"] = int(values["n_fcc_cells"])
    values["rho"] = float(values["rho"])
    values["kT"] = float(values["kT"])
    values["nsteps"] = int(values["nsteps"])
    values["seed"] = int(values["seed"])

    return values


def parse_lattice_path(state_path, root=SIMPLE_LATTICES_V2_ROOT):
    rel = _relative_posix(state_path, root)
    match = LATTICE_RE.match(rel)

    if match is None:
        return None

    values = match.groupdict()
    values["n_fcc_cells"] = int(values["n_fcc_cells"])
    values["rho"] = float(values["rho"])

    return values


def parse_cavitation_state_path(state_path, root=CAVITATION_STATES_ROOT):
    rel = _relative_posix(state_path, root)
    match = CAVITATION_RE.match(rel)

    if match is None:
        return None

    values = match.groupdict()
    values["n_fcc_cells"] = int(values["n_fcc_cells"])
    values["rho"] = float(values["rho"])
    values["kT"] = float(values["kT"])
    values["source_nsteps"] = int(values["source_nsteps"])
    values["source_seed"] = int(values["source_seed"])
    values["radius_fraction"] = float(values["radius_fraction"])

    return values


def _old_state_for_log(log_path, phase_name):
    return Path(log_path).with_name(f"{phase_name}.gsd")


def _try_gsd_state_stats(state_path):
    try:
        import gsd.hoomd
    except Exception:
        return {}

    try:
        with gsd.hoomd.open(name=str(state_path), mode="r") as gsd_file:
            frame_count = len(gsd_file)
            frame = gsd_file[-1]

    except Exception:
        return {}

    box = np.asarray(frame.configuration.box, dtype=np.float64)
    n_particles = int(frame.particles.N)
    volume = float(box[0] * box[1] * box[2])

    return {
        "N": n_particles,
        "BoxLength": float(box[0]),
        "volume": volume,
        "actual_rho": float(n_particles / volume) if volume else np.nan,
        "gsd_frame_count": frame_count,
    }


def _split_thermalized_metadata(old_attrs, info, old_state_path, old_log_path, new_paths):
    target_rho = old_attrs.get("target_rho", info["rho"])
    actual_rho = old_attrs.get("actual_rho", np.nan)
    n_fcc_cells = old_attrs.get("n_fcc_cells", info["n_fcc_cells"])
    kT = old_attrs.get("kT", info["kT"])
    phase_name = old_attrs.get("phase_name", info["phase_name"])

    state = {
        "state_kind": "thermalized",
        "data_version": "v3",
        "migrated_from_data_version": "v2",
        "lattice_type": old_attrs.get("lattice_type", "fcc"),
        "density_mode": old_attrs.get("density_mode", "fixed_N_variable_L"),
        "n_fcc_cells": n_fcc_cells,
        "N": old_attrs.get("N", np.nan),
        "target_rho": target_rho,
        "actual_rho": actual_rho,
        "kT": kT,
        "BoxLength": old_attrs.get("BoxLength", np.nan),
        "volume": old_attrs.get("volume", np.nan),
        "fcc_cell_size": old_attrs.get("fcc_cell_size", np.nan),
        "state_path": str(new_paths["state_path"]),
    }

    run = {
        "run_kind": "thermalization",
        "phase_name": phase_name,
        "nsteps": old_attrs.get("nsteps", info["nsteps"]),
        "seed": old_attrs.get("seed", info["seed"]),
        "dt": old_attrs.get("dt", np.nan),
        "kT": kT,
        "log_period": old_attrs.get("log_period", np.nan),
        "final_timestep": old_attrs.get("final_timestep", np.nan),
        "starting_state_path": old_attrs.get("starting_state_path", ""),
        "state_path": str(new_paths["state_path"]),
        "log_path": str(new_paths["log_path"]),
    }

    lj = {
        "epsilon_LJ": old_attrs.get("epsilon_LJ", np.nan),
        "sigma_LJ": old_attrs.get("sigma_LJ", np.nan),
        "r_cut_LJ": old_attrs.get("r_cut_LJ", np.nan),
        "r_on_LJ": old_attrs.get("r_on_LJ", np.nan),
        "buffer_LJ": old_attrs.get("buffer_LJ", np.nan),
        "lj_mode": old_attrs.get("lj_mode", ""),
    }

    source = {
        "source_data_version": "v2",
        "old_state_path": str(old_state_path),
        "old_log_path": str(old_log_path),
        "migration_note": (
            "Original flat metadata attrs were preserved at /metadata.attrs."
        ),
    }

    paths = {
        "state_path": str(new_paths["state_path"]),
        "log_path": str(new_paths["log_path"]),
        "old_state_path": str(old_state_path),
        "old_log_path": str(old_log_path),
    }

    classification = {
        "phase_separated": old_attrs.get("phase_separated", None),
    }

    return {
        "metadata/state": state,
        "metadata/run": run,
        "metadata/lj": lj,
        "metadata/source": source,
        "metadata/paths": paths,
        "metadata/classification/phase_separation": classification,
    }


def write_v3_thermalized_metadata(
    log_path,
    info,
    old_state_path,
    old_log_path,
    new_paths,
):
    old_attrs = metadata_helpers.read_attrs(
        hdf5_path=log_path,
        group_path="metadata",
    )

    groups = _split_thermalized_metadata(
        old_attrs=old_attrs,
        info=info,
        old_state_path=old_state_path,
        old_log_path=old_log_path,
        new_paths=new_paths,
    )

    voxel_attrs = metadata_helpers.read_attrs(
        hdf5_path=log_path,
        group_path="metadata/phase_separation/voxel",
    )

    if voxel_attrs:
        groups["metadata/classification/phase_separation/voxel"] = voxel_attrs

    pe_drop_attrs = metadata_helpers.read_attrs(
        hdf5_path=log_path,
        group_path="metadata/phase_separation/PE_drop",
    )

    if pe_drop_attrs:
        groups["metadata/classification/phase_separation/PE_drop"] = pe_drop_attrs

    metadata_helpers.write_metadata_groups(
        hdf5_path=log_path,
        groups=groups,
        mode="a",
        overwrite=True,
    )

    return groups


def write_v3_lattice_metadata(
    metadata_path,
    info,
    old_state_path,
    new_paths,
    overwrite=True,
):
    stats = _try_gsd_state_stats(new_paths["state_path"])

    n_fcc_cells = info["n_fcc_cells"]
    target_rho = info["rho"]
    box_length = stats.get("BoxLength", np.nan)

    state = {
        "state_kind": "lattice",
        "data_version": "v3",
        "migrated_from_data_version": "v2",
        "lattice_type": "fcc",
        "density_mode": "fixed_N_variable_L",
        "n_fcc_cells": n_fcc_cells,
        "target_rho": target_rho,
        "actual_rho": stats.get("actual_rho", np.nan),
        "N": stats.get("N", 4 * int(n_fcc_cells) ** 3),
        "BoxLength": box_length,
        "volume": stats.get("volume", np.nan),
        "fcc_cell_size": (
            float(box_length) / int(n_fcc_cells)
            if np.isfinite(box_length)
            else np.nan
        ),
        "state_path": str(new_paths["state_path"]),
    }

    source = {
        "source_data_version": "v2",
        "old_state_path": str(old_state_path),
    }

    paths = {
        "state_path": str(new_paths["state_path"]),
        "metadata_path": str(metadata_path),
        "old_state_path": str(old_state_path),
    }

    metadata_helpers.write_metadata_groups(
        hdf5_path=metadata_path,
        groups={
            "metadata/state": state,
            "metadata/source": source,
            "metadata/paths": paths,
        },
        mode="w" if overwrite else "a",
        overwrite=True,
    )

    return {
        "metadata/state": state,
        "metadata/source": source,
        "metadata/paths": paths,
    }


def _old_source_state_path(info):
    return (
        Path(THERMALIZED_STATES_V2_ROOT)
        / "FCC"
        / f"n_cells_{info['n_fcc_cells']}"
        / f"rho_{info['rho']:.3f}"
        / f"kT_{info['kT']:.3f}"
        / f"nsteps_{info['source_nsteps']}"
        / f"seed_{info['source_seed']}"
        / f"{info['source_phase_name']}.gsd"
    )


def _old_source_log_path(info):
    return _old_source_state_path(info).with_name(
        f"{info['source_phase_name']}_log.hdf5"
    )


def _center_kwargs_from_label(label):
    if label == "center_box":
        return {
            "center": None,
            "random_location": False,
            "bubble_seed": None,
        }

    if label.startswith("random_center_bubble_seed_"):
        return {
            "center": None,
            "random_location": True,
            "bubble_seed": int(label.rsplit("_", 1)[-1]),
        }

    if label.startswith("random_center_seed_"):
        return {
            "center": None,
            "random_location": True,
            "bubble_seed": int(label.rsplit("_", 1)[-1]),
        }

    match = CENTER_RE.match(label)

    if match is not None:
        return {
            "center": [
                float(match.group("x")),
                float(match.group("y")),
                float(match.group("z")),
            ],
            "random_location": False,
            "bubble_seed": None,
        }

    return {
        "center": None,
        "random_location": False,
        "bubble_seed": None,
    }


def _try_cavitation_creation_stats(source_state_path, cavitation_state_path):
    try:
        import gsd.hoomd
    except Exception:
        return {}, {}

    try:
        with gsd.hoomd.open(name=str(source_state_path), mode="r") as source_file:
            source_frame = source_file[-1]

        with gsd.hoomd.open(
            name=str(cavitation_state_path),
            mode="r",
        ) as cavitation_file:
            cavitation_frame = cavitation_file[-1]

    except Exception:
        return {}, {}

    source_positions = np.asarray(source_frame.particles.position)
    cavitation_positions = np.asarray(cavitation_frame.particles.position)

    n_before = int(source_frame.particles.N)
    n_after = int(cavitation_frame.particles.N)
    n_removed = n_before - n_after

    source_box = np.asarray(source_frame.configuration.box, dtype=np.float64)
    volume = float(source_box[0] * source_box[1] * source_box[2])

    stats = {
        "N_before": n_before,
        "N_after": n_after,
        "N_removed": n_removed,
        "fraction_removed": float(n_removed / n_before) if n_before else np.nan,
        "rho_before": float(n_before / volume) if volume else np.nan,
        "rho_after": float(n_after / volume) if volume else np.nan,
        "BoxLength": float(source_box[0]),
        "volume": volume,
    }

    datasets = {}

    if n_removed > 0 and n_before < 250_000:
        cavitation_position_set = {
            tuple(np.round(row, 12))
            for row in cavitation_positions
        }
        removed_indices = [
            index
            for index, row in enumerate(source_positions)
            if tuple(np.round(row, 12)) not in cavitation_position_set
        ]

        if len(removed_indices) == n_removed:
            datasets["creation/removed_particle_indices"] = np.asarray(
                removed_indices,
                dtype=np.int64,
            )
            datasets["creation/removed_particle_positions"] = source_positions[
                removed_indices
            ]

    return stats, datasets


def migrate_thermalized_states(
    dry_run=True,
    link_mode="copy",
    overwrite=False,
    limit=None,
    write_v3_metadata=True,
):
    _require_copy_mode(link_mode)

    rows = []
    log_paths = sorted(Path(THERMALIZED_STATES_V2_ROOT).glob("**/*_log.hdf5"))

    if limit is not None:
        log_paths = log_paths[: int(limit)]

    for old_log_path in log_paths:
        info = parse_thermalized_log_path(old_log_path)

        if info is None:
            rows.append(
                {
                    "object_kind": "thermalized",
                    "action": "skipped_unrecognized_path",
                    "source_path": str(old_log_path),
                }
            )
            continue

        old_state_path = _old_state_for_log(
            log_path=old_log_path,
            phase_name=info["phase_name"],
        )

        new_paths = path_helpers.thermalized_run_paths(
            n_fcc_cells=info["n_fcc_cells"],
            target_rho=info["rho"],
            kT=info["kT"],
            nsteps=info["nsteps"],
            seed=info["seed"],
            phase_name=info["phase_name"],
        )

        pairs = [
            (old_state_path, new_paths["state_path"], "state_gsd", link_mode),
            (old_log_path, new_paths["log_path"], "log_hdf5", link_mode),
        ]

        for source_path, destination_path, file_role, file_link_mode in pairs:
            row = {
                "object_kind": "thermalized",
                "file_role": file_role,
                "link_mode": file_link_mode,
                "source_path": str(source_path),
                "destination_path": str(destination_path),
            }

            if not source_path.exists():
                row["action"] = "missing_source"

            elif dry_run:
                row["action"] = "would_copy"
                if file_role == "log_hdf5" and write_v3_metadata:
                    row["metadata_action"] = "would_write_v3_metadata_groups"

            else:
                result = copy_migrated_file(
                    source_path=source_path,
                    destination_path=destination_path,
                    overwrite=overwrite,
                )
                row["action"] = result["action"]

                if file_role == "log_hdf5" and write_v3_metadata:
                    write_v3_thermalized_metadata(
                        log_path=destination_path,
                        info=info,
                        old_state_path=old_state_path,
                        old_log_path=old_log_path,
                        new_paths=new_paths,
                    )
                    row["metadata_action"] = "wrote_v3_metadata_groups"

            rows.append(row)

    return pd.DataFrame(rows)


def migrate_lattice_states(
    dry_run=True,
    link_mode="copy",
    overwrite=False,
    limit=None,
    write_v3_metadata=True,
):
    _require_copy_mode(link_mode)

    rows = []
    state_paths = sorted(Path(SIMPLE_LATTICES_V2_ROOT).glob("**/lattice.gsd"))

    if limit is not None:
        state_paths = state_paths[: int(limit)]

    for old_state_path in state_paths:
        info = parse_lattice_path(old_state_path)

        if info is None:
            rows.append(
                {
                    "object_kind": "lattice",
                    "action": "skipped_unrecognized_path",
                    "source_path": str(old_state_path),
                }
            )
            continue

        new_paths = path_helpers.lattice_paths(
            n_fcc_cells=info["n_fcc_cells"],
            target_rho=info["rho"],
        )

        row = {
            "object_kind": "lattice",
            "file_role": "state_gsd",
            "link_mode": link_mode,
            "source_path": str(old_state_path),
            "destination_path": str(new_paths["state_path"]),
            "metadata_path": str(new_paths["metadata_path"]),
        }

        if not old_state_path.exists():
            row["action"] = "missing_source"

        elif dry_run:
            row["action"] = "would_copy"
            if write_v3_metadata:
                row["metadata_action"] = "would_write_lattice_metadata"

        else:
            result = copy_migrated_file(
                source_path=old_state_path,
                destination_path=new_paths["state_path"],
                overwrite=overwrite,
            )

            row["action"] = result["action"]

            if write_v3_metadata:
                write_v3_lattice_metadata(
                    metadata_path=new_paths["metadata_path"],
                    info=info,
                    old_state_path=old_state_path,
                    new_paths=new_paths,
                    overwrite=True,
                )
                row["metadata_action"] = "wrote_lattice_metadata"

        rows.append(row)

    return pd.DataFrame(rows)


def migrate_cavitation_states(
    dry_run=True,
    link_mode="copy",
    overwrite=False,
    limit=None,
    compute_creation_stats=True,
):
    _require_copy_mode(link_mode)

    rows = []
    state_paths = sorted(Path(CAVITATION_STATES_ROOT).glob("**/cavitation.gsd"))

    if limit is not None:
        state_paths = state_paths[: int(limit)]

    for old_state_path in state_paths:
        info = parse_cavitation_state_path(old_state_path)

        if info is None:
            rows.append(
                {
                    "object_kind": "cavitation_initial",
                    "action": "skipped_unrecognized_path",
                    "source_path": str(old_state_path),
                }
            )
            continue

        center_kwargs = _center_kwargs_from_label(info["center_label"])

        new_paths = path_helpers.cavitation_state_paths(
            n_fcc_cells=info["n_fcc_cells"],
            source_rho=info["rho"],
            kT=info["kT"],
            source_nsteps=info["source_nsteps"],
            source_seed=info["source_seed"],
            source_phase_name=info["source_phase_name"],
            radius_fraction=info["radius_fraction"],
            **center_kwargs,
        )

        row = {
            "object_kind": "cavitation_initial",
            "file_role": "state_gsd",
            "source_path": str(old_state_path),
            "destination_path": str(new_paths["state_path"]),
            "metadata_path": str(new_paths["creation_metadata_path"]),
        }

        if dry_run:
            row["action"] = "would_copy"
            rows.append(row)
            continue

        result = copy_migrated_file(
            source_path=old_state_path,
            destination_path=new_paths["state_path"],
            overwrite=overwrite,
        )

        source_state_path = _old_source_state_path(info)
        source_log_path = _old_source_log_path(info)

        stats = {}
        datasets = {}

        if compute_creation_stats:
            stats, datasets = _try_cavitation_creation_stats(
                source_state_path=source_state_path,
                cavitation_state_path=old_state_path,
            )

        source_metadata = {
            "source_state_path": str(source_state_path),
            "source_log_path": str(source_log_path),
            "source_data_version": "v2",
        }

        creation_metadata = {
            "state_kind": "cavitation_initial",
            "creation_method": "remove_particles_in_sphere",
            "radius_fraction": info["radius_fraction"],
            "radius_definition": "bubble_radius = radius_fraction * (BoxLength / 2)",
            "source_rho": info["rho"],
            "kT": info["kT"],
            "source_nsteps": info["source_nsteps"],
            "source_seed": info["source_seed"],
            "source_phase_name": info["source_phase_name"],
            "center_label": info["center_label"],
            "random_location": center_kwargs["random_location"],
            "bubble_seed": center_kwargs["bubble_seed"],
            "old_state_path": str(old_state_path),
            "v3_state_path": str(new_paths["state_path"]),
            **stats,
        }

        if center_kwargs["center"] is not None:
            creation_metadata["bubble_center"] = center_kwargs["center"]

        metadata_helpers.write_creation_metadata(
            hdf5_path=new_paths["creation_metadata_path"],
            source=source_metadata,
            creation=creation_metadata,
            datasets=datasets,
            overwrite=True,
        )

        row["action"] = result["action"]
        row["metadata_action"] = "wrote_creation_metadata"
        rows.append(row)

    return pd.DataFrame(rows)


def migrate_v2_to_v3(
    dry_run=True,
    link_mode="copy",
    overwrite=False,
    migrate_lattices=True,
    migrate_thermalized=True,
    migrate_cavitation_states=False,
    write_v3_metadata=True,
    limit=None,
):
    """
    Copy existing V2 saved files into the V3 layout.

    This never reruns a simulation. Start with dry_run=True, review the
    returned DataFrame, then rerun with dry_run=False.
    """

    frames = []

    _require_copy_mode(link_mode)

    if migrate_lattices:
        frames.append(
            migrate_lattice_states(
                dry_run=dry_run,
                link_mode=link_mode,
                overwrite=overwrite,
                limit=limit,
                write_v3_metadata=write_v3_metadata,
            )
        )

    if migrate_thermalized:
        frames.append(
            migrate_thermalized_states(
                dry_run=dry_run,
                link_mode=link_mode,
                overwrite=overwrite,
                limit=limit,
                write_v3_metadata=write_v3_metadata,
            )
        )

    if migrate_cavitation_states:
        frames.append(
            migrate_cavitation_states(
                dry_run=dry_run,
                link_mode=link_mode,
                overwrite=overwrite,
                limit=limit,
            )
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
