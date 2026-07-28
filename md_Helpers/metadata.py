from pathlib import Path

import numpy as np


THERMALIZED_METADATA_REMOVALS = {
    "metadata/state": {
        "data_version",
        "fcc_cell_size",
        "migrated_from_data_version",
    },
    "metadata/run": {
        "final_timestep",
    },
    "metadata/paths": {
        "log_path",
        "state_path",
    },
    "metadata/source": {
        "migration_note",
        "old_log_path",
        "old_state_path",
    },
    "metadata/classification/phase_separation/voxel": {
        "max_voxel_density",
        "mean_voxel_density",
        "min_voxel_density",
        "n_fcc_cells",
        "n_voxels",
        "std_voxel_density",
        "voxel_volume",
    },
}

CAVITATION_CREATION_ATTRIBUTE_REMOVALS = {
    "BoxLength",
    "N_after",
    "N_before",
    "bubble_center_x",
    "bubble_center_y",
    "bubble_center_z",
    "bubble_radius",
    "copied_particle_fields",
    "particle_fraction_removed",
    "radius_definition",
    "rho_after",
    "rho_before",
    "volume",
}

CAVITATION_CREATION_DATASET_REMOVALS = {
    "removed_particle_indices",
    "removed_particle_positions",
}

CAVITATION_CREATION_PATH_ATTRIBUTE_REMOVALS = {
    "creation_metadata_path",
    "state_path",
}

CAVITATION_CREATION_SOURCE_ATTRIBUTES = {
    "source_state_path",
    "source_log_path",
    "source_rho",
    "source_kT",
    "source_nsteps",
    "source_seed",
}


def _open_hdf5(path, mode):
    import h5py
    
    return h5py.File(path, mode=mode)


def _clean_attr_value(value):
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""

        array = np.asarray(value)

        if array.dtype.kind in {"U", "O"}:
            return ",".join(str(item) for item in value)

        return array

    return value


def clean_read_value(value):
    if isinstance(value, bytes):
        return value.decode()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value

    return value


def read_attrs(hdf5_path, group_path):
    hdf5_path = Path(hdf5_path)

    with _open_hdf5(hdf5_path, mode="r") as hdf:
        if group_path not in hdf:
            return {}

        group = hdf[group_path]

        return {
            key: clean_read_value(value)
            for key, value in group.attrs.items()
        }


def write_attrs(group, attrs, overwrite=True):
    for key, value in attrs.items():
        value = _clean_attr_value(value)

        if value is None:
            continue

        if key in group.attrs and not overwrite:
            continue

        group.attrs[key] = value


def write_metadata_groups(hdf5_path, groups, mode="a", overwrite=True):
    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_hdf5(hdf5_path, mode=mode) as hdf:
        for group_path, attrs in groups.items():
            group = hdf.require_group(group_path)
            write_attrs(
                group=group,
                attrs=attrs,
                overwrite=overwrite,
            )

    return hdf5_path

def clear_attrs(hdf5_path, group_path="metadata"):
    hdf5_path = Path(hdf5_path)

    with _open_hdf5(hdf5_path, mode="a") as hdf:
        if group_path not in hdf:
            return False

        group = hdf[group_path]

        for key in list(group.attrs.keys()):
            del group.attrs[key]

    return True


def split_simulation_metadata(
    flat_metadata,
    state_kind="thermalized",
    run_kind="thermalization",
    data_version="v3",
):
    flat_metadata = dict(flat_metadata or {})

    resolved_state_kind = flat_metadata.get("state_kind", state_kind)
    is_thermalized = resolved_state_kind == "thermalized"

    state_keys = [
        "lattice_type",
        "density_mode",
        "n_fcc_cells",
        "N",
        "target_rho",
        "actual_rho",
        "kT",
        "BoxLength",
        "volume",
    ]
    if not is_thermalized:
        state_keys.append("fcc_cell_size")

    run_keys = [
        "phase_name",
        "nsteps",
        "seed",
        "dt",
        "log_period",
    ]
    if not is_thermalized:
        run_keys.append("final_timestep")

    lj_keys = [
        "epsilon_LJ",
        "sigma_LJ",
        "r_cut_LJ",
        "r_on_LJ",
        "buffer_LJ",
        "lj_mode",
    ]

    path_keys = [] if is_thermalized else [
        "state_path",
        "log_path",
        "metadata_path",
    ]

    source_keys = [
        "starting_state_path",
        "source_state_path",
        "source_log_path",
        "source_data_version",
    ]

    state = {"state_kind": resolved_state_kind}
    if not is_thermalized:
        state["data_version"] = flat_metadata.get(
            "data_version",
            data_version,
        )

    for key in state_keys:
        if key in flat_metadata:
            state[key] = flat_metadata[key]

    run = {
        "run_kind": flat_metadata.get("run_kind", run_kind),
    }

    for key in run_keys:
        if key in flat_metadata:
            run[key] = flat_metadata[key]

    lj = {
        key: flat_metadata[key]
        for key in lj_keys
        if key in flat_metadata
    }

    paths = {
        key: flat_metadata[key]
        for key in path_keys
        if key in flat_metadata
    }

    source = {
        key: flat_metadata[key]
        for key in source_keys
        if key in flat_metadata
    }

    classification = {}

    if "phase_separated" in flat_metadata:
        classification["phase_separated"] = flat_metadata["phase_separated"]

    groups = {
        "metadata/state": state,
        "metadata/run": run,
    }

    optional_groups = {
        "metadata/lj": lj,
        "metadata/paths": paths,
        "metadata/source": source,
        "metadata/classification/phase_separation": classification,
    }

    for group_path, attrs in optional_groups.items():
        if attrs:
            groups[group_path] = attrs

    return groups


def cleanup_thermalized_metadata_file(hdf5_path, dry_run=True):
    """Remove retired metadata attributes from one thermalized-state log."""

    hdf5_path = Path(hdf5_path)
    mode = "r" if dry_run else "a"

    with _open_hdf5(hdf5_path, mode=mode) as hdf:
        if "metadata/state" not in hdf:
            raise KeyError("missing metadata/state")

        state_kind = clean_read_value(
            hdf["metadata/state"].attrs.get("state_kind")
        )
        if state_kind != "thermalized":
            raise ValueError(
                f"state_kind is {state_kind!r}, not 'thermalized'"
            )

        found = []
        for group_path, attr_names in THERMALIZED_METADATA_REMOVALS.items():
            if group_path not in hdf:
                continue

            group = hdf[group_path]
            for attr_name in sorted(attr_names):
                if attr_name in group.attrs:
                    found.append({
                        "group": group_path,
                        "attribute": attr_name,
                        "value": clean_read_value(group.attrs[attr_name]),
                    })

        if not dry_run:
            for item in found:
                del hdf[item["group"]].attrs[item["attribute"]]

            paths_group = hdf.get("metadata/paths")
            if (
                paths_group is not None
                and not paths_group.attrs
                and len(paths_group) == 0
            ):
                del hdf["metadata/paths"]

    return {
        "hdf5_path": str(hdf5_path),
        "status": (
            "would_clean"
            if dry_run and found
            else "cleaned"
            if found
            else "already_clean"
        ),
        "removed_count": len(found),
        "removed": found,
    }


def cleanup_thermalized_metadata_tree(root=None, dry_run=True):
    """Clean every thermalized HDF5 log below a root and report failures."""

    if root is None:
        from .paths import THERMALIZED_STATES_V3_ROOT

        root = THERMALIZED_STATES_V3_ROOT

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Thermalized-state root does not exist: {root}")

    reports = []
    for hdf5_path in sorted(root.rglob("*.hdf5")):
        try:
            reports.append(
                cleanup_thermalized_metadata_file(
                    hdf5_path,
                    dry_run=dry_run,
                )
            )
        except Exception as error:
            reports.append({
                "hdf5_path": str(hdf5_path),
                "status": "error",
                "removed_count": 0,
                "removed": [],
                "error": f"{type(error).__name__}: {error}",
            })

    return reports


def cleanup_cavitation_creation_metadata_file(hdf5_path, dry_run=True):
    """Remove retired creation metadata from one cavitation initial state."""

    hdf5_path = Path(hdf5_path)
    mode = "r" if dry_run else "a"

    with _open_hdf5(hdf5_path, mode=mode) as hdf:
        if "metadata/state" not in hdf:
            raise KeyError("missing metadata/state")

        state_kind = clean_read_value(
            hdf["metadata/state"].attrs.get("state_kind")
        )
        if state_kind != "cavitation_initial":
            raise ValueError(
                f"state_kind is {state_kind!r}, not 'cavitation_initial'"
            )

        creation_path = "metadata/creation"
        if creation_path not in hdf:
            raise KeyError("missing metadata/creation")

        creation = hdf[creation_path]
        attr_names = set(CAVITATION_CREATION_ATTRIBUTE_REMOVALS)
        if not bool(creation.attrs.get("random_location", False)):
            attr_names.add("bubble_seed")

        found_attrs = [
            attr_name
            for attr_name in sorted(attr_names)
            if attr_name in creation.attrs
        ]
        found_datasets = [
            dataset_name
            for dataset_name in sorted(CAVITATION_CREATION_DATASET_REMOVALS)
            if dataset_name in creation
        ]

        paths_path = "metadata/paths"
        remove_paths_group = paths_path in hdf
        found_path_attrs = []
        if remove_paths_group:
            paths_group = hdf[paths_path]
            unexpected_attrs = (
                set(paths_group.attrs)
                - CAVITATION_CREATION_PATH_ATTRIBUTE_REMOVALS
            )
            if unexpected_attrs or len(paths_group) != 0:
                raise ValueError(
                    "metadata/paths contains unexpected content: "
                    f"attributes={sorted(unexpected_attrs)}, "
                    f"children={sorted(paths_group.keys())}"
                )
            found_path_attrs = sorted(paths_group.attrs)

        source_path = "metadata/source"
        if source_path not in hdf:
            raise KeyError("missing metadata/source")

        source = hdf[source_path]
        found_source_attrs = sorted(
            set(source.attrs) - CAVITATION_CREATION_SOURCE_ATTRIBUTES
        )

        if not dry_run:
            for attr_name in found_attrs:
                del creation.attrs[attr_name]
            for dataset_name in found_datasets:
                del creation[dataset_name]
            if remove_paths_group:
                del hdf[paths_path]
            for attr_name in found_source_attrs:
                del source.attrs[attr_name]

    removed = [
        {
            "path": f"{creation_path}/{attr_name}",
            "storage": "attribute",
        }
        for attr_name in found_attrs
    ]
    removed.extend(
        {
            "path": f"{creation_path}/{dataset_name}",
            "storage": "dataset",
        }
        for dataset_name in found_datasets
    )
    removed.extend(
        {
            "path": f"{paths_path}/{attr_name}",
            "storage": "attribute",
        }
        for attr_name in found_path_attrs
    )
    if remove_paths_group:
        removed.append({
            "path": paths_path,
            "storage": "group",
        })
    removed.extend(
        {
            "path": f"{source_path}/{attr_name}",
            "storage": "attribute",
        }
        for attr_name in found_source_attrs
    )

    return {
        "hdf5_path": str(hdf5_path),
        "status": (
            "would_clean"
            if dry_run and removed
            else "cleaned"
            if removed
            else "already_clean"
        ),
        "removed_count": len(removed),
        "removed": removed,
    }


def cleanup_cavitation_creation_metadata_tree(root=None, dry_run=True):
    """Clean every cavitation creation HDF5 file below a root."""

    if root is None:
        from .paths import CAVITATION_STATES_V3_ROOT

        root = CAVITATION_STATES_V3_ROOT

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Cavitation-state root does not exist: {root}")

    reports = []
    for hdf5_path in sorted(root.rglob("cavitation_creation.hdf5")):
        try:
            reports.append(
                cleanup_cavitation_creation_metadata_file(
                    hdf5_path,
                    dry_run=dry_run,
                )
            )
        except Exception as error:
            reports.append({
                "hdf5_path": str(hdf5_path),
                "status": "error",
                "removed_count": 0,
                "removed": [],
                "error": f"{type(error).__name__}: {error}",
            })

    return reports


def write_datasets(hdf5_path, datasets, mode="a", overwrite=True):
    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_hdf5(hdf5_path, mode=mode) as hdf:
        for dataset_path, values in datasets.items():
            if values is None:
                continue

            parent_path, dataset_name = str(dataset_path).rsplit("/", 1)
            parent = hdf.require_group(parent_path)

            if dataset_name in parent:
                if not overwrite:
                    continue
                del parent[dataset_name]

            parent.create_dataset(
                dataset_name,
                data=np.asarray(values),
            )

    return hdf5_path
