from collections import Counter
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

CAVITATION_CREATION_ATTRIBUTES = {
    "bubble_center",
    "bubble_method",
    "bubble_seed",
    "particles_removed",
    "periodic_distance",
    "radius",
    "random_location",
}

CAVITATION_CREATION_SOURCE_ATTRIBUTES = {
    "source_state_path",
    "source_log_path",
    "source_rho",
    "source_kT",
    "source_nsteps",
    "source_seed",
}

CAVITATION_CREATION_STATE_ATTRIBUTES = {
    "BoxLength",
    "N",
    "actual_rho",
    "density_mode",
    "kT",
    "lattice_type",
    "n_fcc_cells",
    "state_kind",
    "volume",
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
    """Make one cavitation initial-state file match the current schema."""

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
        state = hdf["metadata/state"]
        missing_state_attrs = sorted(
            CAVITATION_CREATION_STATE_ATTRIBUTES - set(state.attrs)
        )
        if missing_state_attrs:
            raise KeyError(
                "metadata/state is missing required attributes: "
                f"{missing_state_attrs}"
            )

        found_state_attrs = sorted(
            set(state.attrs) - CAVITATION_CREATION_STATE_ATTRIBUTES
        )

        creation_path = "metadata/creation"
        if creation_path not in hdf:
            raise KeyError("missing metadata/creation")

        creation = hdf[creation_path]
        random_location = bool(
            clean_read_value(
                creation.attrs.get("random_location", False)
            )
        )
        allowed_creation_attrs = set(CAVITATION_CREATION_ATTRIBUTES)
        if not random_location:
            allowed_creation_attrs.remove("bubble_seed")

        missing_creation_attrs = sorted(
            allowed_creation_attrs - set(creation.attrs)
        )
        if missing_creation_attrs:
            raise KeyError(
                "metadata/creation is missing required attributes: "
                f"{missing_creation_attrs}"
            )

        found_attrs = sorted(
            set(creation.attrs) - allowed_creation_attrs
        )
        found_creation_children = [
            {
                "name": child_name,
                "storage": type(creation[child_name]).__name__.lower(),
            }
            for child_name in sorted(creation.keys())
        ]

        paths_path = "metadata/paths"
        remove_paths_group = paths_path in hdf
        found_path_attrs = []
        if remove_paths_group:
            paths_group = hdf[paths_path]
            found_path_attrs = sorted(paths_group.attrs)

        source_path = "metadata/source"
        if source_path not in hdf:
            raise KeyError("missing metadata/source")

        source = hdf[source_path]
        missing_source_attrs = sorted(
            CAVITATION_CREATION_SOURCE_ATTRIBUTES - set(source.attrs)
        )
        if missing_source_attrs:
            raise KeyError(
                "metadata/source is missing required attributes: "
                f"{missing_source_attrs}"
            )

        found_source_attrs = sorted(
            set(source.attrs) - CAVITATION_CREATION_SOURCE_ATTRIBUTES
        )

        if not dry_run:
            for attr_name in found_state_attrs:
                del state.attrs[attr_name]
            for attr_name in found_attrs:
                del creation.attrs[attr_name]
            for item in found_creation_children:
                del creation[item["name"]]
            if remove_paths_group:
                del hdf[paths_path]
            for attr_name in found_source_attrs:
                del source.attrs[attr_name]

    removed = [
        {
            "path": f"metadata/state/{attr_name}",
            "storage": "attribute",
        }
        for attr_name in found_state_attrs
    ]
    removed.extend(
        {
            "path": f"{creation_path}/{attr_name}",
            "storage": "attribute",
        }
        for attr_name in found_attrs
    )
    removed.extend(
        {
            "path": f"{creation_path}/{item['name']}",
            "storage": item["storage"],
        }
        for item in found_creation_children
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


def run_cavitation_creation_metadata_cleanup(
    root=None,
    apply_changes=False,
    show_changed_files=False,
):
    """
    Preview or apply the complete cavitation initial-state metadata cleanup.

    This is the notebook-facing entry point. It prints an aggregate report,
    prints every error, and performs a second dry-run verification after an
    applied cleanup.
    """

    if root is None:
        from .paths import CAVITATION_STATES_V3_ROOT

        root = CAVITATION_STATES_V3_ROOT

    root = Path(root)
    reports = cleanup_cavitation_creation_metadata_tree(
        root=root,
        dry_run=not apply_changes,
    )

    if not reports:
        raise RuntimeError(
            f"No cavitation_creation.hdf5 files were found under {root}"
        )

    counts = Counter(report["status"] for report in reports)
    errors = [
        report
        for report in reports
        if report["status"] == "error"
    ]
    changed = [
        report
        for report in reports
        if report["status"] in {"would_clean", "cleaned"}
    ]

    removed_counts = Counter(
        (item["path"], item["storage"])
        for report in changed
        for item in report["removed"]
    )

    print("=" * 100)
    print("CAVITATION INITIAL-STATE METADATA CLEANUP")
    print("=" * 100)
    print(f"Root:           {root}")
    print(
        "Mode:           "
        f"{'APPLY CHANGES' if apply_changes else 'DRY RUN'}"
    )
    print(f"Files checked:  {len(reports)}")

    print("\nSTATUS")
    for status, count in sorted(counts.items()):
        print(f"{status:15} {count}")

    print("\nFIELDS")
    print("=" * 100)
    if removed_counts:
        for (path, storage), count in sorted(removed_counts.items()):
            print(f"/{path:<72} {count:>6} files  [{storage}]")
    else:
        print("No fields need cleanup.")

    if show_changed_files and changed:
        print("\nFILES WITH CHANGES")
        print("=" * 100)
        for report in changed:
            print(f"\n{report['status'].upper()}: {report['hdf5_path']}")
            for item in report["removed"]:
                print(f"  - /{item['path']} [{item['storage']}]")

    print("\nERRORS")
    print("=" * 100)
    if errors:
        for report in errors:
            print(f"\nERROR: {report['hdf5_path']}")
            print(f"       {report['error']}")
    else:
        print("No errors found.")

    verification = []
    remaining = []
    verification_errors = []

    if not apply_changes:
        print("\nDRY RUN ONLY: no files were modified.")
        print("Set APPLY_CHANGES = True and rerun to perform the cleanup.")
    else:
        verification = cleanup_cavitation_creation_metadata_tree(
            root=root,
            dry_run=True,
        )
        remaining = [
            report
            for report in verification
            if report["status"] == "would_clean"
        ]
        verification_errors = [
            report
            for report in verification
            if report["status"] == "error"
        ]

        print("\nVERIFICATION")
        print("=" * 100)
        print(f"Files still needing cleanup: {len(remaining)}")
        print(f"Files with errors:           {len(verification_errors)}")

        if verification_errors:
            for report in verification_errors:
                print(f"\nERROR: {report['hdf5_path']}")
                print(f"       {report['error']}")

        if not remaining and not verification_errors:
            print("\nCleanup applied and verified.")
        else:
            print("\nCleanup ran, but verification did not pass.")

    return {
        "root": str(root),
        "apply_changes": bool(apply_changes),
        "reports": reports,
        "errors": errors,
        "verification": verification,
        "remaining": remaining,
        "verification_errors": verification_errors,
    }


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
