from pathlib import Path

import numpy as np


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
        "fcc_cell_size",
    ]

    run_keys = [
        "phase_name",
        "nsteps",
        "seed",
        "dt",
        "log_period",
        "final_timestep",
    ]

    lj_keys = [
        "epsilon_LJ",
        "sigma_LJ",
        "r_cut_LJ",
        "r_on_LJ",
        "buffer_LJ",
        "lj_mode",
    ]

    path_keys = [
        "state_path",
        "log_path",
        "metadata_path",
    ]

    source_keys = [
        "starting_state_path",
        "source_state_path",
        "source_log_path",
        "old_state_path",
        "old_log_path",
        "source_data_version",
        "migrated_from_data_version",
        "migration_note",
    ]

    state = {
        "state_kind": flat_metadata.get("state_kind", state_kind),
        "data_version": flat_metadata.get("data_version", data_version),
    }

    for key in state_keys:
        if key in flat_metadata:
            state[key] = flat_metadata[key]

    if "migrated_from_data_version" in flat_metadata:
        state["migrated_from_data_version"] = flat_metadata[
            "migrated_from_data_version"
        ]

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


def write_creation_metadata(
    hdf5_path,
    source,
    creation,
    datasets=None,
    overwrite=False,
):
    hdf5_path = Path(hdf5_path)

    if hdf5_path.exists() and not overwrite:
        raise FileExistsError(f"Creation metadata already exists: {hdf5_path}")

    mode = "w" if overwrite or not hdf5_path.exists() else "a"

    write_metadata_groups(
        hdf5_path=hdf5_path,
        mode=mode,
        groups={
            "metadata/source": source,
            "metadata/creation": creation,
        },
        overwrite=True,
    )

    if datasets:
        write_datasets(
            hdf5_path=hdf5_path,
            datasets=datasets,
            mode="a",
            overwrite=True,
        )

    return hdf5_path


def append_run_metadata(
    log_path,
    state,
    source=None,
    run=None,
    creation=None,
    classification=None,
    observables=None,
    overwrite=True,
):
    groups = {
        "metadata/state": state or {},
    }

    optional_groups = {
        "metadata/source": source,
        "metadata/run": run,
        "metadata/creation": creation,
        "metadata/classification": classification,
        "metadata/observables": observables,
    }

    for group_path, attrs in optional_groups.items():
        if attrs:
            groups[group_path] = attrs

    return write_metadata_groups(
        hdf5_path=log_path,
        groups=groups,
        mode="a",
        overwrite=overwrite,
    )
