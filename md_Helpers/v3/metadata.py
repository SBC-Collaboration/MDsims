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
