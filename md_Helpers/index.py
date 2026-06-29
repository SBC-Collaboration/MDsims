from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .paths import (
    CAVITATION_EVOLVED_V3_ROOT,
    CAVITATION_STATES_V3_ROOT,
    EXCITATION_EVOLVED_V3_ROOT,
    EXCITATION_STATES_V3_ROOT,
    SIMPLE_LATTICES_V3_ROOT,
    THERMALIZED_STATES_V3_ROOT,
    index_path,
)


DEFAULT_INDEX_ROOTS = {
    "lattice": SIMPLE_LATTICES_V3_ROOT,
    "thermalized": THERMALIZED_STATES_V3_ROOT,
    "cavitation_initial": CAVITATION_STATES_V3_ROOT,
    "cavitation_evolved": CAVITATION_EVOLVED_V3_ROOT,
    "excitation_initial": EXCITATION_STATES_V3_ROOT,
    "excitation_evolved": EXCITATION_EVOLVED_V3_ROOT,
}


def _clean(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def flatten_metadata(hdf5_path):
    """Flatten all attributes below /metadata into dotted columns."""

    row = {"hdf5_path": str(Path(hdf5_path))}
    with h5py.File(hdf5_path, mode="r") as hdf:
        if "metadata" not in hdf:
            return row

        def collect(name, obj):
            if not isinstance(obj, h5py.Group):
                return
            if name != "metadata" and not name.startswith("metadata/"):
                return
            prefix = name.removeprefix("metadata/").replace("/", ".")
            for key, value in obj.attrs.items():
                column = f"{prefix}.{key}" if prefix else key
                row[column] = _clean(value)

        hdf.visititems(collect)
    return row


def scan_v3_metadata(roots=None):
    """Return one searchable row per V3 metadata/log HDF5 file."""

    roots = roots or DEFAULT_INDEX_ROOTS
    rows = []
    for object_kind, root in roots.items():
        root = Path(root)
        if not root.exists():
            continue
        for hdf5_path in sorted(root.rglob("*.hdf5")):
            row = flatten_metadata(hdf5_path)
            row["object_kind"] = object_kind
            rows.append(row)
    return pd.DataFrame(rows)


def build_v3_index(output_path=None, roots=None):
    """Scan V3 metadata and save a Parquet index."""

    table = scan_v3_metadata(roots=roots)
    output_path = Path(output_path or index_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    return table


def load_v3_index(path=None):
    """Load the saved V3 Parquet index."""

    return pd.read_parquet(Path(path or index_path()))
