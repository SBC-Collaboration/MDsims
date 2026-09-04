"""The two canonical per-run files: trajectory.gsd and run.hdf5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .paths import RunPaths


LOG_DATASETS = {
    "hoomd-data/Simulation/timestep": ((), "i8"),
    "hoomd-data/Simulation/tps": ((), "f8"),
    "hoomd-data/md/compute/ThermodynamicQuantities/kinetic_temperature": (
        (),
        "f8",
    ),
    "hoomd-data/md/compute/ThermodynamicQuantities/pressure": ((), "f8"),
    "hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor": (
        (6,),
        "f8",
    ),
    "hoomd-data/md/compute/ThermodynamicQuantities/potential_energy": (
        (),
        "f8",
    ),
    "hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy": (
        (),
        "f8",
    ),
    "hoomd-data/md/compute/ThermodynamicQuantities/volume": ((), "f8"),
    "hoomd-data/md/compute/ThermodynamicQuantities/num_particles": (
        (),
        "i8",
    ),
    "mdsims/time/run_step": ((), "i8"),
    "mdsims/time/this_lj_time": ((), "f8"),
    "mdsims/time/cumulative_lj_time": ((), "f8"),
    "mdsims/output/trajectory_frame_id": ((), "i8"),
    "mdsims/analysis/speed/max_particle_speed": ((), "f8"),
}


@dataclass(frozen=True)
class StateData:
    positions: np.ndarray
    velocities: np.ndarray
    box: np.ndarray
    n_particles: int
    particle_types: tuple[str, ...]

    @property
    def volume(self) -> float:
        return float(np.prod(self.box[:3]))

    @property
    def density(self) -> float:
        return self.n_particles / self.volume


def _float_or_nan(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _attribute_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype.kind in {"O", "U"}:
            return json.dumps(list(value))
        return array
    if isinstance(value, np.generic):
        return value.item()
    return value


class RunStorage:
    """Append synchronized trajectory frames and thermodynamic samples."""

    def __init__(self, paths: RunPaths):
        self.paths = paths
        self._hdf5 = None
        self._frame_count = 0
        self.samples: dict[str, list[Any]] = {
            "run_step": [],
            "pressure": [],
            "potential_energy": [],
        }
        self.frame_records: list[dict[str, int]] = []

    def open(self, metadata_groups: dict[str, dict[str, Any]]) -> None:
        import h5py

        self.paths.directory.mkdir(parents=True, exist_ok=False)
        self._hdf5 = h5py.File(self.paths.hdf5, mode="w")
        for dataset_path, (tail_shape, dtype) in LOG_DATASETS.items():
            group_path, name = dataset_path.rsplit("/", 1)
            group = self._hdf5.require_group(group_path)
            group.create_dataset(
                name,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                chunks=True,
                dtype=dtype,
            )
        self.write_metadata(metadata_groups)
        self.flush()

    def write_metadata(self, groups: dict[str, dict[str, Any]]) -> None:
        if self._hdf5 is None:
            raise RuntimeError("RunStorage is not open")
        for group_path, attributes in groups.items():
            group = self._hdf5.require_group(group_path)
            for key, value in attributes.items():
                if value is None:
                    continue
                group.attrs[str(key)] = _attribute_value(value)

    def _append_hdf5(self, values: dict[str, Any]) -> None:
        if self._hdf5 is None:
            raise RuntimeError("RunStorage is not open")
        for path, value in values.items():
            dataset = self._hdf5[path]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = value

    def _append_gsd(self, snapshot, timestep: int) -> StateData:
        import gsd.hoomd

        positions = np.asarray(snapshot.particles.position, dtype=np.float64).copy()
        velocities = np.asarray(snapshot.particles.velocity, dtype=np.float64).copy()
        box = np.asarray(snapshot.configuration.box, dtype=np.float64).copy()
        type_ids = np.asarray(snapshot.particles.typeid, dtype=np.uint32).copy()
        images = np.asarray(snapshot.particles.image, dtype=np.int32).copy()
        masses = np.asarray(snapshot.particles.mass, dtype=np.float64).copy()
        particle_types = tuple(str(item) for item in snapshot.particles.types)
        n_particles = int(snapshot.particles.N)

        frame = gsd.hoomd.Frame()
        frame.configuration.step = int(timestep)
        frame.configuration.box = box
        frame.particles.N = n_particles
        frame.particles.types = list(particle_types)
        frame.particles.position = positions
        frame.particles.velocity = velocities
        frame.particles.typeid = type_ids
        frame.particles.image = images
        frame.particles.mass = masses

        mode = "w" if self._frame_count == 0 else "a"
        with gsd.hoomd.open(name=str(self.paths.trajectory), mode=mode) as trajectory:
            trajectory.append(frame)
        self._frame_count += 1

        return StateData(
            positions=positions,
            velocities=velocities,
            box=box,
            n_particles=n_particles,
            particle_types=particle_types,
        )

    def record(
        self,
        simulation,
        thermo,
        run_step: int,
        dt: float,
        prior_lj_time: float = 0.0,
        save_frame: bool = True,
    ) -> StateData:
        """Write one HDF5 sample and optionally its exact GSD frame."""

        snapshot = simulation.state.get_snapshot()
        if save_frame:
            trajectory_frame_id = self._frame_count
            state = self._append_gsd(snapshot, simulation.timestep)
        else:
            trajectory_frame_id = -1
            state = StateData(
                positions=np.asarray(
                    snapshot.particles.position,
                    dtype=np.float64,
                ).copy(),
                velocities=np.asarray(
                    snapshot.particles.velocity,
                    dtype=np.float64,
                ).copy(),
                box=np.asarray(
                    snapshot.configuration.box,
                    dtype=np.float64,
                ).copy(),
                n_particles=int(snapshot.particles.N),
                particle_types=tuple(
                    str(item) for item in snapshot.particles.types
                ),
            )
        speed = np.linalg.norm(state.velocities, axis=1)
        pressure_tensor = np.asarray(
            thermo.pressure_tensor,
            dtype=np.float64,
        )
        values = {
            "hoomd-data/Simulation/timestep": int(simulation.timestep),
            "hoomd-data/Simulation/tps": _float_or_nan(simulation.tps),
            "hoomd-data/md/compute/ThermodynamicQuantities/kinetic_temperature": (
                _float_or_nan(thermo.kinetic_temperature)
            ),
            "hoomd-data/md/compute/ThermodynamicQuantities/pressure": (
                _float_or_nan(thermo.pressure)
            ),
            "hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor": (
                pressure_tensor
            ),
            "hoomd-data/md/compute/ThermodynamicQuantities/potential_energy": (
                _float_or_nan(thermo.potential_energy)
            ),
            "hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy": (
                _float_or_nan(thermo.kinetic_energy)
            ),
            "hoomd-data/md/compute/ThermodynamicQuantities/volume": state.volume,
            "hoomd-data/md/compute/ThermodynamicQuantities/num_particles": (
                state.n_particles
            ),
            "mdsims/time/run_step": int(run_step),
            "mdsims/time/this_lj_time": float(run_step) * float(dt),
            "mdsims/time/cumulative_lj_time": (
                float(prior_lj_time) + float(run_step) * float(dt)
            ),
            "mdsims/output/trajectory_frame_id": trajectory_frame_id,
            "mdsims/analysis/speed/max_particle_speed": (
                float(np.max(speed)) if len(speed) else 0.0
            ),
        }
        self._append_hdf5(values)
        self.samples["run_step"].append(int(run_step))
        self.samples["pressure"].append(values[
            "hoomd-data/md/compute/ThermodynamicQuantities/pressure"
        ])
        self.samples["potential_energy"].append(values[
            "hoomd-data/md/compute/ThermodynamicQuantities/potential_energy"
        ])
        if save_frame:
            self.frame_records.append({
                "trajectory_frame_id": int(trajectory_frame_id),
                "log_index": len(self.samples["run_step"]) - 1,
                "run_step": int(run_step),
                "hoomd_timestep": int(simulation.timestep),
            })
        return state

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def flush(self) -> None:
        if self._hdf5 is not None:
            self._hdf5.flush()

    def close(self) -> None:
        if self._hdf5 is not None:
            self._hdf5.flush()
            self._hdf5.close()
            self._hdf5 = None

    def __enter__(self) -> "RunStorage":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def update_hdf5_metadata(
    path: str | Path,
    groups: dict[str, dict[str, Any]],
) -> None:
    """Update terminal metadata after an active writer has been closed."""

    import h5py

    path = Path(path)
    if not path.exists():
        return
    with h5py.File(path, mode="a") as hdf5:
        for group_path, attributes in groups.items():
            group = hdf5.require_group(group_path)
            for key, value in attributes.items():
                if value is None:
                    continue
                group.attrs[str(key)] = _attribute_value(value)
