"""Resolve and inspect any saved V4 run by its global Run_ID."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .database import SQLiteRunDatabase
from .paths import ProjectPaths
from .visualization import (
    animate_xy_frames,
    plot_log_dataframe,
    plot_phase_histogram,
    plot_xy_frames,
    render_frame,
    render_frames_movie,
)
from .voxel_fit import (
    averaged_trajectory_voxel_histogram,
    fit_trajectory_voxel_mixture,
)


SIM_TYPE_DIRECTORIES = {
    "Thermalization": "Thermalization",
    "Cavitation": "Cavitation",
    "Expanded_FCC": "Expanded_FCC",
    "Excitation_NVE": "Excitation",
    "Excitation_NPH": "Excitation",
}


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


class RunAnalysis:
    """A lazy notebook interface to one SQL-indexed simulation run."""

    def __init__(
        self,
        run_id: str,
        project_paths: ProjectPaths | None = None,
        database: SQLiteRunDatabase | None = None,
    ):
        self.run_id = str(run_id)
        self.project_paths = project_paths or ProjectPaths()
        self.database = database or SQLiteRunDatabase(self.project_paths.database)
        if not self.database.path.exists():
            raise FileNotFoundError(f"SQL database was not found: {self.database.path}")
        self.master_row = self.database.get_run(self.run_id)
        if self.master_row is None:
            raise KeyError(f"Run_ID was not found in MD_Master: {self.run_id}")
        self.sim_type = self.master_row.get("Sim_Type")
        self.state_row = self._load_state_row()

        file_location = self.state_row.get("File_Location") if self.state_row else None
        if file_location:
            directory = Path(file_location).expanduser()
            if not directory.is_absolute():
                directory = self.project_paths.top_directory / directory
        else:
            folder = SIM_TYPE_DIRECTORIES.get(self.sim_type, self.sim_type)
            if not folder:
                raise ValueError(
                    f"Run {self.run_id} does not yet have a Sim_Type or File_Location"
                )
            directory = self.project_paths.top_directory / folder / self.run_id
        self.directory = directory.resolve()
        self.trajectory_path = self.directory / "trajectory.gsd"
        self.hdf5_path = self.directory / "run.hdf5"

    def _load_state_row(self) -> dict[str, Any] | None:
        if self.sim_type == "Thermalization":
            rows = self.database.query_thermalizations(Run_ID=self.run_id, limit=1)
            return rows[0] if rows else None
        if self.sim_type == "Cavitation":
            rows = self.database.query_cavitations(Run_ID=self.run_id, limit=1)
            return rows[0] if rows else None
        # Excitation tables will use the same lookup contract when added.
        return None

    def _require_trajectory(self) -> None:
        if not self.trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory was not found: {self.trajectory_path}")

    def _require_hdf5(self) -> None:
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 log was not found: {self.hdf5_path}")

    @property
    def frame_count(self) -> int:
        self._require_trajectory()
        import gsd.hoomd

        with gsd.hoomd.open(name=str(self.trajectory_path), mode="r") as trajectory:
            return len(trajectory)

    @property
    def started_from_lattice(self) -> bool:
        """Whether this run directly initialized a fresh FCC lattice."""

        return (
            self.sim_type == "Thermalization"
            and self.state_row is not None
            and self.state_row.get("Clone_Run_ID") is None
        )

    def info(self) -> dict[str, Any]:
        """Display SQL rows, resolved paths, status, and frame count."""

        from IPython.display import display
        import pandas as pd

        summary = {
            "Run_ID": self.run_id,
            "Sim_Type": self.sim_type,
            "Status": self.master_row.get("Status"),
            "Directory": str(self.directory),
            "Trajectory": str(self.trajectory_path),
            "HDF5": str(self.hdf5_path),
            "Frame_Count": self.frame_count if self.trajectory_path.exists() else None,
        }
        display(pd.DataFrame([summary]).style.hide(axis="index"))
        display(pd.DataFrame([self.master_row]).style.hide(axis="index"))
        if self.state_row is not None:
            display(pd.DataFrame([self.state_row]).style.hide(axis="index"))
        return summary

    def load_frame(self, frame: int = -1):
        """Load one GSD frame; 0 is initial and -1 is final."""

        self._require_trajectory()
        import gsd.hoomd

        with gsd.hoomd.open(name=str(self.trajectory_path), mode="r") as trajectory:
            return trajectory[int(frame)]

    def load_frames(self, frames: Iterable[int]):
        """Load an explicit sequence of GSD frame indices."""

        self._require_trajectory()
        indices = [int(index) for index in frames]
        import gsd.hoomd

        with gsd.hoomd.open(name=str(self.trajectory_path), mode="r") as trajectory:
            return [trajectory[index] for index in indices]

    def _sampled_indices(self, frame_step: int) -> list[int]:
        frame_step = int(frame_step)
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        count = self.frame_count
        indices = list(range(0, count, frame_step))
        if indices[-1] != count - 1:
            indices.append(count - 1)
        return indices

    def render(self, frame: int = -1, samples: int = 2_000):
        image = render_frame(self.load_frame(frame), samples=samples)
        from IPython.display import display

        display(image)
        return image

    def render_movie(
        self,
        frame_step: int = 1,
        duration: int = 200,
        samples: int = 500,
    ):
        indices = self._sampled_indices(frame_step)
        image = render_frames_movie(
            self.load_frames(indices),
            duration=duration,
            samples=samples,
        )
        from IPython.display import display

        display(image)
        return image

    def xy_slice(
        self,
        frame: int = -1,
        frames: Iterable[int] | None = None,
        z: float = 0.0,
        thickness: float | None = None,
        fraction: float = 0.05,
        point_size: float = 1.0,
        alpha: float = 0.7,
    ):
        indices = [int(frame)] if frames is None else [int(index) for index in frames]
        return plot_xy_frames(
            self.load_frames(indices),
            frame_labels=indices,
            z=z,
            thickness=thickness,
            fraction=fraction,
            point_size=point_size,
            alpha=alpha,
        )

    def xy_slice_movie(
        self,
        frame_step: int = 1,
        z: float = 0.0,
        thickness: float | None = None,
        fraction: float = 0.05,
        point_size: float = 1.0,
        alpha: float = 0.7,
        interval: int = 200,
    ):
        indices = self._sampled_indices(frame_step)
        movie = animate_xy_frames(
            self.load_frames(indices),
            frame_labels=indices,
            z=z,
            thickness=thickness,
            fraction=fraction,
            point_size=point_size,
            alpha=alpha,
            interval=interval,
        )
        from IPython.display import display

        display(movie)
        return movie

    def metadata(self) -> dict[str, Any]:
        """Return all HDF5 attributes keyed by their complete group path."""

        self._require_hdf5()
        import h5py

        result: dict[str, Any] = {}
        with h5py.File(self.hdf5_path, mode="r") as hdf5:
            def collect(name, obj):
                for key, value in obj.attrs.items():
                    result[f"{name}/{key}".strip("/")] = _decode(value)

            hdf5.visititems(collect)
        return result

    def logs_dataframe(self):
        """Return synchronized HDF5 log samples as a pandas DataFrame."""

        self._require_hdf5()
        import h5py
        import pandas as pd

        paths = {
            "hoomd_timestep": "hoomd-data/Simulation/timestep",
            "tps": "hoomd-data/Simulation/tps",
            "kinetic_temperature": (
                "hoomd-data/md/compute/ThermodynamicQuantities/kinetic_temperature"
            ),
            "pressure": "hoomd-data/md/compute/ThermodynamicQuantities/pressure",
            "potential_energy": (
                "hoomd-data/md/compute/ThermodynamicQuantities/potential_energy"
            ),
            "kinetic_energy": (
                "hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy"
            ),
            "volume": "hoomd-data/md/compute/ThermodynamicQuantities/volume",
            "num_particles": (
                "hoomd-data/md/compute/ThermodynamicQuantities/num_particles"
            ),
            "run_step": "mdsims/time/run_step",
            "this_lj_time": "mdsims/time/this_lj_time",
            "cumulative_lj_time": "mdsims/time/cumulative_lj_time",
            "trajectory_frame_id": "mdsims/output/trajectory_frame_id",
            "max_particle_speed": "mdsims/analysis/speed/max_particle_speed",
        }
        with h5py.File(self.hdf5_path, mode="r") as hdf5:
            data = {
                name: np.asarray(hdf5[path])
                for name, path in paths.items()
                if path in hdf5
            }
            tensor_path = (
                "hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor"
            )
            if tensor_path in hdf5:
                tensor = np.asarray(hdf5[tensor_path])
                for index, suffix in enumerate(("xx", "xy", "xz", "yy", "yz", "zz")):
                    data[f"pressure_{suffix}"] = tensor[:, index]
        frame = pd.DataFrame(data)
        if {"potential_energy", "num_particles"} <= set(frame):
            frame["potential_energy_per_particle"] = (
                frame["potential_energy"] / frame["num_particles"]
            )
            frame["PE_per_particle"] = frame["potential_energy_per_particle"]
        if {"kinetic_energy", "num_particles"} <= set(frame):
            frame["kinetic_energy_per_particle"] = (
                frame["kinetic_energy"] / frame["num_particles"]
            )
        if {"num_particles", "volume"} <= set(frame):
            frame["density"] = frame["num_particles"] / frame["volume"]
        frame.insert(0, "log_index", np.arange(len(frame), dtype=int))
        if "trajectory_frame_id" not in frame:
            raise KeyError(
                "run.hdf5 is missing the required trajectory_frame_id mapping"
            )
        frame.insert(1, "frame", frame["trajectory_frame_id"])
        return frame

    def frame_table(self):
        """Return only HDF5 rows that have a corresponding GSD frame."""

        logs = self.logs_dataframe()
        return logs.loc[logs["trajectory_frame_id"] >= 0].reset_index(drop=True)

    def plot_logs(
        self,
        quantities=None,
        x: str = "run_step",
        skip_lattice_transient: bool = True,
        lattice_skip_points: int = 10,
    ):
        """Plot logs and mark the frames used by the averaged voxel histogram."""

        if quantities is None:
            quantities = [
                "kinetic_temperature",
                "pressure",
                "potential_energy_per_particle",
                "volume",
                "max_particle_speed",
            ]
        lattice_skip_points = int(lattice_skip_points)
        if lattice_skip_points < 0:
            raise ValueError("lattice_skip_points cannot be negative")
        skipped = {}
        if skip_lattice_transient and self.started_from_lattice:
            skipped = {
                "pressure": lattice_skip_points,
                "potential_energy_per_particle": lattice_skip_points,
                "PE_per_particle": lattice_skip_points,
            }
        logs = self.logs_dataframe()
        try:
            phase_frame_ids = self.phase_average_frame_ids()
        except (FileNotFoundError, KeyError):
            # In-progress and legacy runs may not have phase-average metadata yet.
            phase_frame_ids = []
        return plot_log_dataframe(
            logs,
            quantities,
            x=x,
            skip_initial_by_quantity=skipped,
            highlight_frame_ids=phase_frame_ids,
        )

    def _phase_fit_attributes(self) -> dict[str, Any]:
        self._require_hdf5()
        import h5py

        with h5py.File(self.hdf5_path, mode="r") as hdf5:
            path = "mdsims/analysis/phase_fit"
            if path not in hdf5:
                return {}
            return {key: _decode(value) for key, value in hdf5[path].attrs.items()}

    def phase_average_frame_ids(self) -> list[int]:
        """Return the exact GSD frames used for the averaged histogram."""

        metadata = self.metadata()
        saved = metadata.get(
            "mdsims/output/Phase_Average_Trajectory_Frame_IDs"
        )
        if saved is None:
            raise KeyError(
                "run.hdf5 is missing Phase_Average_Trajectory_Frame_IDs"
            )
        return [int(index) for index in np.atleast_1d(saved)]

    def plot_phase_fit(self, recompute_missing: bool = True):
        """Plot the exact saved phase-fit data, or clearly label reconstruction."""

        self._require_trajectory()
        fit = self._phase_fit_attributes() if self.hdf5_path.exists() else {}
        frame_ids = self.phase_average_frame_ids()
        exact_saved = "observed_counts" in fit and "density_axis" in fit
        if not exact_saved:
            phase_fit_status = (
                self.state_row.get("Phase_Fit_Status") if self.state_row else fit.get("status")
            )
            should_recompute = (
                phase_fit_status == "Complete"
                or self.sim_type == "Expanded_FCC"
            )
            if should_recompute and recompute_missing:
                fit = fit_trajectory_voxel_mixture(
                    self.trajectory_path,
                    int(self.master_row["N_Cells"]),
                    frame_indices=frame_ids,
                )
                title = (
                    "Expanded FCC averaged histogram and on-demand fit"
                    if self.sim_type == "Expanded_FCC"
                    else "Reconstructed averaged histogram and fit"
                )
            else:
                fit = averaged_trajectory_voxel_histogram(
                    self.trajectory_path,
                    int(self.master_row["N_Cells"]),
                    frame_indices=frame_ids,
                )
                reason = phase_fit_status or "fit data unavailable"
                title = f"Averaged voxel histogram — no fitted curve ({reason})"
        else:
            title = "Saved averaged voxel histogram and phase fit"
        return plot_phase_histogram(fit, title=title)

    def density_profile(
        self,
        frame: int = -1,
        num_slices: int = 60,
        axis: str = "x",
    ):
        """Return number density in equal-volume slabs of one saved frame."""

        import pandas as pd

        axis = str(axis).lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        num_slices = int(num_slices)
        if num_slices <= 0:
            raise ValueError("num_slices must be positive")

        saved = self.load_frame(frame)
        box = np.asarray(saved.configuration.box, dtype=float)
        if box.shape != (6,) or np.any(box[:3] <= 0):
            raise ValueError("frame has an invalid HOOMD box")
        if not np.allclose(box[3:], 0.0):
            raise ValueError(
                "density_profile currently requires an orthorhombic box"
            )

        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        length = float(box[axis_index])
        positions = np.asarray(saved.particles.position, dtype=float)
        coordinate = (
            positions[:, axis_index] + length / 2.0
        ) % length - length / 2.0
        edges = np.linspace(-length / 2.0, length / 2.0, num_slices + 1)
        counts, _ = np.histogram(coordinate, bins=edges)
        slice_volume = float(np.prod(box[:3])) / num_slices
        return pd.DataFrame({
            "slice": np.arange(num_slices, dtype=int),
            f"{axis}_lower": edges[:-1],
            f"{axis}_center": 0.5 * (edges[:-1] + edges[1:]),
            f"{axis}_upper": edges[1:],
            "particle_count": counts.astype(int),
            "slice_volume": np.full(num_slices, slice_volume),
            "number_density": counts / slice_volume,
        })

    def plot_density_profile(
        self,
        frame: int = -1,
        num_slices: int = 60,
        axis: str = "x",
        show_reference: bool = True,
    ):
        """Plot slab number density and return ``(figure, profile_table)``."""

        import matplotlib.pyplot as plt

        axis = str(axis).lower()
        profile = self.density_profile(
            frame=frame,
            num_slices=num_slices,
            axis=axis,
        )
        center_column = f"{axis}_center"
        figure, plot_axis = plt.subplots(figsize=(10, 4.5))
        plot_axis.step(
            profile[center_column].to_numpy(),
            profile["number_density"].to_numpy(),
            where="mid",
            color="black",
            linewidth=1.7,
            label="Saved-frame slab density",
        )

        if show_reference and self.sim_type == "Expanded_FCC" and axis == "x":
            metadata = self.metadata()
            original_length = metadata.get(
                "mdsims/states/source/Original_Box_Length"
            )
            center_density = metadata.get(
                "mdsims/protocol/Density_Target_Center"
            )
            side_density = metadata.get(
                "mdsims/protocol/Side_Density_Actual"
            )
            if original_length is not None:
                half = float(original_length) / 2.0
                plot_axis.axvline(
                    -half,
                    color="tab:blue",
                    linestyle="--",
                    linewidth=1.2,
                    label="Initial center-region boundaries",
                )
                plot_axis.axvline(
                    half,
                    color="tab:blue",
                    linestyle="--",
                    linewidth=1.2,
                )
            if center_density is not None:
                plot_axis.axhline(
                    float(center_density),
                    color="tab:green",
                    linestyle=":",
                    linewidth=1.4,
                    label="Initial center density",
                )
            if side_density is not None:
                plot_axis.axhline(
                    float(side_density),
                    color="tab:orange",
                    linestyle=":",
                    linewidth=1.4,
                    label="Initial side density",
                )

        plot_axis.set_xlabel(f"{axis} position")
        plot_axis.set_ylabel("Number density")
        plot_axis.set_title(f"Frame {frame}: density in {num_slices} {axis}-slabs")
        plot_axis.grid(alpha=0.3)
        plot_axis.legend()
        figure.tight_layout()
        plt.show()
        return figure, profile


def open_run(
    run_id: str,
    project_paths: ProjectPaths | None = None,
    database: SQLiteRunDatabase | None = None,
) -> RunAnalysis:
    """Resolve a run from SQL without loading its GSD or HDF5 contents."""

    return RunAnalysis(run_id, project_paths=project_paths, database=database)
