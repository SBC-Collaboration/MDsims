"""Build and evolve an FCC liquid with lower-density side reservoirs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .database import EXPANDED_FCC_SIM_TYPE, SQLiteRunDatabase, utc_now
from .lattices import FCC_METHOD_VERSION, build_fcc_lattice
from .paths import ProjectPaths, RunPaths
from .signatures import canonical_json, create_run_signature
from .storage import RunStorage, update_hdf5_metadata
from .thermalization import (
    ThermalizationConfig,
    _failure_update,
    _make_simulation_from_frame,
    _state_metadata,
    thermalization_phase_frame_schedule,
)


EXPANDED_FCC_METHOD_VERSION = "expanded_fcc_side_reservoirs_v3"
EXPANDED_FCC_LATTICE_VERSION = "scaled_fcc_center_thinned_sides_v2"
COM_RECENTER_METHOD_VERSION = "mass_weighted_unwrapped_com_v1"
EXPANDED_FCC_TRAJECTORY_VERSION = "initial_recenter_phase_and_final_v2"


@dataclass(frozen=True)
class ExpandedFCCLattice:
    """An FCC center slab and two identical, thinned side regions."""

    positions: np.ndarray
    n_cells: int
    central_particles: int
    particles_per_side: int
    n_particles: int
    target_density: float
    center_density: float
    center_length_scale: float
    side_density: float
    side_extension: float
    side_density_divisor: float
    original_box_length: float
    box: np.ndarray

    @property
    def volume(self) -> float:
        return float(np.prod(self.box[:3]))

    @property
    def density(self) -> float:
        return self.n_particles / self.volume


@dataclass(frozen=True)
class ExpandedFCCConfig(ThermalizationConfig):
    """Thermalization inputs plus controls for two sparse side regions.

    ``side_extension`` is the length added on *each* side in units of the
    original FCC box length. ``side_density_divisor`` divides the central
    particle density, so its default value of 2 produces half-density sides.
    """

    side_extension: float = 1.0
    side_density_divisor: float = 2.0
    com_recenter_period: int = 10_000
    center_length_scale: float = 1.0

    def validate(self) -> None:
        self._validate_common()
        if float(self.side_extension) <= 0:
            raise ValueError("side_extension must be positive")
        if float(self.side_density_divisor) < 1:
            raise ValueError("side_density_divisor must be at least 1")
        if int(self.com_recenter_period) <= 0:
            raise ValueError("com_recenter_period must be positive")
        if float(self.center_length_scale) <= 0:
            raise ValueError("center_length_scale must be positive")
        thermalization_phase_frame_schedule(self.nsteps, self.log_period)

    def signature_parameters(self) -> dict[str, Any]:
        parameters = super().signature_parameters()
        parameters.update({
            "sim_type": EXPANDED_FCC_SIM_TYPE,
            "workflow_version": EXPANDED_FCC_METHOD_VERSION,
            "state_creation_version": EXPANDED_FCC_LATTICE_VERSION,
            "side_extension": float(self.side_extension),
            "side_density_divisor": float(self.side_density_divisor),
            "center_length_scale": float(self.center_length_scale),
            "com_recenter_period": int(self.com_recenter_period),
            "com_recenter_method": COM_RECENTER_METHOD_VERSION,
            "trajectory_storage": EXPANDED_FCC_TRAJECTORY_VERSION,
        })
        return parameters

    @property
    def run_signature(self) -> str:
        return create_run_signature(self.signature_parameters())


def _evenly_spaced_indices(candidate_count: int, selected_count: int) -> np.ndarray:
    """Choose a deterministic, spatially distributed subset of lattice sites."""

    candidate_count = int(candidate_count)
    selected_count = int(selected_count)
    if not 0 < selected_count <= candidate_count:
        raise ValueError("selected_count must be between 1 and candidate_count")
    return np.floor(
        np.arange(selected_count, dtype=np.float64)
        * candidate_count
        / selected_count
    ).astype(np.int64)


def expanded_fcc_frame_schedule(
    nsteps: int,
    log_period: int,
    com_recenter_period: int = 10_000,
) -> list[dict[str, Any]]:
    """Return the union of COM, phase-analysis, and final saved frames."""

    nsteps = int(nsteps)
    com_recenter_period = int(com_recenter_period)
    if com_recenter_period <= 0:
        raise ValueError("com_recenter_period must be positive")
    phase_steps = {
        int(item["run_step"])
        for item in thermalization_phase_frame_schedule(nsteps, log_period)
    }
    recenter_steps = set(range(
        com_recenter_period,
        nsteps + 1,
        com_recenter_period,
    ))
    saved_steps = sorted(phase_steps | recenter_steps | {nsteps})
    return [
        {
            "trajectory_frame_id": frame_id,
            "run_step": step,
            "com_recenter_frame": step in recenter_steps,
            "phase_frame": step in phase_steps,
            "final_frame": step == nsteps,
        }
        for frame_id, step in enumerate(saved_steps, start=1)
    ]


def build_expanded_fcc_lattice(
    n_cells: int,
    density: float,
    side_extension: float = 1.0,
    side_density_divisor: float = 2.0,
    center_length_scale: float = 1.0,
) -> ExpandedFCCLattice:
    """Build a central FCC box with identical lower-density boxes beside it."""

    side_extension = float(side_extension)
    side_density_divisor = float(side_density_divisor)
    center_length_scale = float(center_length_scale)
    if side_extension <= 0:
        raise ValueError("side_extension must be positive")
    if side_density_divisor < 1:
        raise ValueError("side_density_divisor must be at least 1")
    if center_length_scale <= 0:
        raise ValueError("center_length_scale must be positive")

    center = build_fcc_lattice(n_cells, density)
    length = float(center.box_length)
    center_length = center_length_scale * length
    side_length = side_extension * length

    def tiled_local_sites(length_scale: float) -> np.ndarray:
        """Repeat the original FCC box into a local x interval."""

        region_length = length_scale * length
        tiles = []
        for tile in range(int(np.ceil(length_scale))):
            local = center.positions.copy()
            local[:, 0] += length / 2.0 + tile * length
            tiles.append(local)
        candidates = np.concatenate(tiles, axis=0)
        candidates = candidates[candidates[:, 0] < region_length]
        order = np.lexsort(
            (candidates[:, 2], candidates[:, 1], candidates[:, 0])
        )
        return candidates[order]

    center_local = tiled_local_sites(center_length_scale)
    center_positions = center_local.copy()
    center_positions[:, 0] -= center_length / 2.0

    # Express repeated FCC sites in coordinates local to a side region. The
    # same selected local sites are then translated into both side boxes,
    # making their initial particle arrangements exactly identical.
    candidates = tiled_local_sites(side_extension)

    particles_per_side = int(round(
        center.n_particles * side_extension / side_density_divisor
    ))
    if particles_per_side < 1:
        raise ValueError(
            "side region is too small for even one particle at the requested "
            "density; increase n_fcc_cells or side_extension, or reduce "
            "side_density_divisor"
        )
    selected = candidates[
        _evenly_spaced_indices(len(candidates), particles_per_side)
    ]

    left = selected.copy()
    left[:, 0] += -center_length / 2.0 - side_length
    right = selected.copy()
    right[:, 0] += center_length / 2.0
    positions = np.concatenate((left, center_positions, right), axis=0)
    box = np.array([
        center_length + 2.0 * side_length,
        length,
        length,
        0.0,
        0.0,
        0.0,
    ])
    side_density = particles_per_side / (side_length * length**2)
    center_density = len(center_positions) / (center_length * length**2)
    return ExpandedFCCLattice(
        positions=positions,
        n_cells=int(n_cells),
        central_particles=len(center_positions),
        particles_per_side=particles_per_side,
        n_particles=len(positions),
        target_density=float(density),
        center_density=float(center_density),
        center_length_scale=center_length_scale,
        side_density=float(side_density),
        side_extension=side_extension,
        side_density_divisor=side_density_divisor,
        original_box_length=length,
        box=box,
    )


def make_expanded_fcc_frame(
    lattice: ExpandedFCCLattice,
    particle_type: str = "A",
):
    """Convert an expanded lattice to a GSD/HOOMD snapshot frame."""

    import gsd.hoomd

    frame = gsd.hoomd.Frame()
    frame.configuration.step = 0
    frame.configuration.box = lattice.box
    frame.particles.N = lattice.n_particles
    frame.particles.types = [str(particle_type)]
    frame.particles.position = lattice.positions
    frame.particles.typeid = np.zeros(lattice.n_particles, dtype=np.uint32)
    frame.particles.image = np.zeros((lattice.n_particles, 3), dtype=np.int32)
    frame.particles.mass = np.ones(lattice.n_particles, dtype=np.float64)
    return frame


def recenter_snapshot_arrays(
    positions: np.ndarray,
    images: np.ndarray,
    box: np.ndarray,
    masses: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate a periodic state so its unwrapped, mass-weighted COM is zero."""

    positions = np.asarray(positions, dtype=np.float64)
    images = np.asarray(images, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if images.shape != positions.shape:
        raise ValueError("images must have the same shape as positions")
    if box.shape != (6,):
        raise ValueError("box must contain Lx, Ly, Lz, xy, xz, yz")
    if np.any(box[:3] <= 0):
        raise ValueError("box lengths must be positive")

    if masses is None:
        masses = np.ones(len(positions), dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (len(positions),) or np.any(masses <= 0):
        raise ValueError("masses must be a positive length-N array")

    lx, ly, lz, xy, xz, yz = box
    cell = np.array([
        [lx, 0.0, 0.0],
        [xy * ly, ly, 0.0],
        [xz * lz, yz * lz, lz],
    ])
    unwrapped = positions + images @ cell
    com = np.average(unwrapped, axis=0, weights=masses)
    centered = unwrapped - com
    fractional = centered @ np.linalg.inv(cell)
    centered_images = np.floor(fractional + 0.5).astype(np.int32)
    wrapped = centered - centered_images @ cell
    return wrapped, centered_images, com


def _recenter_simulation_com(simulation) -> np.ndarray:
    snapshot = simulation.state.get_snapshot()
    com = np.zeros(3, dtype=np.float64)
    if snapshot.communicator.rank == 0:
        wrapped, images, com = recenter_snapshot_arrays(
            snapshot.particles.position,
            snapshot.particles.image,
            snapshot.configuration.box,
            snapshot.particles.mass,
        )
        snapshot.particles.position[:] = wrapped
        snapshot.particles.image[:] = images
    simulation.state.set_snapshot(snapshot)
    return com


def _metadata(
    run_id: str,
    config: ExpandedFCCConfig,
    paths: RunPaths,
    lattice: ExpandedFCCLattice,
    device_name: str,
) -> dict[str, dict[str, Any]]:
    relative = paths.relative_directory.as_posix()
    return {
        "mdsims/run": {
            "Run_ID": run_id,
            "Run_Signature": config.run_signature,
            "Sim_Type": EXPANDED_FCC_SIM_TYPE,
            "Status": "Initializing",
            "Workflow_Version": EXPANDED_FCC_METHOD_VERSION,
            "Canonical_Config_JSON": canonical_json(
                config.signature_parameters()
            ),
        },
        "mdsims/source": {
            "State_Role": "source",
            "Source_Type": "Expanded_FCC_Lattice",
            "State_Creation_Method": EXPANDED_FCC_LATTICE_VERSION,
            "Central_Lattice_Method": FCC_METHOD_VERSION,
        },
        "mdsims/protocol": {
            "N_Cells": int(config.n_fcc_cells),
            "Therm_kT": float(config.kT),
            "Therm_Seed": int(config.seed),
            "Density_Target_Center": float(config.target_rho),
            "Center_Length_Scale": float(config.center_length_scale),
            "Center_Length": (
                float(config.center_length_scale)
                * float(lattice.original_box_length)
            ),
            "Center_Density_Actual": float(lattice.center_density),
            "Side_Extension_Per_Side": float(config.side_extension),
            "Side_Density_Divisor": float(config.side_density_divisor),
            "Side_Density_Actual": float(lattice.side_density),
            "COM_Recenter_Period": int(config.com_recenter_period),
            "COM_Recenter_Method": COM_RECENTER_METHOD_VERSION,
            "Nsteps": int(config.nsteps),
            "dt": float(config.dt),
            "Ensemble": "NVT",
            "T_Set": float(config.kT),
            "Particle_Type": config.particle_type,
            "Device": device_name,
            "Always_Compute_Pressure": True,
        },
        "mdsims/interaction": {
            "epsilon_LJ": float(config.epsilon_LJ),
            "sigma_LJ": float(config.sigma_LJ),
            "LJ_r_cut": float(config.r_cut_LJ),
            "LJ_r_on": float(config.r_on_LJ),
            "LJ_Mode": config.lj_mode,
            "Neighbor_Buffer": float(config.buffer_LJ),
        },
        "mdsims/output": {
            "File_Location": relative,
            "Trajectory_Path": f"{relative}/trajectory.gsd",
            "HDF5_Path": f"{relative}/run.hdf5",
            "Log_Period": int(config.log_period),
            "Progress_Update_Period": int(config.progress_period),
            "Trajectory_Storage_Method": EXPANDED_FCC_TRAJECTORY_VERSION,
        },
        "mdsims/states/source": {
            "N_Particles": int(lattice.n_particles),
            "Central_Particles": int(lattice.central_particles),
            "Particles_Per_Side": int(lattice.particles_per_side),
            "Original_Box_Length": float(lattice.original_box_length),
            "Box": lattice.box,
            "Volume": float(lattice.volume),
            "Number_Density_Overall": float(lattice.density),
        },
    }


def run_expanded_fcc(
    config: ExpandedFCCConfig,
    project_paths: ProjectPaths | None = None,
    database: SQLiteRunDatabase | None = None,
) -> dict[str, Any]:
    """Create, evolve, recenter, save, and master-index an expanded FCC state."""

    config.validate()
    project_paths = project_paths or ProjectPaths()
    database = database or SQLiteRunDatabase(project_paths.database)
    database.initialize()
    signature = config.run_signature
    existing = database.check_run_exists(signature)
    if existing is not None:
        return {
            "skipped": True,
            "created_new": False,
            "run_id": existing["Run_ID"],
            "run_signature": signature,
            "status": existing["Status"],
            "message": "Matching Run_Signature already exists; simulation skipped.",
            "existing_run": existing,
        }

    run_id = database.reserve_run_id(max_attempts=3)
    run_paths = project_paths.for_run(EXPANDED_FCC_SIM_TYPE, run_id)
    note = (
        f"Expanded FCC: center length {config.center_length_scale:g}L; "
        f"{config.side_extension:g} original box length(s) added per side; "
        "side density divided by "
        f"{config.side_density_divisor:g}."
    )
    if config.notes and str(config.notes).strip():
        note += f" User note: {str(config.notes).strip()}"
    database.update_master(
        run_id,
        Run_Signature=signature,
        N_Cells=int(config.n_fcc_cells),
        Nsteps=int(config.nsteps),
        Current_Nstep=0,
        ElapsedTime=0.0,
        Last_Update_Time=utc_now(),
        Sim_Type=EXPANDED_FCC_SIM_TYPE,
        Status="Initializing",
        Notes=note,
    )

    current_step = 0
    elapsed_time = 0.0
    storage: RunStorage | None = None
    try:
        lattice = build_expanded_fcc_lattice(
            config.n_fcc_cells,
            config.target_rho,
            config.side_extension,
            config.side_density_divisor,
            config.center_length_scale,
        )
        frame = make_expanded_fcc_frame(lattice, config.particle_type)
        simulation, thermo, device_name = _make_simulation_from_frame(
            config,
            frame,
            thermalize_momenta=True,
            ensemble="NVT",
        )
        frame_schedule = expanded_fcc_frame_schedule(
            config.nsteps,
            config.log_period,
            config.com_recenter_period,
        )
        phase_frame_steps = {
            int(item["run_step"])
            for item in frame_schedule
            if item["phase_frame"]
        }
        saved_frame_steps = {
            int(item["run_step"])
            for item in frame_schedule
        }
        storage = RunStorage(run_paths)
        storage.open(_metadata(
            run_id, config, run_paths, lattice, device_name
        ))

        start_time = utc_now()
        database.update_master(
            run_id,
            StartTime=start_time,
            Last_Update_Time=start_time,
            Status="Running",
        )
        simulation.run(0)
        initial_com = _recenter_simulation_com(simulation)
        state = storage.record(
            simulation, thermo, 0, config.dt, save_frame=True
        )
        trajectory_relative = (
            run_paths.relative_directory / "trajectory.gsd"
        ).as_posix()
        storage.write_metadata({
            "mdsims/run": {"Status": "Running", "StartTime": start_time},
            "mdsims/states/initial": _state_metadata(
                "initial",
                run_id,
                state,
                config.n_fcc_cells,
                0,
                simulation.timestep,
                0,
                config.dt,
                trajectory_relative,
            ),
            "mdsims/analysis/com_recenter": {
                "Initial_COM_Before_Recenter": initial_com,
            },
        })

        recenter_steps: list[int] = []
        com_before_recenter: list[np.ndarray] = []
        next_log = int(config.log_period)
        next_progress = int(config.progress_period)
        next_recenter = int(config.com_recenter_period)
        while current_step < int(config.nsteps):
            target = min(
                next_log,
                next_progress,
                next_recenter,
                int(config.nsteps),
            )
            started = time.perf_counter()
            simulation.run(target - current_step)
            elapsed_time += time.perf_counter() - started
            current_step = target

            recenter_now = current_step == next_recenter
            if recenter_now:
                com_before_recenter.append(_recenter_simulation_com(simulation))
                recenter_steps.append(current_step)
                while next_recenter <= current_step:
                    next_recenter += int(config.com_recenter_period)

            final_now = current_step == int(config.nsteps)
            log_now = current_step == next_log or recenter_now or final_now
            if log_now:
                state = storage.record(
                    simulation,
                    thermo,
                    current_step,
                    config.dt,
                    save_frame=current_step in saved_frame_steps,
                )
                while next_log <= current_step:
                    next_log += int(config.log_period)

            progress_now = current_step == next_progress or final_now
            if progress_now:
                database.update_master(
                    run_id,
                    Current_Nstep=current_step,
                    ElapsedTime=elapsed_time,
                    Last_Update_Time=utc_now(),
                )
                storage.flush()
                while next_progress <= current_step:
                    next_progress += int(config.progress_period)

        expected_frames = 1 + len(frame_schedule)
        if storage.frame_count != expected_frames:
            raise RuntimeError(
                f"Expanded FCC saved {storage.frame_count} frames; "
                f"expected {expected_frames}"
            )

        phase_records = [
            record
            for record in storage.frame_records
            if int(record["run_step"]) in phase_frame_steps
        ]
        if len(phase_records) != 5:
            raise RuntimeError(
                "Expanded FCC did not save the five scheduled phase-analysis "
                "frames"
            )
        phase_frame_ids = [
            int(record["trajectory_frame_id"])
            for record in phase_records
        ]

        end_time = utc_now()
        storage.write_metadata({
            "mdsims/run": {
                "Status": "Complete",
                "EndTime": end_time,
                "ElapsedTime": elapsed_time,
            },
            "mdsims/states/final": _state_metadata(
                "final",
                run_id,
                state,
                config.n_fcc_cells,
                storage.frame_count - 1,
                simulation.timestep,
                current_step,
                config.dt,
                trajectory_relative,
            ),
            "mdsims/analysis/com_recenter": {
                "Recenter_Run_Steps": recenter_steps,
                "COM_Before_Recenter": (
                    np.asarray(com_before_recenter, dtype=np.float64)
                    if com_before_recenter
                    else np.empty((0, 3), dtype=np.float64)
                ),
            },
            "mdsims/output": {
                "Trajectory_Frame_Log_Indices": [
                    record["log_index"] for record in storage.frame_records
                ],
                "Trajectory_Frame_Run_Steps": [
                    record["run_step"] for record in storage.frame_records
                ],
                "Trajectory_Frame_HOOMD_Timesteps": [
                    record["hoomd_timestep"]
                    for record in storage.frame_records
                ],
                "Num_Frames": int(storage.frame_count),
                "Phase_Average_Trajectory_Frame_IDs": phase_frame_ids,
                "Phase_Average_Run_Steps": [
                    int(record["run_step"])
                    for record in phase_records
                ],
            },
        })
        storage.close()
        database.update_master(
            run_id,
            Current_Nstep=current_step,
            ElapsedTime=elapsed_time,
            EndTime=end_time,
            Last_Update_Time=end_time,
            Status="Complete",
            Stop_Reason=None,
            Status_Message=None,
        )
        return {
            "skipped": False,
            "created_new": True,
            "run_id": run_id,
            "run_signature": signature,
            "status": "Complete",
            "directory": run_paths.directory,
            "trajectory_path": run_paths.trajectory,
            "hdf5_path": run_paths.hdf5,
            "n_particles": lattice.n_particles,
            "central_particles": lattice.central_particles,
            "center_density": lattice.center_density,
            "center_length_scale": lattice.center_length_scale,
            "particles_per_side": lattice.particles_per_side,
            "side_density": lattice.side_density,
            "num_frames": storage.frame_count,
            "recenter_steps": recenter_steps,
        }
    except KeyboardInterrupt as error:
        if storage is not None:
            storage.close()
            update_hdf5_metadata(run_paths.hdf5, {
                "mdsims/run": {
                    "Status": "Cancelled",
                    "Status_Message": "KeyboardInterrupt",
                }
            })
        _failure_update(
            database, run_id, current_step, elapsed_time, error, "Cancelled"
        )
        raise
    except Exception as error:
        if storage is not None:
            storage.close()
            update_hdf5_metadata(run_paths.hdf5, {
                "mdsims/run": {
                    "Status": "Failed",
                    "Status_Message": f"{type(error).__name__}: {error}",
                }
            })
        try:
            _failure_update(
                database, run_id, current_step, elapsed_time, error, "Failed"
            )
        except Exception as database_error:
            error.add_note(
                f"The failure could not be written to SQL: {database_error}"
            )
        raise
