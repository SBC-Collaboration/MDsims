"""Seed a spherical void in a thermalized state and evolve it in NVT."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .analysis import (
    classify_pe_drop,
    classify_voxel_histogram,
    select_phase_classification,
    thermodynamic_summary,
)
from .database import SQLiteRunDatabase, utc_now
from .paths import ProjectPaths, RunPaths
from .signatures import canonical_json, create_run_signature
from .storage import RunStorage, StateData, update_hdf5_metadata
from .thermalization import (
    CloneRescaleThermalizationConfig,
    _combined_note,
    _failure_update,
    _frame_state_data,
    _inherited_clone_config,
    _make_simulation_from_frame,
    _metadata_value,
    _state_metadata,
    thermalization_log_steps,
    thermalization_phase_frame_schedule,
)
from .voxel_fit import conditional_phase_fit, phase_fit_sql_values


CAVITATION_METHOD_VERSION = "cavitation_spherical_mask_v1"
CAVITATION_LOCATION_METHOD_VERSION = "uniform_box_translation_v1"
CAVITATION_TRAJECTORY_METHOD_VERSION = (
    "initial_plus_log_stride_20_plus_terminal_5_stride_10_v1"
)
CAVITATION_LOG_FRAME_STRIDE = 20
CAVITATION_MIN_LOG_SAMPLES = 41
CAVITATION_MAX_DIAMETER_BOX_FRACTION = 0.85


@dataclass(frozen=True)
class CavitationConfig:
    """Inputs that differ from the completed source thermalization."""

    source_run_id: str
    mask_radius: float
    nsteps: int
    ensemble: str = "NVT"
    random_location: bool = False
    location_seed: int | None = None
    notes: str | None = None

    @property
    def normalized_ensemble(self) -> str:
        return str(self.ensemble).upper()

    def validate(self) -> None:
        if not str(self.source_run_id):
            raise ValueError("source_run_id is required")
        if float(self.mask_radius) <= 0:
            raise ValueError("mask_radius must be positive")
        if int(self.nsteps) <= 0:
            raise ValueError("nsteps must be positive")
        if self.normalized_ensemble != "NVT":
            raise ValueError("cavitation currently supports ensemble='NVT' only")
        if bool(self.random_location):
            if self.location_seed is None:
                raise ValueError(
                    "location_seed is required when random_location=True"
                )
            if int(self.location_seed) < 0:
                raise ValueError("location_seed cannot be negative")
        elif self.location_seed is not None:
            raise ValueError(
                "location_seed must be None when random_location=False"
            )

    def signature_parameters(self, source_frame_id: int) -> dict[str, Any]:
        return {
            "sim_type": "Cavitation",
            "workflow_version": CAVITATION_METHOD_VERSION,
            "source_run_id": str(self.source_run_id),
            "source_frame_id": int(source_frame_id),
            "mask_radius": float(self.mask_radius),
            "random_location": bool(self.random_location),
            "location_seed": (
                int(self.location_seed) if self.location_seed is not None else None
            ),
            "location_method": CAVITATION_LOCATION_METHOD_VERSION,
            "nsteps": int(self.nsteps),
            "ensemble": self.normalized_ensemble,
            "simulation_settings": "inherit_source_thermalization",
            "log_period": "inherit_source_thermalization",
            "trajectory_storage": CAVITATION_TRAJECTORY_METHOD_VERSION,
        }

    def run_signature(self, source_frame_id: int) -> str:
        return create_run_signature(self.signature_parameters(source_frame_id))


def cavitation_frame_schedule(
    nsteps: int,
    log_period: int,
) -> list[dict[str, Any]]:
    """Save every twentieth log plus the five thermalization terminal logs."""

    log_steps = thermalization_log_steps(nsteps, log_period)
    terminal = thermalization_phase_frame_schedule(nsteps, log_period)
    terminal_ordinals = {item["log_ordinal"] for item in terminal}
    result = []
    for ordinal, run_step in enumerate(log_steps, start=1):
        bulk = ordinal % CAVITATION_LOG_FRAME_STRIDE == 0
        phase = ordinal in terminal_ordinals
        if bulk or phase:
            result.append({
                "log_ordinal": ordinal,
                "run_step": run_step,
                "bulk_frame": bulk,
                "phase_frame": phase,
            })
    return result


def _source_context(
    config: CavitationConfig,
    database: SQLiteRunDatabase,
) -> tuple[dict[str, Any], dict[str, Any], int] | dict[str, Any]:
    source_master = database.get_run(config.source_run_id)
    if source_master is None:
        raise KeyError(f"Source Run_ID was not found: {config.source_run_id}")
    if source_master.get("Sim_Type") != "Thermalization":
        raise ValueError("Cavitation source must be a Thermalization run")
    if source_master.get("Status") != "Complete":
        raise ValueError("Cavitation source must have Status='Complete'")
    source = database.get_thermalization(config.source_run_id)
    if source is None:
        raise ValueError(
            "Cavitation source has no completed Thermalization table row"
        )
    phase_status = source.get("Phase_Separation_Status")
    if phase_status == "Separated":
        return {
            "skipped": True,
            "created_new": False,
            "skip_reason": "source_phase_separated",
            "source_run_id": str(config.source_run_id),
            "status": "Skipped",
            "message": "Source thermalization is phase separated; cavitation skipped.",
        }
    if phase_status != "Not_Separated":
        raise ValueError(
            "Cavitation source must have a known homogeneous phase "
            "classification (Phase_Separation_Status='Not_Separated')"
        )
    source_frame_id = int(source["Num_Frames"]) - 1
    if source_frame_id < 0:
        raise ValueError("Cavitation source contains no saved GSD frames")
    return source_master, source, source_frame_id


def _random_center(
    box: np.ndarray,
    random_location: bool,
    location_seed: int | None,
) -> np.ndarray:
    lengths = np.asarray(box, dtype=np.float64)[:3]
    if not bool(random_location):
        return np.zeros(3, dtype=np.float64)
    rng = np.random.default_rng(int(location_seed))
    return rng.uniform(-lengths / 2.0, lengths / 2.0)


def shift_and_mask_positions(
    positions: np.ndarray,
    box: np.ndarray,
    mask_radius: float,
    *,
    random_location: bool = False,
    location_seed: int | None = None,
) -> dict[str, Any]:
    """Center the requested periodic location and return the spherical mask."""

    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    lengths = box[:3]
    radius = float(mask_radius)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if np.any(lengths <= 0):
        raise ValueError("box lengths must be positive")
    if radius <= 0:
        raise ValueError("mask_radius must be positive")
    maximum_radius = (
        CAVITATION_MAX_DIAMETER_BOX_FRACTION * float(np.min(lengths)) / 2.0
    )
    if radius > maximum_radius:
        raise ValueError(
            "mask diameter cannot exceed 85% of the smallest box length: "
            f"radius {radius} > {maximum_radius}"
        )
    center = _random_center(box, random_location, location_seed)
    shifted = positions - center
    shifted = (shifted + lengths / 2.0) % lengths - lengths / 2.0
    removed = np.linalg.norm(shifted, axis=1) < radius
    removed_count = int(np.count_nonzero(removed))
    if removed_count == 0:
        raise ValueError("spherical mask removed zero particles")
    if removed_count == len(positions):
        raise ValueError("spherical mask removed every particle")
    return {
        "shifted_positions": shifted,
        "keep_mask": ~removed,
        "removed_mask": removed,
        "sampled_center": center,
        "translation": -center,
        "particles_before": int(len(positions)),
        "particles_removed": removed_count,
        "particles_after": int(len(positions) - removed_count),
    }


def _masked_frame(source_frame, config: CavitationConfig):
    import gsd.hoomd

    prepared = shift_and_mask_positions(
        source_frame.particles.position,
        source_frame.configuration.box,
        config.mask_radius,
        random_location=config.random_location,
        location_seed=config.location_seed,
    )
    keep = prepared["keep_mask"]
    frame = gsd.hoomd.Frame()
    frame.configuration.step = 0
    frame.configuration.box = np.asarray(
        source_frame.configuration.box,
        dtype=np.float64,
    ).copy()
    frame.particles.N = prepared["particles_after"]
    frame.particles.types = list(source_frame.particles.types)
    frame.particles.position = prepared["shifted_positions"][keep]
    for name in ("velocity", "typeid", "mass"):
        values = np.asarray(getattr(source_frame.particles, name))
        if len(values) == prepared["particles_before"]:
            setattr(frame.particles, name, values[keep].copy())
    frame.particles.image = np.zeros(
        (prepared["particles_after"], 3),
        dtype=np.int32,
    )
    return frame, prepared


def _base_metadata(
    run_id: str,
    signature: str,
    config: CavitationConfig,
    simulation_config,
    run_paths: RunPaths,
    device_name: str,
    source: dict[str, Any],
    source_frame_id: int,
    source_state: StateData,
    initial_state: StateData,
    mask: dict[str, Any],
    frame_schedule: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    relative = run_paths.relative_directory.as_posix()
    phase_ordinals = [
        item["log_ordinal"] for item in frame_schedule if item["phase_frame"]
    ]
    return {
        "mdsims/run": {
            "Run_ID": run_id,
            "Run_Signature": signature,
            "Sim_Type": "Cavitation",
            "Status": "Initializing",
            "Workflow_Version": CAVITATION_METHOD_VERSION,
            "Canonical_Config_JSON": canonical_json(
                config.signature_parameters(source_frame_id)
            ),
        },
        "mdsims/source": {
            "State_Role": "source",
            "Source_Type": "Thermalization_Run",
            "Source_Run_ID": str(config.source_run_id),
            "Source_Frame_ID": int(source_frame_id),
            "Source_File_Location": str(source["File_Location"]),
            "N_Particles": int(source_state.n_particles),
            "Number_Density": float(source_state.density),
            "Box": source_state.box,
        },
        "mdsims/protocol": {
            "N_Cells": int(simulation_config.n_fcc_cells),
            "Therm_kT": float(simulation_config.kT),
            "Therm_Seed": int(simulation_config.seed),
            "Nsteps": int(config.nsteps),
            "dt": float(simulation_config.dt),
            "Ensemble": config.normalized_ensemble,
            "T_Set": float(simulation_config.kT),
            "Particle_Type": simulation_config.particle_type,
            "Device": device_name,
            "Always_Compute_Pressure": True,
            "Inherited_Simulation_Settings": True,
        },
        "mdsims/protocol/cavitation_mask": {
            "Mask_Radius": float(config.mask_radius),
            "Maximum_Diameter_Box_Fraction": (
                CAVITATION_MAX_DIAMETER_BOX_FRACTION
            ),
            "Random_Location": bool(config.random_location),
            "Location_Seed": config.location_seed,
            "Sampled_Center_Pre_Shift": mask["sampled_center"],
            "Translation_Vector": mask["translation"],
            "Particles_Before": mask["particles_before"],
            "Particles_Removed": mask["particles_removed"],
            "Particles_After": mask["particles_after"],
            "Position_Boundary_Rule": "distance < mask_radius",
            "Velocity_Adjustment": "none",
        },
        "mdsims/interaction": {
            "epsilon_LJ": float(simulation_config.epsilon_LJ),
            "sigma_LJ": float(simulation_config.sigma_LJ),
            "LJ_r_cut": float(simulation_config.r_cut_LJ),
            "LJ_r_on": float(simulation_config.r_on_LJ),
            "LJ_Mode": simulation_config.lj_mode,
            "Neighbor_Buffer": float(simulation_config.buffer_LJ),
        },
        "mdsims/output": {
            "File_Location": relative,
            "Trajectory_Path": f"{relative}/trajectory.gsd",
            "HDF5_Path": f"{relative}/run.hdf5",
            "Log_Period": int(simulation_config.log_period),
            "Progress_Update_Period": int(simulation_config.progress_period),
            "Trajectory_Storage_Method": CAVITATION_TRAJECTORY_METHOD_VERSION,
            "Bulk_Frame_Log_Stride": CAVITATION_LOG_FRAME_STRIDE,
            "Trajectory_Frame_Log_Ordinals": [
                item["log_ordinal"] for item in frame_schedule
            ],
            "Phase_Average_Log_Ordinals": phase_ordinals,
        },
        "mdsims/time": {
            "Prior_Cumulative_LJ_Time": float(source["Cumulative_LJ_Time"]),
        },
        "mdsims/states/initial_masked": {
            "State_Role": "initial",
            "N_Particles": int(initial_state.n_particles),
            "Box": initial_state.box,
            "Volume": float(initial_state.volume),
            "Number_Density": float(initial_state.density),
        },
    }


def run_cavitation(
    config: CavitationConfig,
    project_paths: ProjectPaths | None = None,
    database: SQLiteRunDatabase | None = None,
) -> dict[str, Any]:
    """Create or skip one spherical-mask cavitation simulation."""

    config.validate()
    project_paths = project_paths or ProjectPaths()
    database = database or SQLiteRunDatabase(project_paths.database)
    database.initialize()

    context = _source_context(config, database)
    if isinstance(context, dict):
        return context
    source_master, source, source_frame_id = context
    signature = config.run_signature(source_frame_id)
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

    from .run_analysis import open_run

    source_run = open_run(
        config.source_run_id,
        project_paths=project_paths,
        database=database,
    )
    if source_run.frame_count != int(source["Num_Frames"]):
        raise RuntimeError("Source GSD frame count does not match the SQL record")
    source_frame = source_run.load_frame(source_frame_id)
    source_state = _frame_state_data(source_frame)
    source_metadata = source_run.metadata()
    inherited_request = CloneRescaleThermalizationConfig(
        source_run_id=config.source_run_id,
        final_density=source_state.density,
        nsteps=config.nsteps,
        ensemble="NVT",
        notes=config.notes,
    )
    simulation_config = _inherited_clone_config(
        inherited_request,
        source_master,
        source,
        source_metadata,
        source_state,
    )
    minimum_nsteps = CAVITATION_MIN_LOG_SAMPLES * simulation_config.log_period
    if int(config.nsteps) < minimum_nsteps:
        raise ValueError(
            "Cavitation requires nsteps >= 41 * inherited log_period "
            f"({minimum_nsteps})"
        )
    frame_schedule = cavitation_frame_schedule(
        config.nsteps,
        simulation_config.log_period,
    )
    saved_steps = {item["run_step"] for item in frame_schedule}
    phase_steps = {
        item["run_step"] for item in frame_schedule if item["phase_frame"]
    }
    masked_frame, mask = _masked_frame(source_frame, config)
    initial_state = _frame_state_data(masked_frame)

    automatic_note = (
        f"Seeded a centered spherical cavity of radius {config.mask_radius:g} "
        f"from final frame {source_frame_id} of thermalization Run_ID "
        f"{config.source_run_id}; evolved in NVT for {config.nsteps} steps."
    )
    master_note = _combined_note(automatic_note, config.notes)
    run_id = database.reserve_run_id(max_attempts=3)
    run_paths = project_paths.for_run("Cavitation", run_id)
    database.update_master(
        run_id,
        Run_Signature=signature,
        N_Cells=int(source_master["N_Cells"]),
        Nsteps=int(config.nsteps),
        Current_Nstep=0,
        ElapsedTime=0.0,
        Last_Update_Time=utc_now(),
        Sim_Type="Cavitation",
        Status="Initializing",
        Notes=master_note,
    )
    database.add_run_dependency(
        config.source_run_id,
        run_id,
        "cavitation_source",
    )

    current_step = 0
    elapsed_time = 0.0
    storage: RunStorage | None = None
    try:
        simulation, thermo, device_name = _make_simulation_from_frame(
            simulation_config,
            masked_frame,
            thermalize_momenta=False,
            ensemble="NVT",
        )
        prior_lj_time = float(source["Cumulative_LJ_Time"])
        metadata = _base_metadata(
            run_id,
            signature,
            config,
            simulation_config,
            run_paths,
            device_name,
            source,
            source_frame_id,
            source_state,
            initial_state,
            mask,
            frame_schedule,
        )
        storage = RunStorage(run_paths)
        storage.open(metadata)
        start_time = utc_now()
        database.update_master(
            run_id,
            StartTime=start_time,
            Last_Update_Time=start_time,
            Status="Running",
        )

        simulation.run(0)
        state = storage.record(
            simulation,
            thermo,
            0,
            simulation_config.dt,
            prior_lj_time=prior_lj_time,
            save_frame=True,
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
                simulation_config.n_fcc_cells,
                0,
                0,
                0,
                simulation_config.dt,
                trajectory_relative,
                prior_lj_time=prior_lj_time,
            ),
        })

        next_log = min(simulation_config.log_period, config.nsteps)
        next_progress = min(simulation_config.progress_period, config.nsteps)
        while current_step < int(config.nsteps):
            target = min(next_log, next_progress, int(config.nsteps))
            started = time.perf_counter()
            simulation.run(target - current_step)
            elapsed_time += time.perf_counter() - started
            current_step = target
            log_now = current_step == next_log or current_step == config.nsteps
            progress_now = (
                current_step == next_progress or current_step == config.nsteps
            )
            if log_now:
                state = storage.record(
                    simulation,
                    thermo,
                    current_step,
                    simulation_config.dt,
                    prior_lj_time=prior_lj_time,
                    save_frame=current_step in saved_steps,
                )
                while next_log <= current_step:
                    next_log += int(simulation_config.log_period)
            if progress_now:
                database.update_master(
                    run_id,
                    Current_Nstep=current_step,
                    ElapsedTime=elapsed_time,
                    Last_Update_Time=utc_now(),
                )
                storage.flush()
                while next_progress <= current_step:
                    next_progress += int(simulation_config.progress_period)

        expected_frames = 1 + len(frame_schedule)
        if storage.frame_count != expected_frames:
            raise RuntimeError(
                f"Cavitation trajectory saved {storage.frame_count} frames; "
                f"expected {expected_frames}"
            )
        phase_frame_ids = [
            record["trajectory_frame_id"]
            for record in storage.frame_records
            if record["run_step"] in phase_steps
        ]
        if len(phase_frame_ids) != 5:
            raise RuntimeError("Cavitation did not save five phase-analysis frames")

        voxel = classify_voxel_histogram(
            state.positions,
            state.box,
            simulation_config.n_fcc_cells,
            density_threshold=simulation_config.phase_density_threshold,
            voxel_fraction_threshold=(
                simulation_config.phase_voxel_fraction_threshold
            ),
        )
        pe_drop = classify_pe_drop(
            np.asarray(storage.samples["potential_energy"]),
            state.n_particles,
            n_last=simulation_config.pe_drop_n_last,
            drop_threshold=simulation_config.pe_drop_threshold,
            z_limit=simulation_config.pe_drop_z_limit,
            decision_rule=simulation_config.pe_drop_decision_rule,
        )
        selected_phase = select_phase_classification(
            voxel,
            pe_drop,
            method=simulation_config.phase_method,
        )
        phase_fit = conditional_phase_fit(
            voxel,
            run_paths.trajectory,
            simulation_config.n_fcc_cells,
            interface_void_fraction=(
                simulation_config.phase_fit_interface_void_fraction
            ),
            interface_points=simulation_config.phase_fit_interface_points,
            max_iterations=simulation_config.phase_fit_max_iterations,
            frame_indices=phase_frame_ids,
        )
        summary = thermodynamic_summary(
            np.asarray(storage.samples["run_step"]),
            np.asarray(storage.samples["pressure"]),
            np.asarray(storage.samples["potential_energy"]),
            state.n_particles,
            n_last=simulation_config.summary_num_samples,
        )
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
                simulation_config.n_fcc_cells,
                storage.frame_count - 1,
                simulation.timestep,
                current_step,
                simulation_config.dt,
                trajectory_relative,
                prior_lj_time=prior_lj_time,
            ),
            "mdsims/analysis/phase_separation/voxel_histogram": voxel,
            "mdsims/analysis/phase_separation/PE_drop": pe_drop,
            "mdsims/analysis/phase_separation/selected": selected_phase,
            "mdsims/analysis/thermodynamics": summary,
            "mdsims/analysis/phase_fit": phase_fit,
            "mdsims/output": {
                "Trajectory_Frame_Log_Indices": [
                    record["log_index"] for record in storage.frame_records
                ],
                "Trajectory_Frame_Run_Steps": [
                    record["run_step"] for record in storage.frame_records
                ],
                "Trajectory_Frame_HOOMD_Timesteps": [
                    record["hoomd_timestep"] for record in storage.frame_records
                ],
                "Phase_Average_Trajectory_Frame_IDs": phase_frame_ids,
            },
        })
        storage.close()

        cavitation_row = {
            "File_Location": run_paths.relative_directory.as_posix(),
            "Source_Run_ID": str(config.source_run_id),
            "Source_Frame_ID": int(source_frame_id),
            "N_Cells": int(simulation_config.n_fcc_cells),
            "Therm_kT": float(simulation_config.kT),
            "Therm_Seed": int(simulation_config.seed),
            "Source_Density": float(source_state.density),
            "Initial_Density": float(initial_state.density),
            "BoxLength": float(np.min(initial_state.box[:3])),
            "Mask_Radius": float(config.mask_radius),
            "Random_Location": int(bool(config.random_location)),
            "Location_Seed": (
                int(config.location_seed)
                if config.location_seed is not None
                else None
            ),
            "dt": float(simulation_config.dt),
            "Nsteps": int(config.nsteps),
            "This_LJ_Time": float(config.nsteps) * float(simulation_config.dt),
            "Cumulative_LJ_Time": (
                prior_lj_time
                + float(config.nsteps) * float(simulation_config.dt)
            ),
            "Ensemble": "NVT",
            "T_Set": float(simulation_config.kT),
            "P_Set": None,
            "LJ_r_cut": float(simulation_config.r_cut_LJ),
            "LJ_r_on": (
                float(simulation_config.r_on_LJ)
                if simulation_config.lj_mode == "xplor"
                else None
            ),
            "LJ_Mode": simulation_config.lj_mode,
            "Phase_Separation_Status": selected_phase["status"],
            "Phase_Separation_Method": selected_phase["method"],
            "Phase_Separation_Method_Version": selected_phase["method_version"],
            **phase_fit_sql_values(phase_fit),
            **summary,
            "Num_Frames": storage.frame_count,
        }
        database.complete_cavitation(
            run_id,
            cavitation=cavitation_row,
            master={
                "Current_Nstep": current_step,
                "ElapsedTime": elapsed_time,
                "EndTime": end_time,
                "Last_Update_Time": end_time,
                "Status": "Complete",
                "Stop_Reason": None,
                "Status_Message": None,
            },
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
            "mask": mask,
            "phase_separation": selected_phase,
            "phase_fit": phase_fit,
            "thermodynamic_summary": summary,
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
