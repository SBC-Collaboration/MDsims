"""Create, evolve, classify, and register one thermalized FCC state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    classify_pe_drop,
    classify_voxel_histogram,
    select_phase_classification,
    thermodynamic_summary,
)
from .database import SQLiteRunDatabase, utc_now
from .lattices import FCC_METHOD_VERSION, build_fcc_lattice, make_gsd_frame
from .paths import ProjectPaths, RunPaths
from .signatures import canonical_json, create_run_signature
from .storage import RunStorage, StateData, update_hdf5_metadata
from .voxel_fit import conditional_phase_fit, phase_fit_sql_values


THERMALIZATION_METHOD_VERSION = "thermalization_v4_2"


@dataclass(frozen=True)
class ThermalizationConfig:
    """The V3 thermalization inputs, with output and analysis controls."""

    n_fcc_cells: int
    target_rho: float
    nsteps: int
    kT: float = 1.5
    log_period: int = 1_000
    seed: int = 1
    dt: float = 0.005
    epsilon_LJ: float = 1.0
    sigma_LJ: float = 1.0
    r_cut_LJ: float = 2.5
    buffer_LJ: float = 0.4
    lj_mode: str = "xplor"
    r_on_LJ: float = 2.0
    particle_type: str = "A"
    device: str = "auto"
    progress_period: int = 25_000
    phase_method: str = "voxel_histogram"
    phase_density_threshold: float = 0.2
    phase_voxel_fraction_threshold: float = 0.01
    pe_drop_threshold: float = -0.15
    pe_drop_z_limit: float = 5.0
    pe_drop_n_last: int = 100
    pe_drop_decision_rule: str = "either"
    summary_num_samples: int = 100
    phase_fit_interface_void_fraction: float = 0.5
    phase_fit_interface_points: int = 40
    phase_fit_max_iterations: int = 500
    notes: str | None = None

    def validate(self) -> None:
        positive_ints = {
            "n_fcc_cells": self.n_fcc_cells,
            "nsteps": self.nsteps,
            "log_period": self.log_period,
            "progress_period": self.progress_period,
            "pe_drop_n_last": self.pe_drop_n_last,
            "summary_num_samples": self.summary_num_samples,
            "phase_fit_interface_points": self.phase_fit_interface_points,
            "phase_fit_max_iterations": self.phase_fit_max_iterations,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "target_rho": self.target_rho,
            "kT": self.kT,
            "dt": self.dt,
            "epsilon_LJ": self.epsilon_LJ,
            "sigma_LJ": self.sigma_LJ,
            "r_cut_LJ": self.r_cut_LJ,
        }.items():
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.buffer_LJ) < 0:
            raise ValueError("buffer_LJ cannot be negative")
        if self.lj_mode not in {"none", "shift", "xplor"}:
            raise ValueError("lj_mode must be none, shift, or xplor")
        if self.lj_mode == "xplor" and not 0 < self.r_on_LJ < self.r_cut_LJ:
            raise ValueError("xplor requires 0 < r_on_LJ < r_cut_LJ")
        if self.device.lower() not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be auto, cpu, or gpu")
        if int(self.seed) < 0:
            raise ValueError("seed cannot be negative")
        if not 0 <= float(self.phase_fit_interface_void_fraction) <= 1:
            raise ValueError(
                "phase_fit_interface_void_fraction must be between 0 and 1"
            )

    def signature_parameters(self) -> dict[str, Any]:
        """Parameters that define dynamics or canonical saved output."""

        return {
            "sim_type": "Thermalization",
            "workflow_version": THERMALIZATION_METHOD_VERSION,
            "state_creation_version": FCC_METHOD_VERSION,
            "n_fcc_cells": int(self.n_fcc_cells),
            "target_rho": float(self.target_rho),
            "nsteps": int(self.nsteps),
            "therm_kT": float(self.kT),
            "therm_seed": int(self.seed),
            "dt": float(self.dt),
            "ensemble": "NVT",
            "T_set": float(self.kT),
            "epsilon_LJ": float(self.epsilon_LJ),
            "sigma_LJ": float(self.sigma_LJ),
            "LJ_r_cut": float(self.r_cut_LJ),
            "LJ_r_on": float(self.r_on_LJ),
            "LJ_mode": str(self.lj_mode),
            "neighbor_buffer": float(self.buffer_LJ),
            "particle_type": str(self.particle_type),
            "log_period": int(self.log_period),
        }

    @property
    def run_signature(self) -> str:
        return create_run_signature(self.signature_parameters())


def _make_simulation(config: ThermalizationConfig, lattice):
    import hoomd

    frame = make_gsd_frame(lattice, particle_type=config.particle_type)

    def create(device):
        simulation = hoomd.Simulation(device=device, seed=int(config.seed))
        simulation.create_state_from_snapshot(frame)
        return simulation

    preference = config.device.lower()
    if preference == "cpu":
        simulation = create(hoomd.device.CPU())
    elif preference == "gpu":
        simulation = create(hoomd.device.GPU())
    else:
        try:
            simulation = create(hoomd.device.GPU())
        except Exception:
            simulation = create(hoomd.device.CPU())

    integrator = hoomd.md.Integrator(dt=float(config.dt))
    neighbor_list = hoomd.md.nlist.Cell(buffer=float(config.buffer_LJ))
    lj = hoomd.md.pair.LJ(nlist=neighbor_list, mode=config.lj_mode)
    pair = (config.particle_type, config.particle_type)
    lj.params[pair] = {
        "epsilon": float(config.epsilon_LJ),
        "sigma": float(config.sigma_LJ),
    }
    lj.r_cut[pair] = float(config.r_cut_LJ)
    if config.lj_mode == "xplor":
        lj.r_on[pair] = float(config.r_on_LJ)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(
            filter=hoomd.filter.All(),
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=float(config.kT)),
        )
    )
    simulation.operations.integrator = integrator

    thermo = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    simulation.operations.computes.append(thermo)
    # This workflow queries ThermodynamicQuantities manually after run calls.
    # HOOMD otherwise does not guarantee that pair virials and pressure are
    # available for an NVT simulation.
    simulation.always_compute_pressure = True
    simulation.state.thermalize_particle_momenta(
        filter=hoomd.filter.All(),
        kT=float(config.kT),
    )
    return simulation, thermo, type(simulation.device).__name__


def _base_metadata(
    run_id: str,
    signature: str,
    config: ThermalizationConfig,
    paths: RunPaths,
    lattice,
    device_name: str,
) -> dict[str, dict[str, Any]]:
    relative = paths.relative_directory.as_posix()
    return {
        "mdsims/run": {
            "Run_ID": run_id,
            "Run_Signature": signature,
            "Sim_Type": "Thermalization",
            "Status": "Initializing",
            "Workflow_Version": THERMALIZATION_METHOD_VERSION,
            "Canonical_Config_JSON": canonical_json(
                config.signature_parameters()
            ),
        },
        "mdsims/source": {
            "State_Role": "source",
            "Source_Type": "FCC_Lattice",
            "State_Creation_Method": FCC_METHOD_VERSION,
        },
        "mdsims/protocol": {
            "N_Cells": int(config.n_fcc_cells),
            "Therm_kT": float(config.kT),
            "Therm_Seed": int(config.seed),
            "Density_Target": float(config.target_rho),
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
        },
        "mdsims/states/source": {
            "State_Role": "source",
            "N_Cells": int(config.n_fcc_cells),
            "N_Particles": int(lattice.n_particles),
            "Box": [
                lattice.box_length,
                lattice.box_length,
                lattice.box_length,
                0.0,
                0.0,
                0.0,
            ],
            "Volume": float(lattice.volume),
            "Number_Density": float(lattice.actual_density),
        },
    }


def _state_metadata(
    role: str,
    run_id: str,
    state: StateData,
    n_cells: int,
    frame_id: int,
    hoomd_timestep: int,
    run_step: int,
    dt: float,
    trajectory_path: str,
) -> dict[str, Any]:
    return {
        "State_Role": role,
        "Run_ID": run_id,
        "Frame_ID": int(frame_id),
        "HOOMD_Timestep": int(hoomd_timestep),
        "Run_Step": int(run_step),
        "This_LJ_Time": float(run_step) * float(dt),
        "Cumulative_LJ_Time": float(run_step) * float(dt),
        "N_Cells": int(n_cells),
        "N_Particles": int(state.n_particles),
        "Particle_Types": state.particle_types,
        "Box": state.box,
        "Lx": float(state.box[0]),
        "Ly": float(state.box[1]),
        "Lz": float(state.box[2]),
        "Volume": state.volume,
        "Number_Density": state.density,
        "Trajectory_Path": trajectory_path,
    }


def _failure_update(
    database: SQLiteRunDatabase,
    run_id: str,
    current_step: int,
    elapsed_time: float,
    error: BaseException,
    status: str,
) -> None:
    database.update_master(
        run_id,
        Current_Nstep=int(current_step),
        ElapsedTime=float(elapsed_time),
        EndTime=utc_now(),
        Last_Update_Time=utc_now(),
        Status=status,
        Stop_Reason="user_cancel" if status == "Cancelled" else "exception",
        Status_Message=f"{type(error).__name__}: {error}",
    )


def run_thermalization(
    config: ThermalizationConfig,
    project_paths: ProjectPaths | None = None,
    database: SQLiteRunDatabase | None = None,
) -> dict[str, Any]:
    """Run a thermalization or report an existing matching SQL record.

    Duplicate checks are SQL-only: an existing run is reported and no GSD or
    HDF5 file is opened or loaded.
    """

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

    # The first write intentionally creates a row containing only Run_ID.
    run_id = database.reserve_run_id(max_attempts=3)
    run_paths = project_paths.for_run("Thermalization", run_id)
    database.update_master(
        run_id,
        Run_Signature=signature,
        N_Cells=int(config.n_fcc_cells),
        Nsteps=int(config.nsteps),
        Current_Nstep=0,
        ElapsedTime=0.0,
        Last_Update_Time=utc_now(),
        Sim_Type="Thermalization",
        Status="Initializing",
        Notes=config.notes,
    )

    current_step = 0
    elapsed_time = 0.0
    storage: RunStorage | None = None
    try:
        lattice = build_fcc_lattice(config.n_fcc_cells, config.target_rho)
        simulation, thermo, device_name = _make_simulation(config, lattice)
        storage = RunStorage(run_paths)
        storage.open(
            _base_metadata(
                run_id,
                signature,
                config,
                run_paths,
                lattice,
                device_name,
            )
        )

        start_time = utc_now()
        database.update_master(
            run_id,
            StartTime=start_time,
            Last_Update_Time=start_time,
            Status="Running",
        )

        simulation.run(0)
        state = storage.record(simulation, thermo, 0, config.dt)
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
        })

        next_log = min(int(config.log_period), int(config.nsteps))
        next_progress = min(int(config.progress_period), int(config.nsteps))
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
                    config.dt,
                )
                while next_log <= current_step:
                    next_log += int(config.log_period)
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

        voxel = classify_voxel_histogram(
            state.positions,
            state.box,
            config.n_fcc_cells,
            density_threshold=config.phase_density_threshold,
            voxel_fraction_threshold=config.phase_voxel_fraction_threshold,
        )
        pe_drop = classify_pe_drop(
            np.asarray(storage.samples["potential_energy"]),
            state.n_particles,
            n_last=config.pe_drop_n_last,
            drop_threshold=config.pe_drop_threshold,
            z_limit=config.pe_drop_z_limit,
            decision_rule=config.pe_drop_decision_rule,
        )
        phase_fit = conditional_phase_fit(
            voxel,
            run_paths.trajectory,
            config.n_fcc_cells,
            interface_void_fraction=config.phase_fit_interface_void_fraction,
            interface_points=config.phase_fit_interface_points,
            max_iterations=config.phase_fit_max_iterations,
        )
        selected_phase = select_phase_classification(
            voxel,
            pe_drop,
            method=config.phase_method,
        )
        summary = thermodynamic_summary(
            np.asarray(storage.samples["run_step"]),
            np.asarray(storage.samples["pressure"]),
            np.asarray(storage.samples["potential_energy"]),
            state.n_particles,
            n_last=config.summary_num_samples,
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
                config.n_fcc_cells,
                storage.frame_count - 1,
                simulation.timestep,
                current_step,
                config.dt,
                trajectory_relative,
            ),
            "mdsims/analysis/phase_separation/voxel_histogram": voxel,
            "mdsims/analysis/phase_separation/PE_drop": pe_drop,
            "mdsims/analysis/phase_separation/selected": selected_phase,
            "mdsims/analysis/thermodynamics": summary,
            "mdsims/analysis/phase_fit": phase_fit,
        })
        storage.close()

        thermalization_row = {
            "File_Location": run_paths.relative_directory.as_posix(),
            "Clone_Run_ID": None,
            "Clone_Frame_ID": None,
            "Therm_kT": float(config.kT),
            "Therm_Seed": int(config.seed),
            "Density_Start": float(lattice.actual_density),
            "Density_End": float(state.density),
            "BoxLength_Start": float(lattice.box_length),
            "BoxLength_End": float(state.box[0]),
            "dt": float(config.dt),
            "Nsteps": int(config.nsteps),
            "This_LJ_Time": float(config.nsteps) * float(config.dt),
            "Cumulative_LJ_Time": float(config.nsteps) * float(config.dt),
            "Ensemble": "NVT",
            "T_Set": float(config.kT),
            "P_Set": None,
            "LJ_r_cut": float(config.r_cut_LJ),
            "LJ_r_on": (
                float(config.r_on_LJ) if config.lj_mode == "xplor" else None
            ),
            "LJ_Mode": config.lj_mode,
            "Phase_Separation_Status": selected_phase["status"],
            "Phase_Separation_Method": selected_phase["method"],
            "Phase_Separation_Method_Version": selected_phase["method_version"],
            **phase_fit_sql_values(phase_fit),
            **summary,
            "Num_Frames": storage.frame_count,
        }
        database.complete_thermalization(
            run_id,
            thermalization=thermalization_row,
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
            database,
            run_id,
            current_step,
            elapsed_time,
            error,
            "Cancelled",
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
                database,
                run_id,
                current_step,
                elapsed_time,
                error,
                "Failed",
            )
        except Exception as database_error:
            error.add_note(f"The failure could not be written to SQL: {database_error}")
        raise
