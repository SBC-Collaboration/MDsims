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


THERMALIZATION_METHOD_VERSION = "thermalization_v4_3"
CLONE_RESCALE_METHOD_VERSION = "clone_rescale_thermalization_v3"
CLONE_FINAL_FRAME_METHOD_VERSION = "clone_final_frame_v1"
LINEAR_DENSITY_METHOD_VERSION = "linear_density_v1"
CLONE_FINAL_DENSITY_RELATIVE_TOLERANCE = 1e-3
THERMALIZATION_TRAJECTORY_METHOD_VERSION = "initial_plus_terminal_5_stride_10_v1"
THERMALIZATION_PHASE_FRAME_COUNT = 5
THERMALIZATION_PHASE_LOG_STRIDE = 10


def clone_final_density_is_acceptable(actual: float, requested: float) -> bool:
    """Return whether a resized clone finished within 0.1% of its target."""

    return bool(
        np.isclose(
            float(actual),
            float(requested),
            rtol=CLONE_FINAL_DENSITY_RELATIVE_TOLERANCE,
            atol=0.0,
        )
    )


def thermalization_log_steps(nsteps: int, log_period: int) -> list[int]:
    """Return every evolved HDF5 logging step, including the final step."""

    nsteps = int(nsteps)
    log_period = int(log_period)
    if nsteps <= 0 or log_period <= 0:
        raise ValueError("nsteps and log_period must be positive")
    steps = list(range(log_period, nsteps + 1, log_period))
    if not steps or steps[-1] != nsteps:
        steps.append(nsteps)
    return steps


def thermalization_phase_frame_schedule(
    nsteps: int,
    log_period: int,
) -> list[dict[str, int]]:
    """Select five terminal logs separated by ten evolved log points."""

    log_steps = thermalization_log_steps(nsteps, log_period)
    final_ordinal = len(log_steps)
    first_ordinal = final_ordinal - (
        (THERMALIZATION_PHASE_FRAME_COUNT - 1)
        * THERMALIZATION_PHASE_LOG_STRIDE
    )
    if first_ordinal < 1:
        raise ValueError(
            "Thermalization requires at least 41 evolved log points to save "
            "five phase-analysis frames spaced by 10 logs. Reduce log_period "
            "or increase Nsteps."
        )
    ordinals = list(range(
        first_ordinal,
        final_ordinal + 1,
        THERMALIZATION_PHASE_LOG_STRIDE,
    ))
    return [
        {
            "log_ordinal": ordinal,
            "run_step": log_steps[ordinal - 1],
        }
        for ordinal in ordinals
    ]


@dataclass(frozen=True)
class ThermalizationConfig:
    """The V3 thermalization inputs, with output and analysis controls."""

    n_fcc_cells: int
    target_rho: float
    nsteps: int
    kT: float = 1.5
    log_period: int = 1_000
    seed: int = 1
    dt: float = 0.002
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
        thermalization_phase_frame_schedule(self.nsteps, self.log_period)

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


@dataclass(frozen=True)
class CloneRescaleThermalizationConfig:
    """Inputs that differ from a completed source thermalization."""

    source_run_id: str
    final_density: float
    nsteps: int
    notes: str | None = None

    def validate(self) -> None:
        if not str(self.source_run_id):
            raise ValueError("source_run_id is required")
        if float(self.final_density) <= 0:
            raise ValueError("final_density must be positive")
        if int(self.nsteps) <= 0:
            raise ValueError("nsteps must be positive")

    def signature_parameters(self, source_frame_id: int) -> dict[str, Any]:
        """Describe the clone operation without opening source files."""

        return {
            "sim_type": "Thermalization",
            "workflow_version": CLONE_RESCALE_METHOD_VERSION,
            "initialization_method": CLONE_FINAL_FRAME_METHOD_VERSION,
            "source_run_id": str(self.source_run_id),
            "source_frame_id": int(source_frame_id),
            "density_schedule": LINEAR_DENSITY_METHOD_VERSION,
            "final_density": float(self.final_density),
            "nsteps": int(self.nsteps),
            "simulation_settings": "inherit_source",
            "log_period": "inherit_source",
            "trajectory_storage": THERMALIZATION_TRAJECTORY_METHOD_VERSION,
        }

    def run_signature(self, source_frame_id: int) -> str:
        return create_run_signature(self.signature_parameters(source_frame_id))


def _make_simulation_from_frame(
    config: ThermalizationConfig,
    frame,
    *,
    thermalize_momenta: bool,
):
    import hoomd

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
    if thermalize_momenta:
        simulation.state.thermalize_particle_momenta(
            filter=hoomd.filter.All(),
            kT=float(config.kT),
        )
    return simulation, thermo, type(simulation.device).__name__


def _make_simulation(config: ThermalizationConfig, lattice):
    frame = make_gsd_frame(lattice, particle_type=config.particle_type)
    return _make_simulation_from_frame(
        config,
        frame,
        thermalize_momenta=True,
    )


def _add_linear_density_resize(
    simulation,
    final_density: float,
    n_particles: int,
    nsteps: int,
):
    """Scale the box so number density varies linearly with timestep."""

    import hoomd

    final_volume = int(n_particles) / float(final_density)
    box_variant = hoomd.variant.box.InverseVolumeRamp(
        initial_box=simulation.state.box,
        final_volume=final_volume,
        t_start=int(simulation.timestep),
        t_ramp=int(nsteps),
    )
    updater = hoomd.update.BoxResize(
        trigger=hoomd.trigger.Periodic(1),
        box=box_variant,
        filter=hoomd.filter.All(),
    )
    simulation.operations.updaters.append(updater)
    return updater


def _trajectory_policy_metadata(config: ThermalizationConfig) -> dict[str, Any]:
    schedule = thermalization_phase_frame_schedule(
        config.nsteps,
        config.log_period,
    )
    return {
        "Trajectory_Storage_Method": THERMALIZATION_TRAJECTORY_METHOD_VERSION,
        "Initial_Frame_Count": 1,
        "Phase_Average_Frame_Count": THERMALIZATION_PHASE_FRAME_COUNT,
        "Phase_Average_Log_Stride": THERMALIZATION_PHASE_LOG_STRIDE,
        "Phase_Average_Log_Ordinals": [
            item["log_ordinal"] for item in schedule
        ],
        "Phase_Average_Run_Steps": [item["run_step"] for item in schedule],
        "Phase_Average_Trajectory_Frame_IDs": list(range(
            1,
            THERMALIZATION_PHASE_FRAME_COUNT + 1,
        )),
    }


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
            **_trajectory_policy_metadata(config),
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


def _frame_state_data(frame) -> StateData:
    """Extract the state fields needed to validate and describe a GSD frame."""

    return StateData(
        positions=np.asarray(frame.particles.position, dtype=np.float64).copy(),
        velocities=np.asarray(frame.particles.velocity, dtype=np.float64).copy(),
        box=np.asarray(frame.configuration.box, dtype=np.float64).copy(),
        n_particles=int(frame.particles.N),
        particle_types=tuple(str(item) for item in frame.particles.types),
    )


def _metadata_value(
    metadata: dict[str, Any],
    path: str,
    fallback: Any,
) -> Any:
    value = metadata.get(path)
    return fallback if value is None else value


def _clone_base_metadata(
    run_id: str,
    signature: str,
    request: CloneRescaleThermalizationConfig,
    config: ThermalizationConfig,
    paths: RunPaths,
    device_name: str,
    source_state: StateData,
    source_frame_id: int,
    source_timestep: int,
    source_file_location: str,
    prior_lj_time: float,
) -> dict[str, dict[str, Any]]:
    relative = paths.relative_directory.as_posix()
    return {
        "mdsims/run": {
            "Run_ID": run_id,
            "Run_Signature": signature,
            "Sim_Type": "Thermalization",
            "Status": "Initializing",
            "Workflow_Version": CLONE_RESCALE_METHOD_VERSION,
            "Canonical_Config_JSON": canonical_json(
                request.signature_parameters(source_frame_id)
            ),
        },
        "mdsims/source": {
            "State_Role": "source",
            "Source_Type": "Thermalization_Run",
            "State_Creation_Method": CLONE_FINAL_FRAME_METHOD_VERSION,
            "Source_Run_ID": str(request.source_run_id),
            "Source_Frame_ID": int(source_frame_id),
            "Source_HOOMD_Timestep": int(source_timestep),
            "Source_File_Location": str(source_file_location),
        },
        "mdsims/protocol": {
            "N_Cells": int(config.n_fcc_cells),
            "Therm_kT": float(config.kT),
            "Therm_Seed": int(config.seed),
            "Density_Start": float(source_state.density),
            "Density_End_Target": float(request.final_density),
            "Density_End_Relative_Tolerance": (
                CLONE_FINAL_DENSITY_RELATIVE_TOLERANCE
            ),
            "Density_Schedule": "linear_density",
            "Density_Schedule_Version": LINEAR_DENSITY_METHOD_VERSION,
            "Nsteps": int(config.nsteps),
            "dt": float(config.dt),
            "Ensemble": "NVT",
            "T_Set": float(config.kT),
            "Particle_Type": config.particle_type,
            "Device": device_name,
            "Always_Compute_Pressure": True,
            "Inherited_Simulation_Settings": True,
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
            "Log_Period_Inherited_From_Run_ID": str(request.source_run_id),
            **_trajectory_policy_metadata(config),
        },
        "mdsims/time": {
            "Prior_Cumulative_LJ_Time": float(prior_lj_time),
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
    prior_lj_time: float = 0.0,
) -> dict[str, Any]:
    return {
        "State_Role": role,
        "Run_ID": run_id,
        "Frame_ID": int(frame_id),
        "HOOMD_Timestep": int(hoomd_timestep),
        "Run_Step": int(run_step),
        "This_LJ_Time": float(run_step) * float(dt),
        "Cumulative_LJ_Time": (
            float(prior_lj_time) + float(run_step) * float(dt)
        ),
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


def _combined_note(automatic: str, user_note: str | None) -> str:
    if user_note is None or not str(user_note).strip():
        return automatic
    return f"{automatic} User note: {str(user_note).strip()}"


def _clone_request_context(
    request: CloneRescaleThermalizationConfig,
    database: SQLiteRunDatabase,
) -> tuple[dict[str, Any], dict[str, Any], int, str]:
    """Resolve everything needed for duplicate checking using SQL only."""

    source_master = database.get_run(request.source_run_id)
    if source_master is None:
        raise KeyError(
            f"Source Run_ID was not found: {request.source_run_id}"
        )
    if source_master.get("Sim_Type") != "Thermalization":
        raise ValueError("Clone source must be a Thermalization run")
    if source_master.get("Status") != "Complete":
        raise ValueError("Clone source must have Status='Complete'")

    rows = database.query_thermalizations(
        Run_ID=str(request.source_run_id),
        limit=1,
    )
    if not rows:
        raise ValueError(
            "Clone source has no completed Thermalization table row"
        )
    source_thermalization = rows[0]
    source_frame_id = int(source_thermalization["Num_Frames"]) - 1
    if source_frame_id < 0:
        raise ValueError("Clone source contains no saved GSD frames")

    automatic_note = (
        f"Cloned final frame {source_frame_id} from Run_ID "
        f"{request.source_run_id}; box rescaled isotropically with density "
        f"changing linearly from "
        f"{float(source_thermalization['Density_End']):.6f} to "
        f"{float(request.final_density):.6f} over {int(request.nsteps)} steps. "
        "All other simulation settings inherited from the source run."
    )
    return (
        source_master,
        source_thermalization,
        source_frame_id,
        _combined_note(automatic_note, request.notes),
    )


def _inherited_clone_config(
    request: CloneRescaleThermalizationConfig,
    source_master: dict[str, Any],
    source_thermalization: dict[str, Any],
    source_metadata: dict[str, Any],
    source_state: StateData,
) -> ThermalizationConfig:
    """Recreate source dynamics and output settings from its saved record."""

    device = str(_metadata_value(
        source_metadata,
        "mdsims/protocol/Device",
        "auto",
    )).lower()
    if device not in {"cpu", "gpu"}:
        device = "auto"

    lj_mode = str(source_thermalization["LJ_Mode"])
    r_on = source_thermalization["LJ_r_on"]
    if r_on is None:
        r_on = 2.0

    config = ThermalizationConfig(
        n_fcc_cells=int(source_master["N_Cells"]),
        target_rho=float(request.final_density),
        nsteps=int(request.nsteps),
        kT=float(source_thermalization["Therm_kT"]),
        log_period=int(_metadata_value(
            source_metadata,
            "mdsims/output/Log_Period",
            1_000,
        )),
        seed=int(source_thermalization["Therm_Seed"]),
        dt=float(source_thermalization["dt"]),
        epsilon_LJ=float(_metadata_value(
            source_metadata,
            "mdsims/interaction/epsilon_LJ",
            1.0,
        )),
        sigma_LJ=float(_metadata_value(
            source_metadata,
            "mdsims/interaction/sigma_LJ",
            1.0,
        )),
        r_cut_LJ=float(source_thermalization["LJ_r_cut"]),
        buffer_LJ=float(_metadata_value(
            source_metadata,
            "mdsims/interaction/Neighbor_Buffer",
            0.4,
        )),
        lj_mode=lj_mode,
        r_on_LJ=float(r_on),
        particle_type=str(_metadata_value(
            source_metadata,
            "mdsims/protocol/Particle_Type",
            source_state.particle_types[0],
        )),
        device=device,
        progress_period=int(_metadata_value(
            source_metadata,
            "mdsims/output/Progress_Update_Period",
            25_000,
        )),
        phase_method=str(source_thermalization["Phase_Separation_Method"]),
        notes=request.notes,
    )
    config.validate()
    return config


def run_thermalization(
    config: ThermalizationConfig | CloneRescaleThermalizationConfig,
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

    clone_request = (
        config if isinstance(config, CloneRescaleThermalizationConfig) else None
    )
    source_master = None
    source_thermalization = None
    source_frame_id = None
    master_note = config.notes
    if clone_request is None:
        signature = config.run_signature
        n_cells = int(config.n_fcc_cells)
    else:
        (
            source_master,
            source_thermalization,
            source_frame_id,
            master_note,
        ) = _clone_request_context(clone_request, database)
        signature = clone_request.run_signature(source_frame_id)
        n_cells = int(source_master["N_Cells"])

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
        N_Cells=n_cells,
        Nsteps=int(config.nsteps),
        Current_Nstep=0,
        ElapsedTime=0.0,
        Last_Update_Time=utc_now(),
        Sim_Type="Thermalization",
        Status="Initializing",
        Notes=master_note,
    )

    current_step = 0
    elapsed_time = 0.0
    storage: RunStorage | None = None
    try:
        prior_lj_time = 0.0
        if clone_request is None:
            simulation_config = config
            lattice = build_fcc_lattice(
                simulation_config.n_fcc_cells,
                simulation_config.target_rho,
            )
            simulation, thermo, device_name = _make_simulation(
                simulation_config,
                lattice,
            )
            density_start = float(lattice.actual_density)
            box_length_start = float(lattice.box_length)
            metadata = _base_metadata(
                run_id,
                signature,
                simulation_config,
                run_paths,
                lattice,
                device_name,
            )
            source_state = None
            source_timestep = None
            source_file_location = None
        else:
            from .run_analysis import open_run

            source_run = open_run(
                clone_request.source_run_id,
                project_paths=project_paths,
                database=database,
            )
            if source_run.frame_count != int(source_thermalization["Num_Frames"]):
                raise RuntimeError(
                    "Source GSD frame count does not match the SQL record"
                )
            source_frame = source_run.load_frame(source_frame_id)
            source_state = _frame_state_data(source_frame)
            source_metadata = source_run.metadata()
            simulation_config = _inherited_clone_config(
                clone_request,
                source_master,
                source_thermalization,
                source_metadata,
                source_state,
            )
            expected_particles = 4 * int(simulation_config.n_fcc_cells) ** 3
            if source_state.n_particles != expected_particles:
                raise RuntimeError(
                    "Source particle count does not match its recorded N_Cells"
                )
            simulation, thermo, device_name = _make_simulation_from_frame(
                simulation_config,
                source_frame,
                thermalize_momenta=False,
            )
            _add_linear_density_resize(
                simulation,
                clone_request.final_density,
                source_state.n_particles,
                clone_request.nsteps,
            )
            density_start = float(source_state.density)
            box_length_start = float(source_state.box[0])
            prior_lj_time = float(
                source_thermalization["Cumulative_LJ_Time"]
            )
            source_timestep = int(source_frame.configuration.step)
            source_file_location = str(source_thermalization["File_Location"])
            metadata = _clone_base_metadata(
                run_id,
                signature,
                clone_request,
                simulation_config,
                run_paths,
                device_name,
                source_state,
                source_frame_id,
                source_timestep,
                source_file_location,
                prior_lj_time,
            )

        phase_frame_schedule = thermalization_phase_frame_schedule(
            simulation_config.nsteps,
            simulation_config.log_period,
        )
        phase_frame_steps = {
            item["run_step"] for item in phase_frame_schedule
        }
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
        initial_metadata = {
            "mdsims/run": {"Status": "Running", "StartTime": start_time},
            "mdsims/states/initial": _state_metadata(
                "initial",
                run_id,
                state,
                simulation_config.n_fcc_cells,
                0,
                simulation.timestep,
                0,
                simulation_config.dt,
                trajectory_relative,
                prior_lj_time=prior_lj_time,
            ),
        }
        if clone_request is not None:
            source_this_lj_time = float(
                source_thermalization["This_LJ_Time"]
            )
            source_trajectory = (
                f"{source_file_location}/trajectory.gsd"
            )
            initial_metadata["mdsims/states/source"] = _state_metadata(
                "source",
                clone_request.source_run_id,
                source_state,
                simulation_config.n_fcc_cells,
                source_frame_id,
                source_timestep,
                int(source_thermalization["Nsteps"]),
                float(source_thermalization["dt"]),
                source_trajectory,
                prior_lj_time=prior_lj_time - source_this_lj_time,
            )
        storage.write_metadata(initial_metadata)

        next_log = min(
            int(simulation_config.log_period),
            int(simulation_config.nsteps),
        )
        next_progress = min(
            int(simulation_config.progress_period),
            int(simulation_config.nsteps),
        )
        while current_step < int(simulation_config.nsteps):
            target = min(
                next_log,
                next_progress,
                int(simulation_config.nsteps),
            )
            started = time.perf_counter()
            simulation.run(target - current_step)
            elapsed_time += time.perf_counter() - started
            current_step = target

            log_now = (
                current_step == next_log
                or current_step == simulation_config.nsteps
            )
            progress_now = (
                current_step == next_progress
                or current_step == simulation_config.nsteps
            )
            if log_now:
                state = storage.record(
                    simulation,
                    thermo,
                    current_step,
                    simulation_config.dt,
                    prior_lj_time=prior_lj_time,
                    save_frame=current_step in phase_frame_steps,
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

        if storage.frame_count != 1 + THERMALIZATION_PHASE_FRAME_COUNT:
            raise RuntimeError(
                "Thermalization trajectory did not save exactly one initial "
                "frame and five phase-analysis frames"
            )
        phase_frame_ids = [
            record["trajectory_frame_id"]
            for record in storage.frame_records[1:]
        ]

        if clone_request is not None and not clone_final_density_is_acceptable(
            state.density,
            clone_request.final_density,
        ):
            raise RuntimeError(
                "Linear density resize did not reach the requested final density: "
                f"requested {clone_request.final_density}, got {state.density}; "
                "allowed relative difference is 0.1%"
            )

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
        selected_phase = select_phase_classification(
            voxel,
            pe_drop,
            method=simulation_config.phase_method,
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
                    record["hoomd_timestep"]
                    for record in storage.frame_records
                ],
                "Phase_Average_Trajectory_Frame_IDs": phase_frame_ids,
            },
        })
        storage.close()

        thermalization_row = {
            "File_Location": run_paths.relative_directory.as_posix(),
            "Clone_Run_ID": (
                str(clone_request.source_run_id)
                if clone_request is not None
                else None
            ),
            "Clone_Frame_ID": source_frame_id,
            "Therm_kT": float(simulation_config.kT),
            "Therm_Seed": int(simulation_config.seed),
            "Density_Start": density_start,
            "Density_End": float(state.density),
            "BoxLength_Start": box_length_start,
            "BoxLength_End": float(state.box[0]),
            "dt": float(simulation_config.dt),
            "Nsteps": int(simulation_config.nsteps),
            "This_LJ_Time": (
                float(simulation_config.nsteps) * float(simulation_config.dt)
            ),
            "Cumulative_LJ_Time": (
                prior_lj_time
                + float(simulation_config.nsteps) * float(simulation_config.dt)
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


def run_clone_rescale_thermalization(
    source_run_id: str,
    final_density: float,
    nsteps: int,
    *,
    notes: str | None = None,
    project_paths: ProjectPaths | None = None,
    database: SQLiteRunDatabase | None = None,
) -> dict[str, Any]:
    """Clone a completed thermalization and linearly change only its density.

    The final GSD frame supplies the complete particle state. Temperature,
    seed, timestep, interactions, integration timestep, device preference, and
    output periods are inherited from the source run. Positions are scaled by
    HOOMD while velocities are preserved and are not rethermalized.
    """

    request = CloneRescaleThermalizationConfig(
        source_run_id=str(source_run_id),
        final_density=float(final_density),
        nsteps=int(nsteps),
        notes=notes,
    )
    return run_thermalization(
        request,
        project_paths=project_paths,
        database=database,
    )
