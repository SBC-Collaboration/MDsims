"""Expand a completed thermalized liquid by cloning whole periodic boxes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .database import EXPANDED_CLONE_SIM_TYPE, SQLiteRunDatabase, utc_now
from .expanded_fcc import (
    COM_RECENTER_METHOD_VERSION,
    expanded_fcc_frame_schedule,
    _recenter_simulation_com,
)
from .paths import ProjectPaths, RunPaths
from .signatures import canonical_json, create_run_signature
from .storage import RunStorage, StateData, update_hdf5_metadata
from .thermalization import (
    ThermalizationConfig,
    _failure_update,
    _frame_state_data,
    _make_simulation_from_frame,
    _metadata_value,
    _state_metadata,
)


EXPANDED_CLONE_METHOD_VERSION = "thermalized_box_clone_expansion_v1"
EXPANDED_CLONE_STATE_VERSION = "whole_box_tiles_independent_vapor_thinning_v1"
EXPANDED_CLONE_TRAJECTORY_VERSION = "post_preparation_recenter_phase_final_v1"


@dataclass(frozen=True)
class ExpandedCloneConfig:
    """Inputs for cloning a completed thermalization into a slab geometry."""

    source_run_id: str
    liquid_scale: int
    vapor_scale: int
    vapor_density_divisor: float
    kT: float
    nsteps: int
    preparation_steps: int = 50_000
    seed: int | None = None
    com_recenter_period: int = 10_000
    notes: str | None = None

    def validate(self) -> None:
        if not str(self.source_run_id).strip():
            raise ValueError("source_run_id is required")
        for name, value in {
            "liquid_scale": self.liquid_scale,
            "vapor_scale": self.vapor_scale,
        }.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if float(self.vapor_density_divisor) < 1:
            raise ValueError("vapor_density_divisor must be at least 1")
        if float(self.kT) <= 0:
            raise ValueError("kT must be positive")
        if int(self.nsteps) <= 0:
            raise ValueError("nsteps must be positive")
        if int(self.preparation_steps) < 0:
            raise ValueError("preparation_steps cannot be negative")
        if int(self.com_recenter_period) <= 0:
            raise ValueError("com_recenter_period must be positive")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed cannot be negative")

    def signature_parameters(
        self,
        source_frame_id: int,
        effective_seed: int,
    ) -> dict[str, Any]:
        return {
            "sim_type": EXPANDED_CLONE_SIM_TYPE,
            "workflow_version": EXPANDED_CLONE_METHOD_VERSION,
            "state_creation_version": EXPANDED_CLONE_STATE_VERSION,
            "source_run_id": str(self.source_run_id),
            "source_frame_id": int(source_frame_id),
            "liquid_scale": int(self.liquid_scale),
            "vapor_scale_per_side": int(self.vapor_scale),
            "vapor_density_divisor": float(self.vapor_density_divisor),
            "kT": float(self.kT),
            "production_nsteps": int(self.nsteps),
            "preparation_steps": int(self.preparation_steps),
            "seed": int(effective_seed),
            "simulation_settings": "inherit_source_except_temperature_steps_seed",
            "vapor_selection": "independent_fixed_count_without_replacement",
            "velocity_initialization": "regenerated_maxwell_boltzmann",
            "com_recenter_period": int(self.com_recenter_period),
            "com_recenter_method": COM_RECENTER_METHOD_VERSION,
            "trajectory_storage": EXPANDED_CLONE_TRAJECTORY_VERSION,
        }

    def run_signature(self, source_frame_id: int, effective_seed: int) -> str:
        return create_run_signature(
            self.signature_parameters(source_frame_id, effective_seed)
        )


@dataclass(frozen=True)
class ExpandedCloneState:
    """Particle arrays and exact construction statistics for a cloned slab."""

    positions: np.ndarray
    type_ids: np.ndarray
    masses: np.ndarray
    particle_types: tuple[str, ...]
    box: np.ndarray
    source_particles: int
    particles_per_vapor_tile: int
    liquid_particles: int
    vapor_particles_per_side: int
    n_particles: int
    source_density: float
    vapor_density: float
    liquid_scale: int
    vapor_scale: int
    vapor_density_divisor: float

    @property
    def volume(self) -> float:
        return float(np.prod(self.box[:3]))

    @property
    def density(self) -> float:
        return self.n_particles / self.volume

    @property
    def center_length(self) -> float:
        source_length = self.box[0] / (
            self.liquid_scale + 2 * self.vapor_scale
        )
        return self.liquid_scale * source_length


def build_expanded_clone_state(
    source_frame,
    liquid_scale: int,
    vapor_scale: int,
    vapor_density_divisor: float,
    seed: int,
) -> ExpandedCloneState:
    """Tile a whole source box and independently thin every vapor tile."""

    liquid_scale = int(liquid_scale)
    vapor_scale = int(vapor_scale)
    vapor_density_divisor = float(vapor_density_divisor)
    seed = int(seed)
    if liquid_scale <= 0 or vapor_scale <= 0:
        raise ValueError("liquid_scale and vapor_scale must be positive")
    if vapor_density_divisor < 1:
        raise ValueError("vapor_density_divisor must be at least 1")
    if seed < 0:
        raise ValueError("seed cannot be negative")

    source_box = np.asarray(source_frame.configuration.box, dtype=np.float64)
    if source_box.shape != (6,) or np.any(source_box[:3] <= 0):
        raise ValueError("source frame has an invalid HOOMD box")
    if not np.allclose(source_box[3:], 0.0):
        raise ValueError(
            "Expanded clone currently requires an orthorhombic source box"
        )

    source_positions = np.asarray(
        source_frame.particles.position,
        dtype=np.float64,
    ).copy()
    source_particles = int(source_frame.particles.N)
    if source_positions.shape != (source_particles, 3):
        raise ValueError("source particle positions do not match particles.N")
    particle_types = tuple(str(item) for item in source_frame.particles.types)
    if len(particle_types) != 1:
        raise ValueError(
            "Expanded clone currently supports one particle type because the "
            "inherited LJ workflow defines one pair interaction"
        )
    type_ids = np.asarray(source_frame.particles.typeid, dtype=np.uint32).copy()
    masses = np.asarray(source_frame.particles.mass, dtype=np.float64).copy()
    if type_ids.shape != (source_particles,):
        raise ValueError("source type IDs do not match particles.N")
    if masses.shape != (source_particles,) or np.any(masses <= 0):
        raise ValueError("source masses must be a positive length-N array")

    lx, ly, lz = source_box[:3]
    # Rewrap the source before translating it so every tile occupies exactly
    # one complete source interval, regardless of saved periodic images.
    source_positions = (
        (source_positions + source_box[:3] / 2.0) % source_box[:3]
        - source_box[:3] / 2.0
    )
    particles_per_vapor_tile = int(round(
        source_particles / vapor_density_divisor
    ))
    if particles_per_vapor_tile < 1:
        raise ValueError(
            "vapor_density_divisor leaves fewer than one particle per vapor "
            "tile"
        )

    total_tiles = liquid_scale + 2 * vapor_scale
    new_lx = total_tiles * lx
    rng = np.random.default_rng(seed)
    position_tiles = []
    type_id_tiles = []
    mass_tiles = []
    for tile in range(total_tiles):
        is_vapor = tile < vapor_scale or tile >= vapor_scale + liquid_scale
        if is_vapor:
            selected = np.sort(rng.choice(
                source_particles,
                size=particles_per_vapor_tile,
                replace=False,
            ))
        else:
            selected = np.arange(source_particles)
        translated = source_positions[selected].copy()
        translated[:, 0] += (tile + 0.5) * lx - new_lx / 2.0
        position_tiles.append(translated)
        type_id_tiles.append(type_ids[selected])
        mass_tiles.append(masses[selected])

    positions = np.concatenate(position_tiles, axis=0)
    expanded_type_ids = np.concatenate(type_id_tiles)
    expanded_masses = np.concatenate(mass_tiles)
    liquid_particles = liquid_scale * source_particles
    vapor_particles_per_side = vapor_scale * particles_per_vapor_tile
    n_particles = liquid_particles + 2 * vapor_particles_per_side
    source_volume = float(lx * ly * lz)
    return ExpandedCloneState(
        positions=positions,
        type_ids=expanded_type_ids,
        masses=expanded_masses,
        particle_types=particle_types,
        box=np.array([new_lx, ly, lz, 0.0, 0.0, 0.0]),
        source_particles=source_particles,
        particles_per_vapor_tile=particles_per_vapor_tile,
        liquid_particles=liquid_particles,
        vapor_particles_per_side=vapor_particles_per_side,
        n_particles=n_particles,
        source_density=source_particles / source_volume,
        vapor_density=particles_per_vapor_tile / source_volume,
        liquid_scale=liquid_scale,
        vapor_scale=vapor_scale,
        vapor_density_divisor=vapor_density_divisor,
    )


def make_expanded_clone_frame(state: ExpandedCloneState):
    """Convert the constructed arrays to a fresh GSD/HOOMD frame."""

    import gsd.hoomd

    frame = gsd.hoomd.Frame()
    frame.configuration.step = 0
    frame.configuration.box = state.box
    frame.particles.N = state.n_particles
    frame.particles.types = list(state.particle_types)
    frame.particles.position = state.positions
    frame.particles.velocity = np.zeros((state.n_particles, 3))
    frame.particles.typeid = state.type_ids
    frame.particles.image = np.zeros((state.n_particles, 3), dtype=np.int32)
    frame.particles.mass = state.masses
    return frame


def _remove_net_momentum(simulation) -> np.ndarray:
    """Remove the mass-weighted center-of-mass velocity exactly."""

    snapshot = simulation.state.get_snapshot()
    velocity = np.zeros(3, dtype=np.float64)
    if snapshot.communicator.rank == 0:
        masses = np.asarray(snapshot.particles.mass, dtype=np.float64)
        velocities = np.asarray(snapshot.particles.velocity, dtype=np.float64)
        velocity = np.average(velocities, axis=0, weights=masses)
        snapshot.particles.velocity[:] = velocities - velocity
    simulation.state.set_snapshot(snapshot)
    return velocity


def _source_context(
    request: ExpandedCloneConfig,
    database: SQLiteRunDatabase,
) -> tuple[dict[str, Any], dict[str, Any], int, int, str]:
    source_master = database.get_run(request.source_run_id)
    if source_master is None:
        raise KeyError(f"Source Run_ID was not found: {request.source_run_id}")
    if source_master.get("Sim_Type") != "Thermalization":
        raise ValueError("Expanded clone source must be a Thermalization run")
    if source_master.get("Status") != "Complete":
        raise ValueError("Expanded clone source must have Status='Complete'")
    source_thermalization = database.get_thermalization(request.source_run_id)
    if source_thermalization is None:
        raise ValueError("Source has no completed Thermalization table row")
    phase_status = source_thermalization.get("Phase_Separation_Status")
    if phase_status != "Not_Separated":
        raise ValueError(
            "Expanded clone source must be a homogeneous thermalization "
            "with Phase_Separation_Status='Not_Separated'"
        )
    source_frame_id = int(source_thermalization["Num_Frames"]) - 1
    if source_frame_id < 0:
        raise ValueError("Source thermalization has no saved final frame")
    effective_seed = (
        int(request.seed)
        if request.seed is not None
        else int(source_thermalization["Therm_Seed"])
    )
    automatic_note = (
        f"Expanded final frame {source_frame_id} from Thermalization Run_ID "
        f"{request.source_run_id}: {int(request.liquid_scale)} liquid tile(s), "
        f"{int(request.vapor_scale)} vapor tile(s) per side, vapor density "
        f"divided by {float(request.vapor_density_divisor):g}; regenerated "
        f"momenta at kT={float(request.kT):g}; prepared for "
        f"{int(request.preparation_steps)} steps then evolved for "
        f"{int(request.nsteps)} production steps."
    )
    if request.notes and str(request.notes).strip():
        automatic_note += f" User note: {str(request.notes).strip()}"
    return (
        source_master,
        source_thermalization,
        source_frame_id,
        effective_seed,
        automatic_note,
    )


def _inherited_config(
    request: ExpandedCloneConfig,
    source_master: dict[str, Any],
    source_thermalization: dict[str, Any],
    source_metadata: dict[str, Any],
    source_state: StateData,
    effective_seed: int,
) -> ThermalizationConfig:
    device = str(_metadata_value(
        source_metadata, "mdsims/protocol/Device", "auto"
    )).lower()
    if device not in {"cpu", "gpu"}:
        device = "auto"
    r_on = source_thermalization["LJ_r_on"]
    if r_on is None:
        r_on = 2.0
    config = ThermalizationConfig(
        n_fcc_cells=int(source_master["N_Cells"]),
        target_rho=float(source_state.density),
        nsteps=int(request.nsteps),
        kT=float(request.kT),
        log_period=int(_metadata_value(
            source_metadata, "mdsims/output/Log_Period", 1_000
        )),
        seed=int(effective_seed),
        dt=float(source_thermalization["dt"]),
        epsilon_LJ=float(_metadata_value(
            source_metadata, "mdsims/interaction/epsilon_LJ", 1.0
        )),
        sigma_LJ=float(_metadata_value(
            source_metadata, "mdsims/interaction/sigma_LJ", 1.0
        )),
        r_cut_LJ=float(source_thermalization["LJ_r_cut"]),
        buffer_LJ=float(_metadata_value(
            source_metadata, "mdsims/interaction/Neighbor_Buffer", 0.4
        )),
        lj_mode=str(source_thermalization["LJ_Mode"]),
        r_on_LJ=float(r_on),
        particle_type=str(_metadata_value(
            source_metadata,
            "mdsims/protocol/Particle_Type",
            source_state.particle_types[0],
        )),
        device=device,
        progress_period=int(_metadata_value(
            source_metadata, "mdsims/output/Progress_Update_Period", 25_000
        )),
        phase_method=str(source_thermalization["Phase_Separation_Method"]),
        notes=request.notes,
    )
    config.validate()
    return config


def _metadata(
    run_id: str,
    signature: str,
    request: ExpandedCloneConfig,
    config: ThermalizationConfig,
    paths: RunPaths,
    device_name: str,
    source_state: StateData,
    source_frame_id: int,
    source_timestep: int,
    source_file_location: str,
    constructed: ExpandedCloneState,
    prior_lj_time: float,
) -> dict[str, dict[str, Any]]:
    relative = paths.relative_directory.as_posix()
    return {
        "mdsims/run": {
            "Run_ID": run_id,
            "Run_Signature": signature,
            "Sim_Type": EXPANDED_CLONE_SIM_TYPE,
            "Status": "Initializing",
            "Workflow_Version": EXPANDED_CLONE_METHOD_VERSION,
            "Canonical_Config_JSON": canonical_json(
                request.signature_parameters(source_frame_id, config.seed)
            ),
        },
        "mdsims/source": {
            "State_Role": "source",
            "Source_Type": "Thermalization_Run",
            "State_Creation_Method": EXPANDED_CLONE_STATE_VERSION,
            "Source_Run_ID": str(request.source_run_id),
            "Source_Frame_ID": int(source_frame_id),
            "Source_HOOMD_Timestep": int(source_timestep),
            "Source_File_Location": str(source_file_location),
        },
        "mdsims/protocol": {
            "N_Cells": int(config.n_fcc_cells),
            "Therm_kT": float(config.kT),
            "Therm_Seed": int(config.seed),
            "Nsteps": int(config.nsteps),
            "Preparation_Steps": int(request.preparation_steps),
            "Total_Integrated_Steps": int(
                request.preparation_steps + config.nsteps
            ),
            "dt": float(config.dt),
            "Ensemble": "NVT",
            "T_Set": float(config.kT),
            "Particle_Type": config.particle_type,
            "Device": device_name,
            "Inherited_Simulation_Settings": True,
            "Liquid_Scale": int(request.liquid_scale),
            "Vapor_Scale_Per_Side": int(request.vapor_scale),
            "Vapor_Density_Divisor": float(request.vapor_density_divisor),
            "Center_Length": float(constructed.center_length),
            "Center_Density_Actual": float(constructed.source_density),
            "Side_Density_Actual": float(constructed.vapor_density),
            "COM_Recenter_Period": int(request.com_recenter_period),
            "COM_Recenter_Method": COM_RECENTER_METHOD_VERSION,
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
            "Trajectory_Storage_Method": EXPANDED_CLONE_TRAJECTORY_VERSION,
        },
        "mdsims/states/source": {
            "N_Particles": int(source_state.n_particles),
            "Box": source_state.box,
            "Volume": float(source_state.volume),
            "Number_Density": float(source_state.density),
        },
        "mdsims/states/constructed": {
            "N_Particles": int(constructed.n_particles),
            "Source_Particles_Per_Tile": int(constructed.source_particles),
            "Particles_Per_Vapor_Tile": int(
                constructed.particles_per_vapor_tile
            ),
            "Liquid_Particles": int(constructed.liquid_particles),
            "Vapor_Particles_Per_Side": int(
                constructed.vapor_particles_per_side
            ),
            "Box": constructed.box,
            "Volume": float(constructed.volume),
            "Number_Density_Overall": float(constructed.density),
            "Velocity_Initialization": "regenerated_maxwell_boltzmann",
            "Vapor_Selection": (
                "independent_fixed_count_without_replacement"
            ),
        },
        "mdsims/time": {
            "Prior_Cumulative_LJ_Time": float(prior_lj_time),
        },
    }


def run_expanded_clone(
    request: ExpandedCloneConfig,
    project_paths: ProjectPaths | None = None,
    database: SQLiteRunDatabase | None = None,
) -> dict[str, Any]:
    """Construct, prepare, and evolve a thermalized-box clone expansion."""

    request.validate()
    project_paths = project_paths or ProjectPaths()
    database = database or SQLiteRunDatabase(project_paths.database)
    database.initialize()
    (
        source_master,
        source_thermalization,
        source_frame_id,
        effective_seed,
        note,
    ) = _source_context(request, database)
    signature = request.run_signature(source_frame_id, effective_seed)
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
    run_paths = project_paths.for_run(EXPANDED_CLONE_SIM_TYPE, run_id)
    database.update_master(
        run_id,
        Run_Signature=signature,
        N_Cells=int(source_master["N_Cells"]),
        Nsteps=int(request.nsteps),
        Current_Nstep=0,
        ElapsedTime=0.0,
        Last_Update_Time=utc_now(),
        Sim_Type=EXPANDED_CLONE_SIM_TYPE,
        Status="Initializing",
        Notes=note,
    )
    database.add_run_dependency(
        request.source_run_id,
        run_id,
        "expanded_clone_source",
    )

    current_step = 0
    elapsed_time = 0.0
    storage: RunStorage | None = None
    try:
        from .run_analysis import open_run

        source_run = open_run(
            request.source_run_id,
            project_paths=project_paths,
            database=database,
        )
        if source_run.frame_count != int(source_thermalization["Num_Frames"]):
            raise RuntimeError("Source GSD frame count does not match SQL")
        source_frame = source_run.load_frame(source_frame_id)
        source_state = _frame_state_data(source_frame)
        source_metadata = source_run.metadata()
        config = _inherited_config(
            request,
            source_master,
            source_thermalization,
            source_metadata,
            source_state,
            effective_seed,
        )
        constructed = build_expanded_clone_state(
            source_frame,
            request.liquid_scale,
            request.vapor_scale,
            request.vapor_density_divisor,
            effective_seed,
        )
        frame = make_expanded_clone_frame(constructed)
        simulation, thermo, device_name = _make_simulation_from_frame(
            config,
            frame,
            thermalize_momenta=True,
            ensemble="NVT",
        )
        prior_lj_time = (
            float(source_thermalization["Cumulative_LJ_Time"])
            + float(request.preparation_steps) * float(config.dt)
        )
        source_file_location = str(source_thermalization["File_Location"])
        source_timestep = int(source_frame.configuration.step)
        frame_schedule = expanded_fcc_frame_schedule(
            config.nsteps,
            config.log_period,
            request.com_recenter_period,
        )
        phase_frame_steps = {
            int(item["run_step"])
            for item in frame_schedule
            if item["phase_frame"]
        }
        saved_frame_steps = {
            int(item["run_step"]) for item in frame_schedule
        }
        storage = RunStorage(run_paths)
        storage.open(_metadata(
            run_id,
            signature,
            request,
            config,
            run_paths,
            device_name,
            source_state,
            source_frame_id,
            source_timestep,
            source_file_location,
            constructed,
            prior_lj_time,
        ))

        start_time = utc_now()
        database.update_master(
            run_id,
            StartTime=start_time,
            Last_Update_Time=start_time,
            Status="Running",
        )
        simulation.run(0)
        constructed_com = _recenter_simulation_com(simulation)
        initial_velocity = _remove_net_momentum(simulation)

        # Preparation is deliberately excluded from production run_step and
        # from thermodynamic logging. It is nevertheless included in elapsed
        # time, HOOMD timestep, and cumulative LJ time.
        remaining_preparation = int(request.preparation_steps)
        while remaining_preparation > 0:
            chunk = min(remaining_preparation, int(config.progress_period))
            started = time.perf_counter()
            simulation.run(chunk)
            elapsed_time += time.perf_counter() - started
            remaining_preparation -= chunk
            database.update_master(
                run_id,
                ElapsedTime=elapsed_time,
                Last_Update_Time=utc_now(),
            )
            storage.flush()

        post_preparation_com = _recenter_simulation_com(simulation)
        post_preparation_velocity = _remove_net_momentum(simulation)
        simulation.run(0)
        state = storage.record(
            simulation,
            thermo,
            0,
            config.dt,
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
                config.n_fcc_cells,
                0,
                simulation.timestep,
                0,
                config.dt,
                trajectory_relative,
                prior_lj_time=prior_lj_time,
            ),
            "mdsims/analysis/preparation": {
                "Constructed_COM_Before_Recenter": constructed_com,
                "Initial_COM_Velocity_Removed": initial_velocity,
                "Preparation_Steps": int(request.preparation_steps),
                "Post_Preparation_COM_Before_Recenter": post_preparation_com,
                "Post_Preparation_COM_Velocity_Removed": (
                    post_preparation_velocity
                ),
            },
        })

        recenter_steps: list[int] = []
        com_before_recenter: list[np.ndarray] = []
        next_log = int(config.log_period)
        next_progress = int(config.progress_period)
        next_recenter = int(request.com_recenter_period)
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
                    next_recenter += int(request.com_recenter_period)
            final_now = current_step == int(config.nsteps)
            log_now = current_step == next_log or recenter_now or final_now
            if log_now:
                state = storage.record(
                    simulation,
                    thermo,
                    current_step,
                    config.dt,
                    prior_lj_time=prior_lj_time,
                    save_frame=current_step in saved_frame_steps,
                )
                while next_log <= current_step:
                    next_log += int(config.log_period)
            if current_step == next_progress or final_now:
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
                f"Expanded clone saved {storage.frame_count} frames; "
                f"expected {expected_frames}"
            )
        phase_records = [
            record
            for record in storage.frame_records
            if int(record["run_step"]) in phase_frame_steps
        ]
        if len(phase_records) != 5:
            raise RuntimeError(
                "Expanded clone did not save five phase-analysis frames"
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
                prior_lj_time=prior_lj_time,
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
                    record["hoomd_timestep"] for record in storage.frame_records
                ],
                "Num_Frames": int(storage.frame_count),
                "Phase_Average_Trajectory_Frame_IDs": phase_frame_ids,
                "Phase_Average_Run_Steps": [
                    int(record["run_step"]) for record in phase_records
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
            "source_run_id": str(request.source_run_id),
            "source_frame_id": source_frame_id,
            "directory": run_paths.directory,
            "trajectory_path": run_paths.trajectory,
            "hdf5_path": run_paths.hdf5,
            "n_particles": constructed.n_particles,
            "liquid_particles": constructed.liquid_particles,
            "vapor_particles_per_side": constructed.vapor_particles_per_side,
            "particles_per_vapor_tile": (
                constructed.particles_per_vapor_tile
            ),
            "source_density": constructed.source_density,
            "vapor_density": constructed.vapor_density,
            "overall_density": constructed.density,
            "preparation_steps": int(request.preparation_steps),
            "production_steps": int(config.nsteps),
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
