"""Isolated timestep-convergence runs for NVT thermalization."""

import json
from pathlib import Path

import hoomd

from . import lattices
from . import metadata as metadata_helpers
from . import runs
from . import simulation as simulation_helpers
from .paths import PROJECT_ROOT, lattice_paths


DEFAULT_DT_VALIDATION_ROOT = (
    PROJECT_ROOT / "Validation" / "Thermalization_dt_test_v1"
)


def _format_dt(value):
    return f"{float(value):.5f}"


def _format_physical_time(value):
    return f"{float(value):.3f}"


def timestep_validation_paths(
    n_fcc_cells,
    target_rho,
    kT,
    dt,
    physical_time,
    seed,
    base_folder=DEFAULT_DT_VALIDATION_ROOT,
):
    """Build paths that explicitly identify dt and physical duration."""
    folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / f"rho_{float(target_rho):.3f}"
        / f"kT_{float(kT):.3f}"
        / f"physical_time_{_format_physical_time(physical_time)}"
        / f"dt_{_format_dt(dt)}"
        / f"seed_{int(seed)}"
    )

    sweep_folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / f"rho_{float(target_rho):.3f}"
        / f"kT_{float(kT):.3f}"
        / f"physical_time_{_format_physical_time(physical_time)}"
    )

    return {
        "folder": folder,
        "state_path": folder / "thermalized_final.gsd",
        "log_path": folder / "thermalization_log.hdf5",
        "trajectory_path": folder / "thermalization_trajectory.gsd",
        "manifest_path": sweep_folder / "sweep_manifest.json",
    }


def nsteps_for_physical_time(physical_time, dt):
    """Return the nearest whole number of steps for a physical duration."""
    physical_time = float(physical_time)
    dt = float(dt)
    if physical_time <= 0:
        raise ValueError("physical_time must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    nsteps = int(round(physical_time / dt))
    if nsteps < 1:
        raise ValueError("physical_time must span at least one timestep")
    return nsteps


def _write_manifest(manifest_path, configuration, results):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": configuration,
        "runs": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
                if key != "simulation"
            }
            for result in results
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def run_thermalization_dt_sweep(
    n_fcc_cells,
    target_rho,
    kT,
    timesteps,
    physical_time,
    seeds=(1,),
    log_interval=5.0,
    trajectory_interval=None,
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    buffer_LJ=0.4,
    lj_mode="xplor",
    r_on_LJ=2.0,
    base_folder=DEFAULT_DT_VALIDATION_ROOT,
    overwrite=False,
):
    """
    Run isolated NVT thermalizations at several timesteps.

    Every run covers approximately the same physical duration. Results are
    written beneath ``base_folder`` and are never placed in the production
    Thermalized_States_v3 tree.
    """
    timesteps = [float(dt) for dt in timesteps]
    seeds = [int(seed) for seed in seeds]
    physical_time = float(physical_time)
    log_interval = float(log_interval)
    if trajectory_interval is not None:
        trajectory_interval = float(trajectory_interval)

    if not timesteps:
        raise ValueError("timesteps must contain at least one value")
    if not seeds:
        raise ValueError("seeds must contain at least one value")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if trajectory_interval is not None and trajectory_interval <= 0:
        raise ValueError("trajectory_interval must be positive")

    frame = lattices.make_lattice_frame(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        end_print=True,
        overwrite=False,
    )

    configuration = {
        "validation_kind": "thermalization_dt_convergence",
        "base_folder": str(Path(base_folder)),
        "n_fcc_cells": int(n_fcc_cells),
        "target_rho": float(target_rho),
        "kT": float(kT),
        "timesteps": timesteps,
        "physical_time_requested": physical_time,
        "seeds": seeds,
        "log_interval_requested": log_interval,
        "trajectory_interval_requested": trajectory_interval,
        "epsilon_LJ": float(epsilon_LJ),
        "sigma_LJ": float(sigma_LJ),
        "r_cut_LJ": float(r_cut_LJ),
        "buffer_LJ": float(buffer_LJ),
        "lj_mode": str(lj_mode),
        "r_on_LJ": float(r_on_LJ),
    }
    results = []
    manifest_path = timestep_validation_paths(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        dt=timesteps[0],
        physical_time=physical_time,
        seed=seeds[0],
        base_folder=base_folder,
    )["manifest_path"]
    _write_manifest(manifest_path, configuration, results)

    starting_state_path = str(lattice_paths(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
    )["state_path"])

    for dt in timesteps:
        nsteps = nsteps_for_physical_time(physical_time, dt)
        actual_physical_time = nsteps * dt
        log_period = max(1, int(round(log_interval / dt)))
        trajectory_period = None
        if trajectory_interval is not None:
            trajectory_period = max(
                1,
                int(round(trajectory_interval / dt)),
            )

        for seed in seeds:
            paths = timestep_validation_paths(
                n_fcc_cells=n_fcc_cells,
                target_rho=target_rho,
                kT=kT,
                dt=dt,
                physical_time=physical_time,
                seed=seed,
                base_folder=base_folder,
            )

            state_exists = paths["state_path"].exists()
            log_exists = paths["log_path"].exists()
            trajectory_exists = paths["trajectory_path"].exists()
            output_complete = state_exists and log_exists
            if trajectory_interval is not None:
                output_complete = output_complete and trajectory_exists

            if output_complete and not overwrite:
                results.append({
                    **paths,
                    "dt": dt,
                    "seed": seed,
                    "nsteps": nsteps,
                    "physical_time_actual": actual_physical_time,
                    "trajectory_period": trajectory_period,
                    "status": "existing",
                })
                continue
            requested_output_exists = state_exists or log_exists
            if trajectory_interval is not None:
                requested_output_exists = (
                    requested_output_exists or trajectory_exists
                )
            if requested_output_exists and not overwrite:
                raise FileExistsError(
                    "Incomplete validation output exists. Use overwrite=True "
                    f"after inspecting: {paths['folder']}"
                )

            simulation = simulation_helpers.make_simulation(
                frame=frame,
                target_rho=target_rho,
                n_fcc_cells=n_fcc_cells,
                seed=seed,
                dt=dt,
                kT=kT,
                ensemble="NVT",
                epsilon_LJ=epsilon_LJ,
                sigma_LJ=sigma_LJ,
                r_cut_LJ=r_cut_LJ,
                buffer_LJ=buffer_LJ,
                lj_mode=lj_mode,
                r_on_LJ=r_on_LJ,
                starting_state_path=starting_state_path,
            )
            simulation.state.thermalize_particle_momenta(
                filter=hoomd.filter.All(),
                kT=kT,
            )

            logger_handle = runs.start_hdf5_logger(
                simulation=simulation,
                log_path=paths["log_path"],
                log_period=log_period,
            )
            trajectory_handle = None
            if trajectory_period is not None:
                trajectory_handle = runs.start_gsd_trajectory_writer(
                    simulation=simulation,
                    trajectory_path=paths["trajectory_path"],
                    trajectory_period=trajectory_period,
                    mode="wb",
                )
            try:
                simulation.run(0)
                simulation.run(nsteps)
            finally:
                if trajectory_handle is not None:
                    runs.stop_gsd_trajectory_writer(
                        simulation=simulation,
                        writer_handle=trajectory_handle,
                    )
                runs.stop_hdf5_logger(
                    simulation=simulation,
                    logger_objects=logger_handle,
                )

            metadata = runs.build_simulation_metadata(
                simulation=simulation,
                phase_name="thermalization_dt_validation",
                n_fcc_cells=n_fcc_cells,
                target_rho=target_rho,
                seed=seed,
                dt=dt,
                kT=kT,
                epsilon_LJ=epsilon_LJ,
                sigma_LJ=sigma_LJ,
                r_cut_LJ=r_cut_LJ,
                r_on_LJ=r_on_LJ,
                buffer_LJ=buffer_LJ,
                lj_mode=lj_mode,
                log_period=log_period,
                nsteps=nsteps,
                starting_state_path=starting_state_path,
            )
            metadata.update({
                "state_kind": "dt_validation_thermalized",
                "run_kind": "thermalization_dt_validation",
                "state_path": str(paths["state_path"]),
                "log_path": str(paths["log_path"]),
            })
            runs.write_hdf5_metadata(paths["log_path"], metadata)
            metadata_helpers.write_metadata_groups(
                hdf5_path=paths["log_path"],
                groups={
                    "metadata/validation": {
                        "physical_time_requested": physical_time,
                        "physical_time_actual": actual_physical_time,
                        "log_interval_requested": log_interval,
                        "trajectory_interval_requested": trajectory_interval,
                        "trajectory_period": trajectory_period,
                        "validation_root": str(Path(base_folder)),
                    }
                },
                mode="a",
                overwrite=True,
            )
            runs.save_final_state(simulation, paths["state_path"])

            results.append({
                **paths,
                "dt": dt,
                "seed": seed,
                "nsteps": nsteps,
                "physical_time_actual": actual_physical_time,
                "log_period": log_period,
                "trajectory_period": trajectory_period,
                "status": "created",
            })
            _write_manifest(manifest_path, configuration, results)

    manifest_path = _write_manifest(manifest_path, configuration, results)
    return {
        "base_folder": Path(base_folder),
        "manifest_path": manifest_path,
        "configuration": configuration,
        "runs": results,
    }
