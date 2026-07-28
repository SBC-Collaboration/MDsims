from pathlib import Path


# ============================================================
# Project roots
# ============================================================

PROJECT_ROOT = Path("/exp/e961/data/MDsims-data/pnichols")


SIMPLE_LATTICES_V3_ROOT = PROJECT_ROOT / "Simple_Lattices_v3"
THERMALIZED_STATES_V3_ROOT = PROJECT_ROOT / "Thermalized_States_v3"

CAVITATION_STATES_V3_ROOT = PROJECT_ROOT / "Cavitation_States_v3"
CAVITATION_EVOLVED_V3_ROOT = PROJECT_ROOT / "Cavitation_Evolved_v3"

EXCITATION_STATES_V3_ROOT = PROJECT_ROOT / "Excitation_States_v3"
EXCITATION_EVOLVED_V3_ROOT = PROJECT_ROOT / "Excitation_Evolved_v3"
EXCITATION_EVOLVED_V3_LEGACY_ROOT = (
    PROJECT_ROOT / "Excitation_Evolved_v3_legacy_single_dt"
)

MASTER_CSVS_V3_ROOT = PROJECT_ROOT / "Master_CSVs_v3"
RUN_LOGS_ROOT = PROJECT_ROOT / "run_logs"


def format_float(value, decimals=3):
    return f"{float(value):.{decimals}f}"


def format_dt(value):
    """Format a timestep size without hiding meaningful small digits."""

    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text if text else "0"


def center_label(center=None, random_location=False, seed=None):
    if random_location:
        if seed is None:
            return "random_center"
        return f"random_center_seed_{int(seed)}"

    if center is None:
        return "center_box"

    return (
        f"center_x_{format_float(center[0])}"
        f"_y_{format_float(center[1])}"
        f"_z_{format_float(center[2])}"
    )


def lattice_paths(
    n_fcc_cells,
    target_rho,
    base_folder=SIMPLE_LATTICES_V3_ROOT,
):
    folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / f"rho_{format_float(target_rho)}"
    )

    return {
        "folder": folder,
        "state_path": folder / "lattice.gsd",
        "metadata_path": folder / "lattice_metadata.hdf5",
        "state_kind": "lattice",
    }


def thermalized_run_paths(
    n_fcc_cells,
    target_rho,
    kT,
    nsteps,
    seed,
    phase_name="randomization",
    base_folder=THERMALIZED_STATES_V3_ROOT,
):
    seed_label = "unknown" if seed is None else str(int(seed))

    folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / f"rho_{format_float(target_rho)}"
        / f"kT_{format_float(kT)}"
        / f"nsteps_{int(nsteps)}"
        / f"seed_{seed_label}"
    )

    return {
        "folder": folder,
        "state_path": folder / f"{phase_name}.gsd",
        "log_path": folder / f"{phase_name}_log.hdf5",
        "phase_name": phase_name,
        "state_kind": "thermalized",
    }


def cavitation_state_paths(
    n_fcc_cells,
    source_rho,
    kT,
    source_nsteps,
    source_seed,
    radius,
    source_phase_name="randomization",
    center=None,
    random_location=False,
    bubble_seed=None,
    base_folder=CAVITATION_STATES_V3_ROOT,
):
    folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / f"source_rho_{format_float(source_rho)}"
        / f"kT_{format_float(kT)}"
        / f"source_nsteps_{int(source_nsteps)}"
        / f"source_seed_{int(source_seed)}"
        / f"source_phase_{source_phase_name}"
        / f"radius_{format_float(radius)}"
        / center_label(
            center=center,
            random_location=random_location,
            seed=bubble_seed,
        )
    )

    return {
        "folder": folder,
        "state_path": folder / "cavitation_initial.gsd",
        "creation_metadata_path": folder / "cavitation_creation.hdf5",
        "state_kind": "cavitation_initial",
    }


def cavitation_evolved_paths(
    n_fcc_cells,
    source_rho,
    kT,
    source_nsteps,
    source_seed,
    radius,
    evolve_kT,
    evolve_nsteps,
    evolve_seed,
    source_phase_name="randomization",
    center=None,
    random_location=False,
    bubble_seed=None,
    base_folder=CAVITATION_EVOLVED_V3_ROOT,
):
    state_paths = cavitation_state_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=source_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        radius=radius,
        source_phase_name=source_phase_name,
        center=center,
        random_location=random_location,
        bubble_seed=bubble_seed,
        base_folder=base_folder,
    )

    folder = (
        state_paths["folder"]
        / f"evolve_kT_{format_float(evolve_kT)}"
        / f"nsteps_{int(evolve_nsteps)}"
        / f"seed_{int(evolve_seed)}"
    )

    return {
        "folder": folder,
        "trajectory_path": folder / "cavitation_trajectory.gsd",
        "final_state_path": folder / "cavitation_final.gsd",
        "log_path": folder / "cavitation_log.hdf5",
        "state_kind": "cavitation_evolved",
    }


def excitation_state_paths(
    n_fcc_cells,
    source_rho,
    kT,
    source_nsteps,
    source_seed,
    method,
    radius,
    energy,
    source_phase_name="randomization",
    center=None,
    random_location=False,
    excitation_seed=None,
    base_folder=EXCITATION_STATES_V3_ROOT,
):
    folder = (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{int(n_fcc_cells)}"
        / f"source_kT_{format_float(kT)}"
        / f"rho_{format_float(source_rho)}"
        / f"source_nsteps_{int(source_nsteps)}"
        / f"source_seed_{int(source_seed)}"
        / f"method_{method}"
        / f"radius_{format_float(radius)}"
        / f"energy_{format_float(energy)}"
        / center_label(
            center=center,
            random_location=random_location,
            seed=excitation_seed,
        )
    )

    return {
        "folder": folder,
        "state_path": folder / "excitation_initial.gsd",
        "creation_metadata_path": folder / "excitation_creation.hdf5",
        "state_kind": "excitation_initial",
    }


def excitation_evolved_paths(
    n_fcc_cells,
    source_rho,
    kT,
    source_nsteps,
    source_seed,
    method,
    radius,
    energy,
    dt2=None,
    nsteps2=None,
    evolve_seed=None,
    dt1=0.0005,
    nsteps1=200_000,
    source_phase_name="randomization",
    center=None,
    random_location=False,
    excitation_seed=None,
    base_folder=EXCITATION_EVOLVED_V3_ROOT,
    evolve_kT=None,
    evolve_nsteps=None,
    dt=None,
    ensemble="NVE",
    pressure=None,
    tauS=None,
    pressure_couple="xyz",
    barostat_gamma=0.0,
):
    """
    Build paths for the two-segment V3 excitation evolution format.

    ``dt1`` and ``nsteps1`` describe the short-timestep first segment.
    ``dt2`` and ``nsteps2`` describe the caller-selected second segment.
    ``dt`` and ``evolve_nsteps`` remain temporary aliases for older notebooks.
    """

    if dt2 is None:
        dt2 = dt
    if nsteps2 is None:
        nsteps2 = evolve_nsteps

    if dt2 is None:
        raise ValueError("dt2 is required")
    if nsteps2 is None:
        raise ValueError("nsteps2 is required")
    if evolve_seed is None:
        raise ValueError("evolve_seed is required")
    if float(dt1) <= 0 or float(dt2) <= 0:
        raise ValueError("dt1 and dt2 must be positive")
    if int(nsteps1) <= 0 or int(nsteps2) <= 0:
        raise ValueError("nsteps1 and nsteps2 must be positive")

    ensemble = str(ensemble).upper()
    if ensemble not in {"NVE", "NPH"}:
        raise ValueError("ensemble must be 'NVE' or 'NPH'")
    if ensemble == "NPH":
        if pressure is None:
            raise ValueError("pressure is required when ensemble='NPH'")
        if tauS is None:
            tauS = 1000.0 * float(dt2)
        if float(tauS) <= 0:
            raise ValueError("tauS must be positive")

    state_paths = excitation_state_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=source_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
        method=method,
        radius=radius,
        energy=energy,
        center=center,
        random_location=random_location,
        excitation_seed=excitation_seed,
        base_folder=base_folder,
    )

    evolution_root = state_paths["folder"]
    if ensemble == "NPH":
        evolution_root = (
            evolution_root
            / "ensemble_NPH"
            / f"pressure_{format_float(pressure)}"
            / f"tauS_{format_float(tauS)}"
            / f"couple_{pressure_couple}"
            / f"gamma_{format_float(barostat_gamma)}"
        )

    folder = (
        evolution_root
        / f"segment_1_dt_{format_dt(dt1)}"
        / f"nsteps_{int(nsteps1)}"
        / f"segment_2_dt_{format_dt(dt2)}"
        / f"nsteps_{int(nsteps2)}"
        / f"seed_{int(evolve_seed)}"
    )

    segment_1_folder = folder / "segment_1"
    segment_2_folder = folder / "segment_2"

    def segment_paths(segment_index, segment_folder, segment_dt, segment_nsteps):
        return {
            "segment_index": int(segment_index),
            "folder": segment_folder,
            "dt": float(segment_dt),
            "tauS": (
                float(tauS)
                if ensemble == "NPH"
                else None
            ),
            "nsteps": int(segment_nsteps),
            "trajectory_path": (
                segment_folder / "excitation_trajectory.gsd"
            ),
            "final_state_path": segment_folder / "excitation_final.gsd",
            "log_path": segment_folder / "excitation_log.hdf5",
            "barostat_dof_path": (
                segment_folder / "barostat_dof.npy"
                if ensemble == "NPH"
                else None
            ),
            "state_kind": "excitation_evolved_segment",
        }

    segment_1 = segment_paths(1, segment_1_folder, dt1, nsteps1)
    segment_2 = segment_paths(2, segment_2_folder, dt2, nsteps2)

    return {
        "folder": folder,
        "manifest_path": folder / "evolution_manifest.hdf5",
        "segment_1": segment_1,
        "segment_2": segment_2,
        "segment_paths": [segment_1, segment_2],
        "trajectory_paths": [
            segment_1["trajectory_path"],
            segment_2["trajectory_path"],
        ],
        "log_paths": [
            segment_1["log_path"],
            segment_2["log_path"],
        ],
        # Compatibility keys point at the overall final segment.
        "trajectory_path": segment_2["trajectory_path"],
        "final_state_path": segment_2["final_state_path"],
        "log_path": segment_2["log_path"],
        "dt1": float(dt1),
        "nsteps1": int(nsteps1),
        "dt2": float(dt2),
        "nsteps2": int(nsteps2),
        "total_nsteps": int(nsteps1) + int(nsteps2),
        "total_physical_time": (
            float(dt1) * int(nsteps1)
            + float(dt2) * int(nsteps2)
        ),
        "state_kind": "excitation_evolved",
        "evolution_format": "two_segment_dt_v1",
        "ensemble": ensemble,
        "pressure": None if pressure is None else float(pressure),
        "tauS": None if tauS is None else float(tauS),
        "pressure_couple": str(pressure_couple),
        "barostat_gamma": float(barostat_gamma),
    }


def legacy_excitation_evolved_paths(
    n_fcc_cells,
    source_rho,
    kT,
    source_nsteps,
    source_seed,
    method,
    radius,
    energy,
    evolve_nsteps,
    evolve_seed,
    dt=0.0005,
    source_phase_name="randomization",
    center=None,
    random_location=False,
    excitation_seed=None,
    base_folder=EXCITATION_EVOLVED_V3_LEGACY_ROOT,
):
    """Build paths for archived single-dt excitation results."""

    state_paths = excitation_state_paths(
        n_fcc_cells=n_fcc_cells,
        source_rho=source_rho,
        kT=kT,
        source_nsteps=source_nsteps,
        source_seed=source_seed,
        source_phase_name=source_phase_name,
        method=method,
        radius=radius,
        energy=energy,
        center=center,
        random_location=random_location,
        excitation_seed=excitation_seed,
        base_folder=base_folder,
    )
    folder = (
        state_paths["folder"]
        / f"dt_{format_dt(dt)}"
        / f"nsteps_{int(evolve_nsteps)}"
        / f"seed_{int(evolve_seed)}"
    )
    return {
        "folder": folder,
        "trajectory_path": folder / "excitation_trajectory.gsd",
        "final_state_path": folder / "excitation_final.gsd",
        "log_path": folder / "excitation_log.hdf5",
        "state_kind": "excitation_evolved_legacy_single_dt",
    }


def index_path(name="v3_simulation_index", base_folder=MASTER_CSVS_V3_ROOT):
    return Path(base_folder) / f"{name}.parquet"
