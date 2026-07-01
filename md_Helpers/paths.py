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

MASTER_CSVS_V3_ROOT = PROJECT_ROOT / "Master_CSVs_v3"


def format_float(value, decimals=3):
    return f"{float(value):.{decimals}f}"


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
        / f"source_rho_{format_float(source_rho)}"
        / f"kT_{format_float(kT)}"
        / f"source_nsteps_{int(source_nsteps)}"
        / f"source_seed_{int(source_seed)}"
        / f"source_phase_{source_phase_name}"
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
    evolve_kT,
    evolve_nsteps,
    evolve_seed,
    source_phase_name="randomization",
    center=None,
    random_location=False,
    excitation_seed=None,
    base_folder=EXCITATION_EVOLVED_V3_ROOT,
):
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
        / f"evolve_kT_{format_float(evolve_kT)}"
        / f"nsteps_{int(evolve_nsteps)}"
        / f"seed_{int(evolve_seed)}"
    )

    return {
        "folder": folder,
        "trajectory_path": folder / "excitation_trajectory.gsd",
        "final_state_path": folder / "excitation_final.gsd",
        "log_path": folder / "excitation_log.hdf5",
        "state_kind": "excitation_evolved",
    }


def index_path(name="v3_simulation_index", base_folder=MASTER_CSVS_V3_ROOT):
    return Path(base_folder) / f"{name}.parquet"
