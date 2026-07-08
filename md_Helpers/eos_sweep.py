from pathlib import Path

import numpy as np
import pandas as pd

from .paths import MASTER_CSVS_V3_ROOT


DEFAULT_LOWER_STOP = -0.03
DEFAULT_UPPER_STOP = 0.18


def rounded_range(start, stop, step, decimals=6):
    """
    Inclusive floating-point range for density and temperature grids.
    """

    start = float(start)
    stop = float(stop)
    step = float(step)

    if step <= 0:
        raise ValueError("step must be positive")

    count = int(np.floor((stop - start) / step + 0.5)) + 1
    values = start + step * np.arange(max(count, 0))
    values = values[values <= stop + 0.5 * step]

    return [round(float(value), decimals) for value in values]


def default_eos_summary_path(
    n_fcc_cells,
    output_path=None,
    output_name=None,
):
    if output_path is not None:
        return Path(output_path)

    if output_name is None:
        output_name = f"adaptive_pressure_window_ncells_{int(n_fcc_cells)}.csv"

    return Path(MASTER_CSVS_V3_ROOT) / output_name


def default_eos_plot_paths(n_fcc_cells, output_dir=None):
    if output_dir is None:
        output_dir = Path(MASTER_CSVS_V3_ROOT) / "EOS_Plots"
    else:
        output_dir = Path(output_dir)

    return {
        "pressure": output_dir / f"PvsDensity_Ncells_{int(n_fcc_cells)}.png",
        "potential_energy": (
            output_dir / f"PEvsDensity_Ncells_{int(n_fcc_cells)}.png"
        ),
    }


def clean_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None or pd.isna(value):
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pressure_stop_region(
    pressure,
    lower_stop=DEFAULT_LOWER_STOP,
    upper_stop=DEFAULT_UPPER_STOP,
):
    if pd.isna(pressure):
        return "missing"

    pressure = float(pressure)

    if pressure < lower_stop:
        return "below_stop"

    if pressure <= upper_stop:
        return "inside_stops"

    return "above_stop"


def _density_key(value):
    return round(float(value), 6)


def _upsert_row(rows_by_key, row):
    key = (
        int(row["n_fcc_cells"]),
        _density_key(row["target_rho"]),
        round(float(row["kT"]), 6),
        int(row["nsteps"]),
        int(row["seed"]),
    )

    rows_by_key[key] = row
    return key


def _load_existing_rows(output_path):
    output_path = Path(output_path)
    rows_by_key = {}

    if not output_path.exists():
        return rows_by_key

    existing = pd.read_csv(output_path)

    required = {"n_fcc_cells", "target_rho", "kT", "nsteps", "seed"}
    if not required.issubset(existing.columns):
        return rows_by_key

    for row in existing.to_dict("records"):
        _upsert_row(rows_by_key, row)

    return rows_by_key


def _pressure_column(n_last):
    return f"pressure_mean_last{int(n_last)}"


def _pe_column(n_last):
    return f"PE_per_particle_mean_last{int(n_last)}"


def _summarize_thermalized_state(
    n_fcc_cells,
    target_rho,
    kT,
    nsteps,
    seed,
    log_period,
    n_last,
    overwrite=False,
    overwrite_lattice=False,
    phase_name="randomization",
    **simulation_kwargs,
):
    from . import master_csv
    from . import simulation

    result = simulation.get_or_make_thermalized_state(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        kT=kT,
        nsteps=nsteps,
        seed=seed,
        log_period=log_period,
        phase_name=phase_name,
        overwrite=overwrite,
        overwrite_lattice=overwrite_lattice,
        **simulation_kwargs,
    )

    row = master_csv.summarize_thermalization_log(
        result["paths"]["log_path"],
        n_last=n_last,
    )
    row["created_new"] = bool(result.get("created_new", False))

    return row


def _annotate_sweep_row(
    row,
    lower_stop,
    upper_stop,
    scan_direction,
    adaptive_pass,
    n_last,
):
    pressure_col = _pressure_column(n_last)
    pressure = row.get(pressure_col, np.nan)

    row["pressure_stop_region"] = pressure_stop_region(
        pressure,
        lower_stop=lower_stop,
        upper_stop=upper_stop,
    )
    row["inside_pressure_stops"] = (
        not pd.isna(pressure)
        and lower_stop <= float(pressure) <= upper_stop
    )
    row["scan_direction"] = scan_direction
    row["adaptive_pass"] = adaptive_pass

    return row


def _choose_start_density(
    rows_by_key,
    n_fcc_cells,
    kT,
    nsteps,
    seed,
    rho_min_hard,
    rho_max_hard,
    fallback_rho,
    n_last,
    lower_stop,
    upper_stop,
):
    pressure_midpoint = 0.5 * (lower_stop + upper_stop)
    candidates = []

    for row in rows_by_key.values():
        if int(row.get("n_fcc_cells", -1)) != int(n_fcc_cells):
            continue
        if round(float(row.get("kT", np.nan)), 6) != round(float(kT), 6):
            continue
        if int(row.get("nsteps", -1)) != int(nsteps):
            continue
        if int(row.get("seed", -1)) != int(seed):
            continue

        rho = float(row.get("target_rho", np.nan))
        pressure = float(row.get(_pressure_column(n_last), np.nan))
        if pd.isna(rho) or pd.isna(pressure):
            continue
        if not (rho_min_hard <= rho <= rho_max_hard):
            continue

        candidates.append((abs(pressure - pressure_midpoint), rho))

    if candidates:
        return min(candidates)[1]

    return float(fallback_rho)


def run_eos_pressure_window_sweep(
    n_fcc_cells=28,
    kT_start=0.70,
    kT_stop=1.00,
    kT_step=0.02,
    initial_rho=0.71,
    rho_step=0.005,
    rho_min_hard=0.50,
    rho_max_hard=0.85,
    lower_stop=DEFAULT_LOWER_STOP,
    upper_stop=DEFAULT_UPPER_STOP,
    nsteps=1_000_000,
    log_period=1_000,
    seed=1,
    n_last=100,
    output_path=None,
    output_name=None,
    overwrite=False,
    overwrite_lattice=False,
    phase_name="randomization",
    stop_down_on_phase_separation=True,
    continue_on_error=False,
    **simulation_kwargs,
):
    """
    Run or load a V3 liquid EOS sweep between pressure stop bounds.

    The defaults reproduce the old adaptive sweep extent used for the n=30 EOS
    plots: scan down until pressure drops below -0.03 and scan up until
    pressure rises above 0.18. Rerunning this helper updates the summary CSV
    and reuses existing thermalized states unless ``overwrite=True``.
    """

    output_path = default_eos_summary_path(
        n_fcc_cells=n_fcc_cells,
        output_path=output_path,
        output_name=output_name,
    )
    rows_by_key = _load_existing_rows(output_path)
    kT_values = rounded_range(kT_start, kT_stop, kT_step, decimals=6)

    for kT in kT_values:
        start_rho = _choose_start_density(
            rows_by_key=rows_by_key,
            n_fcc_cells=n_fcc_cells,
            kT=kT,
            nsteps=nsteps,
            seed=seed,
            rho_min_hard=rho_min_hard,
            rho_max_hard=rho_max_hard,
            fallback_rho=initial_rho,
            n_last=n_last,
            lower_stop=lower_stop,
            upper_stop=upper_stop,
        )
        start_rho = round(start_rho / rho_step) * rho_step
        start_rho = min(max(start_rho, rho_min_hard), rho_max_hard)

        for scan_direction, rho_values in [
            (
                "up",
                rounded_range(start_rho, rho_max_hard, rho_step),
            ),
            (
                "down",
                rounded_range(
                    rho_min_hard,
                    max(rho_min_hard, start_rho - rho_step),
                    rho_step,
                )[::-1],
            ),
        ]:
            for target_rho in rho_values:
                print(
                    "EOS sweep:",
                    f"n={int(n_fcc_cells)}",
                    f"kT={float(kT):.3f}",
                    f"rho={float(target_rho):.3f}",
                    f"direction={scan_direction}",
                )

                error = None
                try:
                    row = _summarize_thermalized_state(
                        n_fcc_cells=n_fcc_cells,
                        target_rho=target_rho,
                        kT=kT,
                        nsteps=nsteps,
                        seed=seed,
                        log_period=log_period,
                        n_last=n_last,
                        overwrite=overwrite,
                        overwrite_lattice=overwrite_lattice,
                        phase_name=phase_name,
                        **simulation_kwargs,
                    )
                except Exception as caught_error:
                    error = caught_error
                    row = {
                        "status": "failed",
                        "error": repr(caught_error),
                        "n_fcc_cells": int(n_fcc_cells),
                        "target_rho": float(target_rho),
                        "actual_rho": float(target_rho),
                        "kT": float(kT),
                        "nsteps": int(nsteps),
                        "seed": int(seed),
                    }

                row = _annotate_sweep_row(
                    row=row,
                    lower_stop=lower_stop,
                    upper_stop=upper_stop,
                    scan_direction=scan_direction,
                    adaptive_pass="pressure_window",
                    n_last=n_last,
                )
                _upsert_row(rows_by_key, row)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows_by_key.values()).to_csv(
                    output_path,
                    index=False,
                )

                if error is not None and not continue_on_error:
                    raise error

                pressure = row.get(_pressure_column(n_last), np.nan)
                phase_separated = clean_bool(row.get("phase_separated", False))

                if pd.isna(pressure):
                    print(
                        "Stopping this scan direction because pressure "
                        "was not available."
                    )
                    break

                pressure = float(pressure)

                if scan_direction == "up" and pressure > upper_stop:
                    break

                if scan_direction == "down":
                    if pressure < lower_stop:
                        break
                    if stop_down_on_phase_separation and phase_separated:
                        break

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_by_key.values()).to_csv(output_path, index=False)

    table = pd.DataFrame(rows_by_key.values())

    if not table.empty:
        sort_columns = [
            col for col in [
                "n_fcc_cells",
                "kT",
                "target_rho",
                "nsteps",
                "seed",
            ]
            if col in table.columns
        ]
        table = table.sort_values(sort_columns).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    return table


def liquid_eos_table(
    table,
    n_last=100,
    n_fcc_cells=None,
    completed_only=True,
    non_phase_separated_only=True,
):
    table = pd.read_csv(table) if isinstance(table, (str, Path)) else table
    df = table.copy()

    if n_fcc_cells is not None and "n_fcc_cells" in df.columns:
        df = df[df["n_fcc_cells"].astype(int) == int(n_fcc_cells)]

    if completed_only and "status" in df.columns:
        df = df[df["status"].eq("completed")]

    if non_phase_separated_only and "phase_separated" in df.columns:
        df = df[~df["phase_separated"].apply(clean_bool)]

    required = ["actual_rho", "kT", _pressure_column(n_last), _pe_column(n_last)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"EOS table is missing columns: {missing}")

    return df.sort_values(["kT", "actual_rho"]).reset_index(drop=True)


def plot_eos_density_curves(
    table,
    n_last=100,
    n_fcc_cells=None,
    output_dir=None,
    save=True,
    show=True,
):
    """
    Plot the pressure-density and PE/N-density curves used for Seitz inputs.
    """

    import matplotlib.pyplot as plt

    df = liquid_eos_table(
        table,
        n_last=n_last,
        n_fcc_cells=n_fcc_cells,
    )

    if df.empty:
        raise ValueError("No completed non-phase-separated EOS rows to plot")

    if n_fcc_cells is None and "n_fcc_cells" in df.columns:
        unique_n = sorted(df["n_fcc_cells"].dropna().astype(int).unique())
        n_label = unique_n[0] if len(unique_n) == 1 else "mixed"
    else:
        n_label = int(n_fcc_cells)

    pressure_col = _pressure_column(n_last)
    pe_col = _pe_column(n_last)
    paths = (
        default_eos_plot_paths(n_label, output_dir=output_dir)
        if save
        else {}
    )

    figures = {}

    fig, ax = plt.subplots(figsize=(12, 8))
    for kT, group in df.groupby("kT"):
        ax.errorbar(
            group["actual_rho"],
            group[pressure_col],
            yerr=group.get(f"pressure_std_last{int(n_last)}"),
            marker="o",
            capsize=3,
            label=f"kT = {float(kT):.2f}",
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.35)
    ax.axhline(0.2, color="black", linewidth=1, alpha=0.2, linestyle="--")
    ax.set_xlabel("Actual density, N / V")
    ax.set_ylabel("Pressure")
    ax.set_title(
        "Pressure vs actual density\n"
        f"FCC, n_cells = {n_label}, non-phase-separated only, "
        "connected by temperature"
    )
    ax.grid(alpha=0.3)
    ax.legend(title="Temperature", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    figures["pressure"] = fig

    fig, ax = plt.subplots(figsize=(12, 8))
    for kT, group in df.groupby("kT"):
        ax.errorbar(
            group["actual_rho"],
            group[pe_col],
            yerr=group.get(f"PE_per_particle_std_last{int(n_last)}"),
            marker="o",
            capsize=3,
            label=f"kT = {float(kT):.2f}",
        )
    ax.set_xlabel("Actual density, N / V")
    ax.set_ylabel("PE / N")
    ax.set_title(
        "Potential energy per particle vs actual density\n"
        f"FCC, n_cells = {n_label}, non-phase-separated only, "
        "connected by temperature"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figures["potential_energy"] = fig

    if save:
        for plot_path in paths.values():
            plot_path.parent.mkdir(parents=True, exist_ok=True)
        figures["pressure"].savefig(paths["pressure"], dpi=160)
        figures["potential_energy"].savefig(paths["potential_energy"], dpi=160)

    if show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)

    return {
        "table": df,
        "figures": figures,
        "paths": paths,
    }
