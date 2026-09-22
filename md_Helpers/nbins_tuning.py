"""In-memory sensitivity checks for the cavitation voxel-bin choice."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .paths import ProjectPaths
from .seitz import calculate_cavitation_seitz, query_seitz_eos_states
from .voxel_fit import PHASE_FIT_SQL_FIELDS, fit_trajectory_voxel_mixture


def _validated_nbins(nbins_values: Iterable[int]) -> list[int]:
    values = list(dict.fromkeys(int(value) for value in nbins_values))
    if not values or any(value <= 0 for value in values):
        raise ValueError("nbins_values must contain positive integers")
    return values


def _trajectory_path(row, project_paths: ProjectPaths) -> Path:
    location = Path(str(row["File_Location"])).expanduser()
    if not location.is_absolute():
        location = project_paths.top_directory / location
    return location / "trajectory.gsd"


def refit_cavitation_nbins(
    cavitations,
    nbins_values: Iterable[int],
    *,
    project_paths: ProjectPaths | None = None,
    num_frames: int = 5,
    frame_indices=None,
    **fit_options: Any,
):
    """Refit selected cavitation trajectories for several voxel resolutions.

    Results exist only in the returned DataFrame. This function never updates
    SQL, HDF5, or GSD files.
    """

    import pandas as pd

    required = {"Run_ID", "N_Cells", "File_Location"}
    missing = required - set(cavitations.columns)
    if missing:
        raise ValueError(f"cavitations is missing columns: {sorted(missing)}")
    if cavitations.empty:
        raise ValueError("No cavitation rows were provided")
    bins = _validated_nbins(nbins_values)
    paths = project_paths or ProjectPaths()
    records = []
    for _, row in cavitations.iterrows():
        trajectory_path = _trajectory_path(row, paths)
        for nbins in bins:
            fit = fit_trajectory_voxel_mixture(
                trajectory_path,
                int(row["N_Cells"]),
                num_frames=num_frames,
                frame_indices=frame_indices,
                nbins=nbins,
                **fit_options,
            )
            records.append({"Run_ID": str(row["Run_ID"]), **fit})
    return pd.DataFrame.from_records(records).sort_values(
        ["voxel_nbins", "Run_ID"]
    ).reset_index(drop=True)


def cavitations_with_nbins_fit(cavitations, fit_results, nbins: int):
    """Return a copy of cavitation rows with one trial fit substituted."""

    trial = fit_results.loc[fit_results["voxel_nbins"] == int(nbins)].copy()
    if trial.empty:
        raise ValueError(f"No trial fits were found for nbins={int(nbins)}")
    if trial["Run_ID"].duplicated().any():
        raise ValueError(f"Trial fits contain duplicate runs for nbins={int(nbins)}")
    result = cavitations.copy()
    result["Run_ID"] = result["Run_ID"].astype(str)
    indexed = trial.set_index("Run_ID")
    missing = set(result["Run_ID"]) - set(indexed.index)
    if missing:
        raise ValueError(
            f"Trial fits are missing Run_ID values: {sorted(missing)}"
        )
    for field in PHASE_FIT_SQL_FIELDS:
        result[field] = result["Run_ID"].map(indexed[field])
    return result


def plot_nbins_phase_fits(fit_results, run_id: str | None = None):
    """Plot the histogram fits for one run across all trial bin counts."""

    import matplotlib.pyplot as plt

    run_ids = fit_results["Run_ID"].astype(str).unique()
    if len(run_ids) == 0:
        raise ValueError("fit_results is empty")
    run_id = str(run_ids[0] if run_id is None else run_id)
    selected = fit_results.loc[
        fit_results["Run_ID"].astype(str) == run_id
    ].sort_values("voxel_nbins")
    if selected.empty:
        raise ValueError(f"Run_ID was not found in fit_results: {run_id}")
    columns = min(3, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.0 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for axis, (_, fit) in zip(flat_axes, selected.iterrows()):
        density = np.asarray(fit["density_axis"], dtype=float)
        observed = np.asarray(fit["observed_counts"], dtype=float)
        axis.step(density, observed, where="mid", color="black", label="data")
        axis.plot(density, fit["model_counts"], label="mixture")
        axis.plot(density, fit["gas_counts"], linestyle=":", label="gas")
        axis.plot(density, fit["liquid_counts"], linestyle="--", label="liquid")
        axis.plot(
            density,
            fit["interface_counts"],
            linestyle="-.",
            label="interface",
        )
        axis.set(
            title=(
                f"nbins={int(fit['voxel_nbins'])}, "
                rf"$\rho_l$={fit['rho_liquid']:.4g}"
            ),
            xlabel="Voxel density",
            ylabel="Mean voxel count",
        )
        axis.grid(alpha=0.25)
    for axis in flat_axes[len(selected):]:
        axis.set_visible(False)
    flat_axes[0].legend(fontsize="small")
    figure.suptitle(f"Voxel-mixture sensitivity: {run_id}")
    return figure, axes


def plot_nbins_seitz(
    cavitations,
    fit_results,
    database=None,
    *,
    eos_states=None,
    eos_n_cells: int = 45,
    eos_filters: Mapping[str, Any] | None = None,
    eos_columns: Mapping[str, str] | None = None,
    temperature_atol: float = 1e-8,
    physical_scale=None,
    energy_unit: str = "eV",
    density_unit: str = "mol/L",
    figure_size=(10, 6),
    dpi: int = 120,
):
    """Overlay Seitz curves produced by each trial voxel-bin count."""

    import matplotlib.pyplot as plt
    import pandas as pd

    if eos_states is None:
        if database is None:
            raise ValueError("Pass database or an explicit eos_states table")
        eos_states = query_seitz_eos_states(
            database,
            n_cells=eos_n_cells,
            **dict(eos_filters or {}),
        )
    results = []
    for nbins in sorted(fit_results["voxel_nbins"].unique()):
        trial_rows = cavitations_with_nbins_fit(cavitations, fit_results, nbins)
        trial = calculate_cavitation_seitz(
            trial_rows,
            eos_states,
            eos_columns=eos_columns,
            temperature_atol=temperature_atol,
        )
        trial.insert(1, "voxel_nbins", int(nbins))
        results.append(trial)
    combined = pd.concat(results, ignore_index=True)

    x_column = "rho_liquid"
    x_uncertainty_column = "rho_liquid_uncertainty"
    y_column = "Q"
    y_uncertainty_column = "Q_uncertainty"
    x_label = r"$\rho_\mathrm{liquid}$"
    y_label = "Seitz Q"
    if physical_scale is not None:
        x_column = f"rho_liquid_{density_unit}"
        x_uncertainty_column = f"rho_liquid_uncertainty_{density_unit}"
        y_column = f"Q_{energy_unit}"
        y_uncertainty_column = f"Q_uncertainty_{energy_unit}"
        combined[x_column] = physical_scale.number_density(
            combined["rho_liquid"].to_numpy(dtype=float), unit=density_unit
        )
        combined[x_uncertainty_column] = (
            physical_scale.number_density_uncertainty(
                combined["rho_liquid"].to_numpy(dtype=float),
                combined["rho_liquid_uncertainty"].to_numpy(dtype=float),
                unit=density_unit,
            )
        )
        combined[y_column] = physical_scale.energy(
            combined["Q"].to_numpy(dtype=float), unit=energy_unit
        )
        combined[y_uncertainty_column] = physical_scale.energy_uncertainty(
            combined["Q"].to_numpy(dtype=float),
            combined["Q_uncertainty"].to_numpy(dtype=float),
            unit=energy_unit,
        )
        x_label = rf"$\rho_\mathrm{{liquid}}$ ({density_unit})"
        y_label = f"Seitz Q ({energy_unit})"

    figure, axis = plt.subplots(
        figsize=figure_size, dpi=int(dpi), constrained_layout=True
    )
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for index, (nbins, group) in enumerate(
        combined.groupby("voxel_nbins", sort=True)
    ):
        for temperature, temperature_group in group.groupby("Therm_kT", sort=True):
            temperature_group = temperature_group.sort_values("rho_liquid")
            axis.errorbar(
                temperature_group[x_column],
                temperature_group[y_column],
                xerr=temperature_group[x_uncertainty_column],
                yerr=temperature_group[y_uncertainty_column],
                marker=markers[index % len(markers)],
                linewidth=1.2,
                capsize=2,
                label=f"nbins={int(nbins)}, kT={temperature:g}",
            )
    axis.set(
        xlabel=x_label,
        ylabel=y_label,
        title="Seitz sensitivity to voxel nbins",
    )
    axis.grid(alpha=0.3)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    return figure, axis, combined
