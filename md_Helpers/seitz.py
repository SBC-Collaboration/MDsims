"""Seitz-threshold calculations and plots for V4 cavitation results.

The implementation preserves the intensive V3 expression

    Q = Nc * [(uc - u0) + P0 * (1/rho_c - 1/rho_0)]

and its uncertainty convention.  ``uc`` and ``u0`` are potential energies per
particle, ``rho_c`` is the mean density after applying the cavitation mask,
and ``rho_0`` is the fitted liquid density after cavitation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_EOS_COLUMNS = {
    "temperature": "Therm_kT",
    "density": "Density_End",
    "pressure": "Pressure_Mean",
    "pressure_uncertainty": "Pressure_SEM",
    "energy": "PE_Per_Particle_Mean",
    "energy_uncertainty": "PE_Per_Particle_SEM",
}


def seitz_threshold(nc, uc, u0, p0, rho_c, rho_0):
    """Return the intensive-form Seitz threshold used in V3."""

    nc = np.asarray(nc, dtype=float)
    uc = np.asarray(uc, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    rho_c = np.asarray(rho_c, dtype=float)
    rho_0 = np.asarray(rho_0, dtype=float)
    if np.any(nc <= 0):
        raise ValueError("nc must be positive")
    if np.any(rho_c <= 0) or np.any(rho_0 <= 0):
        raise ValueError("rho_c and rho_0 must be positive")
    return nc * ((uc - u0) + p0 * (1.0 / rho_c - 1.0 / rho_0))


def seitz_threshold_uncertainty(
    nc,
    p0,
    rho_c,
    rho_0,
    uc_uncertainty=0.0,
    u0_uncertainty=0.0,
    p0_uncertainty=0.0,
    rho_0_uncertainty=0.0,
):
    """Propagate the independent uncertainties using the V3 convention.

    ``Nc`` and ``rho_c`` are treated as exact.  The returned dictionary gives
    the absolute contribution from each input as well as their quadrature sum.
    """

    nc = float(nc)
    p0 = float(p0)
    rho_c = float(rho_c)
    rho_0 = float(rho_0)
    delta_v = 1.0 / rho_c - 1.0 / rho_0
    components = {
        "uc": abs(nc) * float(uc_uncertainty),
        "u0": abs(nc) * float(u0_uncertainty),
        "P0": abs(nc * delta_v) * float(p0_uncertainty),
        "rho_0": abs(nc * p0 / rho_0**2) * float(rho_0_uncertainty),
    }
    finite = [value for value in components.values() if np.isfinite(value)]
    components["total"] = (
        float(np.sqrt(np.sum(np.square(finite)))) if finite else np.nan
    )
    return components


def _numeric_table(table, required_columns, label):
    import pandas as pd

    result = pd.DataFrame(table).copy()
    missing = [column for column in required_columns if column not in result]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")
    for column in required_columns:
        if column != "Run_ID":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _local_linear_estimate(density, values, uncertainties, target_density):
    """Interpolate a mean/SEM and return the local piecewise-linear slope."""

    density = np.asarray(density, dtype=float)
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    order = np.argsort(density)
    density = density[order]
    values = values[order]
    uncertainties = uncertainties[order]
    target_density = float(target_density)

    if len(density) == 0:
        raise ValueError("No finite EOS points are available")
    if target_density < density[0] or target_density > density[-1]:
        raise ValueError(
            f"density {target_density:g} is outside the EOS range "
            f"[{density[0]:g}, {density[-1]:g}]"
        )

    value = float(np.interp(target_density, density, values))
    finite_uncertainty = np.isfinite(uncertainties)
    mean_uncertainty = (
        float(
            np.interp(
                target_density,
                density[finite_uncertainty],
                uncertainties[finite_uncertainty],
            )
        )
        if np.any(finite_uncertainty)
        else np.nan
    )

    if len(density) < 2:
        slope = np.nan
    elif target_density >= density[-1]:
        left, right = len(density) - 2, len(density) - 1
        slope = (values[right] - values[left]) / (
            density[right] - density[left]
        )
    else:
        right = int(np.searchsorted(density, target_density, side="right"))
        left = max(0, right - 1)
        slope = (values[right] - values[left]) / (
            density[right] - density[left]
        )
    return value, float(slope), mean_uncertainty


def _eos_estimate(
    eos_states,
    temperature,
    target_density,
    target_density_uncertainty,
    value_column,
    uncertainty_column,
    temperature_column,
    density_column,
    temperature_atol,
):
    same_temperature = eos_states.loc[
        np.isclose(
            eos_states[temperature_column].to_numpy(dtype=float),
            float(temperature),
            atol=float(temperature_atol),
            rtol=0.0,
        )
    ].copy()
    if same_temperature.empty:
        available = sorted(eos_states[temperature_column].dropna().unique())
        raise ValueError(
            f"No EOS states match kT={float(temperature):g}; "
            f"available temperatures are {available}"
        )

    grouped = (
        same_temperature.groupby(density_column, as_index=False)
        .agg(
            value=(value_column, "mean"),
            uncertainty=(uncertainty_column, "mean"),
        )
        .sort_values(density_column)
    )
    value, slope, mean_uncertainty = _local_linear_estimate(
        grouped[density_column],
        grouped["value"],
        grouped["uncertainty"],
        target_density,
    )
    density_uncertainty = (
        abs(slope) * float(target_density_uncertainty)
        if np.isfinite(slope) and np.isfinite(target_density_uncertainty)
        else np.nan
    )
    parts = [
        part
        for part in (mean_uncertainty, density_uncertainty)
        if np.isfinite(part)
    ]
    uncertainty = (
        float(np.sqrt(np.sum(np.square(parts)))) if parts else np.nan
    )
    return {
        "value": value,
        "slope": slope,
        "mean_uncertainty": mean_uncertainty,
        "density_uncertainty": density_uncertainty,
        "uncertainty": uncertainty,
    }


def query_seitz_eos_states(
    database,
    n_cells: int = 45,
    **filters: Any,
):
    """Query the default homogeneous V4 thermalization EOS states."""

    from .database import thermalization_dataframe

    query = {
        "N_Cells": int(n_cells),
        "Phase_Separation_Status": "Not_Separated",
        # The EOS cell defines fixed-density data by excluding rescaled clones.
        "Clone_Run_ID": None,
        **filters,
    }
    database.initialize()
    return thermalization_dataframe(database, **query)


def calculate_cavitation_seitz(
    cavitations,
    eos_states,
    *,
    eos_columns: Mapping[str, str] | None = None,
    eos_method: str = "linear",
    temperature_atol: float = 1e-8,
):
    """Calculate V3-style Seitz terms for selected V4 cavitation rows.

    The returned DataFrame contains the inputs, interpolated EOS quantities,
    local EOS slopes, uncertainty components, ``Q``, and ``Q_uncertainty``.
    A custom EOS table can be used by mapping its column names with
    ``eos_columns``.
    """

    import pandas as pd

    if str(eos_method).lower() != "linear":
        raise ValueError("eos_method currently supports only 'linear'")
    columns = {**DEFAULT_EOS_COLUMNS, **dict(eos_columns or {})}
    unknown = set(columns) - set(DEFAULT_EOS_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown eos_columns keys: {sorted(unknown)}")

    cavitation_columns = [
        "Run_ID",
        "N_Cells",
        "Therm_kT",
        "Nsteps",
        "Initial_Density",
        "BoxLength",
        "rho_liquid",
        "rho_liquid_unc",
        "PE_Per_Particle_Mean",
        "PE_Per_Particle_SEM",
    ]
    cavitations = _numeric_table(
        cavitations, cavitation_columns, "cavitations"
    )
    eos_required = list(dict.fromkeys(columns.values()))
    eos_states = _numeric_table(eos_states, eos_required, "eos_states")
    eos_states = eos_states.dropna(
        subset=[
            columns["temperature"],
            columns["density"],
            columns["pressure"],
            columns["energy"],
        ]
    )
    if eos_states.empty:
        raise ValueError("No finite EOS states are available")
    if cavitations.empty:
        return pd.DataFrame()

    finite_required = [
        "N_Cells",
        "Therm_kT",
        "Nsteps",
        "Initial_Density",
        "BoxLength",
        "rho_liquid",
        "PE_Per_Particle_Mean",
    ]
    invalid = ~np.all(
        np.isfinite(cavitations[finite_required].to_numpy(dtype=float)), axis=1
    )
    if invalid.any():
        run_ids = cavitations.loc[invalid, "Run_ID"].astype(str).tolist()
        raise ValueError(
            "Cavitation rows have missing Seitz inputs: " + ", ".join(run_ids)
        )

    records = []
    for _, row in cavitations.iterrows():
        rho_c = float(row["Initial_Density"])
        volume = float(row["BoxLength"]) ** 3
        nc = int(round(rho_c * volume))
        rho_0 = float(row["rho_liquid"])
        rho_0_uncertainty = float(row["rho_liquid_unc"])
        uc = float(row["PE_Per_Particle_Mean"])
        uc_uncertainty = float(row["PE_Per_Particle_SEM"])
        if not np.isfinite(rho_0_uncertainty) or rho_0_uncertainty < 0:
            rho_0_uncertainty = 0.0
        if not np.isfinite(uc_uncertainty) or uc_uncertainty < 0:
            uc_uncertainty = 0.0

        common = dict(
            eos_states=eos_states,
            temperature=float(row["Therm_kT"]),
            target_density=rho_0,
            target_density_uncertainty=rho_0_uncertainty,
            temperature_column=columns["temperature"],
            density_column=columns["density"],
            temperature_atol=temperature_atol,
        )
        try:
            u0 = _eos_estimate(
                value_column=columns["energy"],
                uncertainty_column=columns["energy_uncertainty"],
                **common,
            )
            p0 = _eos_estimate(
                value_column=columns["pressure"],
                uncertainty_column=columns["pressure_uncertainty"],
                **common,
            )
        except ValueError as error:
            raise ValueError(
                f"Could not evaluate EOS for cavitation Run_ID "
                f"{row['Run_ID']}: {error}"
            ) from error
        q = float(seitz_threshold(nc, uc, u0["value"], p0["value"], rho_c, rho_0))
        q_uncertainty = seitz_threshold_uncertainty(
            nc=nc,
            p0=p0["value"],
            rho_c=rho_c,
            rho_0=rho_0,
            uc_uncertainty=uc_uncertainty,
            u0_uncertainty=u0["uncertainty"],
            p0_uncertainty=p0["uncertainty"],
            rho_0_uncertainty=rho_0_uncertainty,
        )
        records.append({
            "Run_ID": str(row["Run_ID"]),
            "N_Cells": int(row["N_Cells"]),
            "Therm_kT": float(row["Therm_kT"]),
            "Nsteps": int(row["Nsteps"]),
            "Nc": nc,
            "V": volume,
            "rho_c": rho_c,
            "uc": uc,
            "uc_uncertainty": uc_uncertainty,
            "rho_liquid": rho_0,
            "rho_liquid_uncertainty": rho_0_uncertainty,
            "u_EOS": u0["value"],
            "u_EOS_uncertainty": u0["uncertainty"],
            "u_EOS_mean_uncertainty": u0["mean_uncertainty"],
            "u_EOS_density_uncertainty": u0["density_uncertainty"],
            "u_EOS_density_slope": u0["slope"],
            "P_EOS": p0["value"],
            "P_EOS_uncertainty": p0["uncertainty"],
            "P_EOS_mean_uncertainty": p0["mean_uncertainty"],
            "P_EOS_density_uncertainty": p0["density_uncertainty"],
            "P_EOS_density_slope": p0["slope"],
            "Q": q,
            "Q_uncertainty": q_uncertainty["total"],
            "Q_uncertainty_uc_component": q_uncertainty["uc"],
            "Q_uncertainty_u_EOS_component": q_uncertainty["u0"],
            "Q_uncertainty_P_EOS_component": q_uncertainty["P0"],
            "Q_uncertainty_rho_liquid_component": q_uncertainty["rho_0"],
        })
    return pd.DataFrame.from_records(records).sort_values(
        ["Therm_kT", "N_Cells", "Nsteps", "rho_liquid", "Run_ID"]
    ).reset_index(drop=True)


def plot_cavitation_seitz(
    cavitations,
    database=None,
    *,
    eos_states=None,
    eos_n_cells: int = 45,
    eos_filters: Mapping[str, Any] | None = None,
    eos_columns: Mapping[str, str] | None = None,
    eos_method: str = "linear",
    temperature_atol: float = 1e-8,
    figure_size=(10, 6),
    dpi: int = 120,
    title: str = "Seitz Q vs liquid density",
):
    """Calculate and plot Seitz ``Q`` for a cavitation DataFrame.

    If ``eos_states`` is omitted, homogeneous thermalization states are queried
    from ``database`` using ``eos_n_cells`` (45 by default) and any additional
    ``eos_filters``.  Return ``(figure, axis, results)``.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if eos_states is None:
        if database is None:
            raise ValueError("Pass database or an explicit eos_states table")
        eos_states = query_seitz_eos_states(
            database,
            n_cells=eos_n_cells,
            **dict(eos_filters or {}),
        )
    results = calculate_cavitation_seitz(
        cavitations,
        eos_states,
        eos_columns=eos_columns,
        eos_method=eos_method,
        temperature_atol=temperature_atol,
    )
    if results.empty:
        raise ValueError("No cavitation rows were provided")

    figure, axis = plt.subplots(
        figsize=figure_size, dpi=int(dpi), constrained_layout=True
    )
    n_cells_values = sorted(results["N_Cells"].unique())
    temperatures = sorted(results["Therm_kT"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["*", "P", "^", "D", "v", "s", "X", "o"]
    cell_colors = {
        value: colors[index % len(colors)]
        for index, value in enumerate(n_cells_values)
    }
    temperature_markers = {
        value: markers[index % len(markers)]
        for index, value in enumerate(temperatures)
    }

    for (temperature, n_cells, nsteps), group in results.groupby(
        ["Therm_kT", "N_Cells", "Nsteps"], sort=True
    ):
        group = group.sort_values("rho_liquid")
        axis.errorbar(
            group["rho_liquid"],
            group["Q"],
            xerr=group["rho_liquid_uncertainty"],
            yerr=group["Q_uncertainty"],
            color=cell_colors[n_cells],
            marker=temperature_markers[temperature],
            linestyle="-",
            linewidth=1.5,
            markersize=7,
            capsize=3,
            elinewidth=1.1,
            capthick=1.1,
        )

    cell_handles = [
        Line2D(
            [0], [0], color=cell_colors[value], marker="o", linestyle="-",
            label=f"Ncells={int(value)}",
        )
        for value in n_cells_values
    ]
    temperature_handles = [
        Line2D(
            [0], [0], color="black", marker=temperature_markers[value],
            linestyle="None", markersize=7, label=f"kT={value:g}",
        )
        for value in temperatures
    ]
    cells_legend = axis.legend(
        handles=cell_handles,
        title="Ncells (color)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    axis.add_artist(cells_legend)
    axis.legend(
        handles=temperature_handles,
        title="Temperature (marker)",
        bbox_to_anchor=(1.02, 0.55),
        loc="upper left",
    )
    axis.set_xlabel(r"$\rho_\mathrm{liquid}$")
    axis.set_ylabel("Q")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    return figure, axis, results
