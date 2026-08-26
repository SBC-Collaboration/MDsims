"""Batch liquid/gas density fits for phase-separated thermalized states."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .master_csv import summarize_thermalization_log
from .paths import THERMALIZED_STATES_V3_ROOT
from .spatial import nbins_for_ncells
from .voxel_fit import fit_trajectory_tail_voxel_histogram


DEFAULT_OUTPUT_COLUMNS = [
    "status",
    "n_fcc_cells",
    "target_rho",
    "actual_rho",
    "kT",
    "nsteps",
    "seed",
    "phase_name",
    "phase_separated",
    "phase_sep_low_density_fraction",
    "state_path",
    "log_path",
    "voxel_nbins",
    "frames_available",
    "frames_fitted",
    "frame_indices",
    "timesteps",
    "gas_density",
    "gas_density_se",
    "gas_density_ci95_low",
    "gas_density_ci95_high",
    "liquid_density",
    "liquid_density_se",
    "liquid_density_ci95_low",
    "liquid_density_ci95_high",
    "liquid_to_gas_density_ratio",
    "liquid_to_gas_density_ratio_se",
    "liquid_to_gas_density_ratio_ci95_low",
    "liquid_to_gas_density_ratio_ci95_high",
    "gas_weight",
    "liquid_weight",
    "interface_weight",
    "fit_success",
    "fit_message",
    "uncertainty_method",
    "AIC",
    "BIC",
    "error",
]


def _clean_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _state_path_from_summary(summary, log_path):
    for key in ("state_path", "final_state_path"):
        value = summary.get(key)
        if value is not None and str(value).strip():
            return Path(value)
    return Path(log_path).with_name(
        Path(log_path).name.replace("_log.hdf5", ".gsd")
    )


def _read_voxel_phase_attrs(log_path):
    """Read method-specific voxel attributes without importing GSD."""

    import h5py

    group_path = "metadata/classification/phase_separation/voxel"
    with h5py.File(Path(log_path), mode="r") as hdf:
        if group_path not in hdf:
            return {}
        return {
            key: value.item() if hasattr(value, "item") else value
            for key, value in hdf[group_path].attrs.items()
        }


def _path_component_value(path, prefix, cast=float):
    for component in Path(path).parts:
        if component.startswith(prefix):
            try:
                return cast(component[len(prefix):])
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _metadata_or_path(summary, key, log_path, prefix, cast=float):
    value = summary.get(key, np.nan)
    try:
        if value is not None and not pd.isna(value):
            return cast(value)
    except (TypeError, ValueError):
        pass
    return _path_component_value(log_path, prefix, cast=cast)


def discover_phase_separated_thermalized_states(
    root=THERMALIZED_STATES_V3_ROOT,
    n_last=100,
):
    """Return metadata for every thermalization log marked phase separated."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Thermalization root does not exist: {root}")

    rows = []
    for log_path in sorted(root.rglob("*_log.hdf5")):
        summary = summarize_thermalization_log(log_path, n_last=n_last)
        try:
            voxel_attrs = _read_voxel_phase_attrs(log_path)
        except (OSError, KeyError, ValueError):
            voxel_attrs = {}
        phase_separated = _clean_bool(
            voxel_attrs.get("phase_separated", summary.get("phase_separated"))
        )
        if phase_separated is not True:
            continue

        state_path = _state_path_from_summary(summary, log_path)
        rows.append({
            "n_fcc_cells": _metadata_or_path(
                summary,
                "n_fcc_cells",
                log_path,
                "n_cells_",
                cast=int,
            ),
            "target_rho": _metadata_or_path(
                summary,
                "target_rho",
                log_path,
                "rho_",
            ),
            "actual_rho": summary.get("actual_rho", np.nan),
            "kT": _metadata_or_path(summary, "kT", log_path, "kT_"),
            "nsteps": _metadata_or_path(
                summary,
                "nsteps",
                log_path,
                "nsteps_",
                cast=int,
            ),
            "seed": _metadata_or_path(
                summary,
                "seed",
                log_path,
                "seed_",
                cast=int,
            ),
            "phase_name": summary.get("phase_name", "randomization"),
            "phase_separated": True,
            "phase_sep_low_density_fraction": voxel_attrs.get(
                "low_density_fraction",
                summary.get(
                    "low_density_fraction",
                    summary.get("phase_sep_low_density_fraction", np.nan),
                ),
            ),
            "state_path": str(state_path),
            "log_path": str(log_path),
            "state_exists": state_path.exists(),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values([
            "n_fcc_cells",
            "kT",
            "target_rho",
            "seed",
            "state_path",
        ]).reset_index(drop=True)
    return table


def _positive_delta_summary(value, gradient, covariance, confidence_z=1.96):
    """Propagate covariance to a positive scalar and a log-scale interval."""

    value = float(value)
    gradient = np.asarray(gradient, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    variance = float(gradient @ covariance @ gradient)

    if not np.isfinite(variance) or variance < 0.0 or value <= 0.0:
        return {
            "value": value,
            "se": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    standard_error = float(np.sqrt(variance))
    log_standard_error = standard_error / value
    return {
        "value": value,
        "se": standard_error,
        "ci95_low": float(
            value * np.exp(-float(confidence_z) * log_standard_error)
        ),
        "ci95_high": float(
            value * np.exp(float(confidence_z) * log_standard_error)
        ),
    }


def density_uncertainties_from_fit(fit, confidence_z=1.96):
    """Propagate the full transformed-parameter covariance to three outputs.

    Returns uncertainty summaries for fitted gas density, fitted liquid
    density, and their liquid/gas ratio. The parameterization matches
    :func:`md_Helpers.voxel_fit.fit_voxel_count_mixture`.
    """

    covariance = np.asarray(fit["parameter_covariance"], dtype=float)
    if covariance.shape != (5, 5):
        raise ValueError(
            f"Expected a 5x5 parameter covariance, found {covariance.shape}"
        )

    voxel_volume = float(fit["voxel_volume"])
    gas_mean = float(fit["gas_mean_count"])
    liquid_mean = float(fit["liquid_mean_count"])
    mean_gap = liquid_mean - gas_mean

    gas_density = gas_mean / voxel_volume
    liquid_density = liquid_mean / voxel_volume
    density_ratio = liquid_mean / gas_mean

    gas_gradient = np.array([
        gas_mean / voxel_volume,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
    liquid_gradient = np.array([
        gas_mean / voxel_volume,
        mean_gap / voxel_volume,
        0.0,
        0.0,
        0.0,
    ])
    ratio_gap = mean_gap / gas_mean
    ratio_gradient = np.array([
        -ratio_gap,
        ratio_gap,
        0.0,
        0.0,
        0.0,
    ])

    return {
        "gas_density": _positive_delta_summary(
            gas_density,
            gas_gradient,
            covariance,
            confidence_z=confidence_z,
        ),
        "liquid_density": _positive_delta_summary(
            liquid_density,
            liquid_gradient,
            covariance,
            confidence_z=confidence_z,
        ),
        "liquid_to_gas_density_ratio": _positive_delta_summary(
            density_ratio,
            ratio_gradient,
            covariance,
            confidence_z=confidence_z,
        ),
    }


def _flatten_uncertainties(uncertainties):
    row = {}
    for name, summary in uncertainties.items():
        row[name] = summary["value"]
        row[f"{name}_se"] = summary["se"]
        row[f"{name}_ci95_low"] = summary["ci95_low"]
        row[f"{name}_ci95_high"] = summary["ci95_high"]
    return row


def _analysis_key(
    state_path,
    voxel_nbins,
    nframes,
    skip,
    tail_fraction,
    interface_void_fraction,
    interface_points,
):
    return "|".join([
        str(state_path),
        f"nbins={int(voxel_nbins)}",
        f"nframes={int(nframes)}",
        f"skip={int(skip)}",
        f"tail={float(tail_fraction):.8g}",
        f"interface={float(interface_void_fraction):.8g}",
        f"points={int(interface_points)}",
    ])


def _ordered_columns(table):
    first = [column for column in DEFAULT_OUTPUT_COLUMNS if column in table]
    remaining = [column for column in table if column not in first]
    return table[[*first, *remaining]]


def analyze_phase_separated_thermalized_states(
    root=THERMALIZED_STATES_V3_ROOT,
    states=None,
    output_path=None,
    voxel_nbins=None,
    nframes=5,
    skip=5,
    tail_fraction=0.5,
    interface_void_fraction=0.5,
    interface_points=40,
    max_iterations=500,
    resume=True,
):
    """Fit every phase-separated thermalized state and optionally save CSV.

    The fitting call and its defaults match the cavitation pooled-tail voxel
    fit. Legacy thermalized ``randomization.gsd`` files commonly contain only
    one frame; ``frames_fitted`` records the actual number used.
    """

    if states is None:
        states = discover_phase_separated_thermalized_states(root=root)
    else:
        states = pd.DataFrame(states).copy()

    output_path = Path(output_path) if output_path is not None else None
    existing = pd.DataFrame()
    if resume and output_path is not None and output_path.exists():
        existing = pd.read_csv(output_path)

    completed_by_key = {}
    if not existing.empty and "analysis_key" in existing:
        for _, old_row in existing.iterrows():
            if old_row.get("status") == "completed":
                completed_by_key[str(old_row["analysis_key"])] = old_row.to_dict()

    rows = []
    total = len(states)
    for position, (_, state) in enumerate(states.iterrows(), start=1):
        state_path = Path(state["state_path"])
        n_fcc_cells = int(state["n_fcc_cells"])
        selected_nbins = (
            int(voxel_nbins)
            if voxel_nbins is not None
            else nbins_for_ncells(n_fcc_cells)
        )
        key = _analysis_key(
            state_path,
            selected_nbins,
            nframes,
            skip,
            tail_fraction,
            interface_void_fraction,
            interface_points,
        )

        if key in completed_by_key:
            rows.append(completed_by_key[key])
            print(f"[{position}/{total}] cached: {state_path}")
            continue

        row = state.to_dict()
        row.update({
            "analysis_key": key,
            "status": "fit_failed",
            "voxel_nbins": selected_nbins,
            "nframes_requested": int(nframes),
            "frame_skip": int(skip),
            "tail_fraction": float(tail_fraction),
            "interface_void_fraction": float(interface_void_fraction),
            "interface_points": int(interface_points),
            "error": "",
        })

        print(f"[{position}/{total}] fitting: {state_path}")
        try:
            if not state_path.exists():
                raise FileNotFoundError(state_path)

            import gsd.hoomd

            with gsd.hoomd.open(name=str(state_path), mode="r") as trajectory:
                row["frames_available"] = len(trajectory)

            fit = fit_trajectory_tail_voxel_histogram(
                trajectory_path=state_path,
                voxel_nbins=selected_nbins,
                nframes=nframes,
                skip=skip,
                tail_fraction=tail_fraction,
                interface_void_fraction=interface_void_fraction,
                interface_points=interface_points,
                max_iterations=max_iterations,
            )
            uncertainty = density_uncertainties_from_fit(fit)
            row.update(_flatten_uncertainties(uncertainty))
            row.update({
                "status": "completed" if fit["success"] else "fit_failed",
                "frames_fitted": int(fit["nframes"]),
                "frame_indices": json.dumps(fit["frame_indices"]),
                "timesteps": json.dumps(fit["timesteps"]),
                "gas_weight": float(fit["gas_weight"]),
                "liquid_weight": float(fit["liquid_weight"]),
                "interface_weight": float(fit["interface_weight"]),
                "fit_success": bool(fit["success"]),
                "fit_message": str(fit["message"]),
                "uncertainty_method": str(fit["uncertainty_method"]),
                "AIC": float(fit["AIC"]),
                "BIC": float(fit["BIC"]),
            })
        except Exception as error:
            row["error"] = repr(error)

        rows.append(row)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _ordered_columns(pd.DataFrame(rows)).to_csv(output_path, index=False)

    result = _ordered_columns(pd.DataFrame(rows))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def _as_selection(values):
    if values is None:
        return None
    if np.isscalar(values):
        return [values]
    return list(values)


def select_phase_density_results(
    table,
    ncells=None,
    temperatures=None,
    target_densities=None,
    seeds=None,
    completed_only=True,
    atol=1e-10,
):
    """Filter a batch result table using scalar or iterable selections."""

    selected = pd.DataFrame(table).copy()
    if completed_only and "status" in selected:
        selected = selected[selected["status"].eq("completed")]

    exact_filters = {
        "n_fcc_cells": _as_selection(ncells),
        "seed": _as_selection(seeds),
    }
    for column, values in exact_filters.items():
        if values is not None:
            selected = selected[selected[column].isin(values)]

    close_filters = {
        "kT": _as_selection(temperatures),
        "target_rho": _as_selection(target_densities),
    }
    for column, values in close_filters.items():
        if values is not None:
            numeric = pd.to_numeric(selected[column], errors="coerce")
            mask = np.logical_or.reduce([
                np.isclose(numeric, float(value), atol=atol, rtol=0.0)
                for value in values
            ])
            selected = selected[mask]

    sort_columns = [
        column
        for column in ["n_fcc_cells", "kT", "target_rho", "seed"]
        if column in selected
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def plot_density_ratio_with_uncertainty(table, ax=None, uncertainty="se"):
    """Plot density ratio with SE or 95% CI bars, using kT for one ncell."""

    import matplotlib.pyplot as plt

    selected = pd.DataFrame(table).copy()
    selected = selected[selected["status"].eq("completed")]
    if selected.empty:
        raise ValueError("No completed fits are available to plot")
    if uncertainty not in {"se", "ci95"}:
        raise ValueError("uncertainty must be 'se' or 'ci95'")

    unique_ncells = selected["n_fcc_cells"].dropna().unique()
    if len(unique_ncells) == 1:
        x_column = "kT"
        xlabel = "kT"
        group_columns = ["target_rho"]
    else:
        x_column = "kT"
        xlabel = "kT"
        group_columns = ["n_fcc_cells", "target_rho"]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for group_key, group in selected.groupby(group_columns, dropna=False):
        group = group.sort_values(x_column)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        if len(unique_ncells) == 1:
            label = f"rho={group_key[0]:g}"
        else:
            label = f"ncell={group_key[0]:g}, rho={group_key[1]:g}"
        ratio = group["liquid_to_gas_density_ratio"]
        if uncertainty == "se":
            error = group["liquid_to_gas_density_ratio_se"]
        else:
            error = np.vstack([
                ratio - group["liquid_to_gas_density_ratio_ci95_low"],
                group["liquid_to_gas_density_ratio_ci95_high"] - ratio,
            ])
        ax.errorbar(
            group[x_column],
            ratio,
            yerr=error,
            marker="o",
            linestyle="-",
            capsize=3,
            label=label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Fitted density ratio, $\rho_l/\rho_g$")
    if len(unique_ncells) == 1:
        ax.set_title(f"Phase-separated thermalized states: ncell={unique_ncells[0]:g}")
    else:
        ax.set_title("Phase-separated thermalized states")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.figure.text(
        0.99,
        0.01,
        "Error bars: 1 SE" if uncertainty == "se" else "Error bars: 95% CI",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    return ax
