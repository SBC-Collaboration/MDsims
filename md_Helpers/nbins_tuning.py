"""In-memory sensitivity checks for the cavitation voxel-bin choice."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .paths import ProjectPaths
from .seitz import calculate_cavitation_seitz, query_seitz_eos_states
from .voxel_fit import (
    PHASE_FIT_SQL_FIELDS,
    _finite_difference_hessian,
    _standard_uncertainty,
    fit_trajectory_voxel_gaussian,
    fit_trajectory_voxel_mixture,
)


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


def fit_liquid_nbins(
    state,
    nbins_values: Iterable[int],
    *,
    project_paths: ProjectPaths | None = None,
    num_frames: int = 5,
    frame_indices=None,
    **fit_options: Any,
):
    """Fit one Gaussian to a liquid state for each requested voxel resolution.

    ``state`` may be a row-like mapping/Series or a one-row DataFrame with
    ``Run_ID``, ``N_Cells``, and ``File_Location``.  The trajectory and database
    remain unchanged; all results are returned in memory.
    """

    import pandas as pd

    if hasattr(state, "columns"):
        if len(state) != 1:
            raise ValueError("state must contain exactly one liquid-state row")
        row = state.iloc[0]
    else:
        row = state
    required = {"Run_ID", "N_Cells", "File_Location"}
    missing = required - set(row.index if hasattr(row, "index") else row)
    if missing:
        raise ValueError(f"state is missing fields: {sorted(missing)}")

    bins = _validated_nbins(nbins_values)
    paths = project_paths or ProjectPaths()
    trajectory_path = _trajectory_path(row, paths)
    records = []
    for nbins in bins:
        fit = fit_trajectory_voxel_gaussian(
            trajectory_path,
            int(row["N_Cells"]),
            num_frames=num_frames,
            frame_indices=frame_indices,
            nbins=nbins,
            **fit_options,
        )
        records.append({"Run_ID": str(row["Run_ID"]), **fit})
    return pd.DataFrame.from_records(records).sort_values("voxel_nbins").reset_index(
        drop=True
    )


def plot_liquid_nbins_gaussians(fit_results):
    """Show each Gaussian fit with raw and standardized residuals."""

    import matplotlib.pyplot as plt

    selected = fit_results.sort_values("voxel_nbins")
    if selected.empty:
        raise ValueError("fit_results is empty")
    columns = min(3, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(
        3 * rows,
        columns,
        figsize=(5.2 * columns, 6.8 * rows),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [
                value for _ in range(rows) for value in (3, 1, 1)
            ]
        },
    )
    panel_axes = []
    for panel_index, (_, fit) in enumerate(selected.iterrows()):
        panel_row, column = divmod(panel_index, columns)
        axis = axes[3 * panel_row, column]
        residual_axis = axes[3 * panel_row + 1, column]
        standardized_axis = axes[3 * panel_row + 2, column]
        density = np.asarray(fit["density_axis"], dtype=float)
        observed = np.asarray(fit["observed_counts"], dtype=float)
        fitted = np.asarray(fit["gaussian_counts"], dtype=float)
        residual = observed - fitted
        frames_used = int(fit["frames_used"])
        voxels_per_frame = int(fit["n_voxels_per_frame"])
        fitted_probability = fitted / voxels_per_frame
        residual_variance = (
            voxels_per_frame
            * fitted_probability
            * (1.0 - fitted_probability)
            / frames_used
        )
        expected_total = fitted * frames_used
        standardized = np.full_like(residual, np.nan, dtype=float)
        valid = (
            (expected_total >= 5.0)
            & np.isfinite(residual_variance)
            & (residual_variance > 0.0)
        )
        standardized[valid] = residual[valid] / np.sqrt(
            residual_variance[valid]
        )
        axis.step(
            density,
            observed,
            where="mid",
            color="black",
            label="last-5-frame average",
        )
        axis.plot(density, fitted, color="tab:red", label="Gaussian")
        axis.set(
            title=(
                f"nbins={int(fit['voxel_nbins'])}, "
                rf"$\mu_\rho$={fit['gaussian_mean_density']:.4g}, "
                rf"$\sigma_\rho$={fit['gaussian_sigma_density']:.4g}"
            ),
            ylabel="Mean number of voxels",
        )
        axis.grid(alpha=0.25)
        residual_axis.step(
            density,
            residual,
            where="mid",
            color="tab:blue",
        )
        residual_axis.axhline(0.0, color="black", linewidth=0.8)
        residual_axis.set(
            ylabel="Data − fit\n(voxels)",
        )
        residual_axis.grid(alpha=0.25)
        standardized_axis.step(
            density,
            standardized,
            where="mid",
            color="tab:green",
        )
        standardized_axis.axhline(0.0, color="black", linewidth=0.8)
        standardized_axis.axhline(
            2.0,
            color="tab:orange",
            linestyle="--",
            linewidth=0.9,
        )
        standardized_axis.axhline(
            -2.0,
            color="tab:orange",
            linestyle="--",
            linewidth=0.9,
        )
        standardized_axis.set(
            xlabel="Voxel density",
            ylabel="Standardized\nresidual",
        )
        standardized_axis.grid(alpha=0.25)
        panel_axes.append((axis, residual_axis, standardized_axis))
    for panel_index in range(len(selected), rows * columns):
        panel_row, column = divmod(panel_index, columns)
        axes[3 * panel_row, column].set_visible(False)
        axes[3 * panel_row + 1, column].set_visible(False)
        axes[3 * panel_row + 2, column].set_visible(False)
    panel_axes[0][0].legend(fontsize="small")
    figure.suptitle(f"Liquid voxel Gaussian fits: {selected.iloc[0]['Run_ID']}")
    return figure, axes


def plot_liquid_gaussian_mean_vs_nbins(fit_results, *, density: bool = True):
    """Plot fitted Gaussian mean (and its uncertainty) against ``nbins``."""

    import matplotlib.pyplot as plt

    selected = fit_results.sort_values("voxel_nbins")
    if selected.empty:
        raise ValueError("fit_results is empty")
    mean_column = "gaussian_mean_density" if density else "gaussian_mean"
    uncertainty_column = f"{mean_column}_unc"
    y_label = "Gaussian mean voxel density" if density else "Gaussian mean occupancy"
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.errorbar(
        selected["voxel_nbins"],
        selected[mean_column],
        yerr=selected[uncertainty_column],
        marker="o",
        capsize=3,
    )
    axis.set(
        xlabel="nbins (per box dimension)",
        ylabel=y_label,
        title="Liquid Gaussian mean vs voxel resolution",
    )
    axis.grid(alpha=0.3)
    return figure, axis


def refit_skewed_phase_nbins(
    normal_fits,
    *,
    interface_points: int | None = None,
    max_iterations: int = 1000,
):
    """Replace the liquid Gaussian with a skew-normal in phase-mixture fits.

    The vapor remains Poisson and the interface remains the same average of
    vapor/liquid fractional convolutions used by the ordinary mixture model.
    The skew-normal is parameterized by its actual mean, scale, and shape, so
    shape=0 reduces exactly to the ordinary Gaussian liquid component.
    """

    import pandas as pd
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from scipy.stats import poisson, skewnorm

    if normal_fits.empty:
        raise ValueError("normal_fits is empty")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")

    records = []
    root_two_over_pi = np.sqrt(2.0 / np.pi)
    for _, normal_fit in normal_fits.sort_values("voxel_nbins").iterrows():
        count_axis = np.asarray(normal_fit["count_axis"], dtype=float)
        observed_average = np.asarray(normal_fit["observed_counts"], dtype=float)
        frames_used = int(normal_fit["frames_used"])
        observed = observed_average * frames_used
        nbins = int(normal_fit["voxel_nbins"])
        voxel_volume = float(normal_fit["voxel_volume"])
        box_volume = float(normal_fit["box_volume"])
        points = int(
            normal_fit["interface_points"]
            if interface_points is None
            else interface_points
        )
        if points <= 0:
            raise ValueError("interface_points must be positive")

        gas_guess = max(1e-3, float(normal_fit["rho_gas"]) * voxel_volume)
        liquid_mean_guess = max(
            gas_guess + 1e-2,
            float(normal_fit["rho_liquid"]) * voxel_volume,
        )
        liquid_component = np.asarray(normal_fit["liquid_counts"], dtype=float)
        if liquid_component.sum() > 0:
            scale_guess = np.sqrt(
                np.average(
                    (count_axis - liquid_mean_guess) ** 2,
                    weights=liquid_component,
                )
            )
        else:
            scale_guess = np.sqrt(liquid_mean_guess)
        scale_guess = max(0.1, float(scale_guess))
        weights_guess = np.array(
            [
                normal_fit["gas_weight"],
                normal_fit["liquid_weight"],
                normal_fit["interface_weight"],
            ],
            dtype=float,
        )
        weights_guess = np.clip(weights_guess, 1e-8, None)
        weights_guess /= weights_guess.sum()

        def unpack(parameters):
            gas_mean = np.exp(parameters[0])
            gap = np.exp(parameters[1])
            liquid_mean = gas_mean + gap
            liquid_scale = np.exp(parameters[2])
            logits = np.array([parameters[3], parameters[4], 0.0])
            weights = np.exp(logits - logsumexp(logits))
            alpha = parameters[5]
            return gas_mean, gap, liquid_mean, liquid_scale, weights, alpha

        def discrete_skew_normal(mean, scale, alpha):
            delta = alpha / np.sqrt(1.0 + alpha**2)
            location = mean - scale * delta * root_two_over_pi
            lower = count_axis - 0.5
            lower[count_axis == 0] = -np.inf
            return skewnorm.cdf(
                count_axis + 0.5,
                a=alpha,
                loc=location,
                scale=scale,
            ) - skewnorm.cdf(
                lower,
                a=alpha,
                loc=location,
                scale=scale,
            )

        def component_probabilities(gas_mean, liquid_mean, liquid_scale, alpha):
            gas = poisson.pmf(count_axis, gas_mean)
            liquid = discrete_skew_normal(liquid_mean, liquid_scale, alpha)
            interface = np.zeros_like(gas)
            fractions = (np.arange(points) + 0.5) / points
            for gas_fraction in fractions:
                gas_part = poisson.pmf(count_axis, gas_fraction * gas_mean)
                liquid_fraction = 1.0 - gas_fraction
                liquid_part = discrete_skew_normal(
                    liquid_fraction * liquid_mean,
                    np.sqrt(liquid_fraction) * liquid_scale,
                    alpha,
                )
                interface += np.convolve(gas_part, liquid_part)[: len(count_axis)]
            return gas, liquid, interface / points

        def objective(parameters):
            gas_mean, _, liquid_mean, liquid_scale, weights, alpha = unpack(
                parameters
            )
            components = component_probabilities(
                gas_mean,
                liquid_mean,
                liquid_scale,
                alpha,
            )
            mixture = sum(
                weight * component
                for weight, component in zip(weights, components)
            )
            return float(
                -np.dot(observed, np.log(np.clip(mixture, 1e-300, None)))
            )

        maximum_count = max(2.0, float(count_axis[-1]))
        candidates = []
        for alpha_start in (-2.0, 0.0, 2.0):
            initial = np.array(
                [
                    np.log(gas_guess),
                    np.log(liquid_mean_guess - gas_guess),
                    np.log(scale_guess),
                    np.log(weights_guess[0] / weights_guess[2]),
                    np.log(weights_guess[1] / weights_guess[2]),
                    alpha_start,
                ]
            )
            candidates.append(
                minimize(
                    objective,
                    initial,
                    method="L-BFGS-B",
                    bounds=[
                        (np.log(1e-3), np.log(maximum_count)),
                        (np.log(1e-2), np.log(2.0 * maximum_count)),
                        (np.log(0.05), np.log(maximum_count)),
                        (-12.0, 12.0),
                        (-12.0, 12.0),
                        (-20.0, 20.0),
                    ],
                    options={"maxiter": int(max_iterations)},
                )
            )
        optimum = min(candidates, key=lambda result: result.fun)
        gas_mean, gap, liquid_mean, liquid_scale, weights, alpha = unpack(
            optimum.x
        )
        components = component_probabilities(
            gas_mean,
            liquid_mean,
            liquid_scale,
            alpha,
        )
        expected_voxels = float(nbins**3)
        gas_counts = expected_voxels * weights[0] * components[0]
        liquid_counts = expected_voxels * weights[1] * components[1]
        interface_counts = expected_voxels * weights[2] * components[2]
        model_counts = gas_counts + liquid_counts + interface_counts

        hessian = _finite_difference_hessian(objective, optimum.x)
        try:
            covariance = np.linalg.inv(hessian)
            covariance_method = "inverse_hessian"
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(hessian)
            covariance_method = "pseudo_inverse_hessian"
        covariance = 0.5 * (covariance + covariance.T)

        gas_gradient = np.array(
            [gas_mean / voxel_volume, 0, 0, 0, 0, 0]
        )
        liquid_gradient = np.array(
            [gas_mean / voxel_volume, gap / voxel_volume, 0, 0, 0, 0]
        )
        alpha_gradient = np.array([0, 0, 0, 0, 0, 1.0])
        log_likelihood = -float(optimum.fun)
        records.append(
            {
                "Run_ID": str(normal_fit["Run_ID"]),
                "success": bool(optimum.success),
                "message": str(optimum.message),
                "method": "averaged_voxel_skew_liquid_mixture",
                "voxel_nbins": nbins,
                "frames_used": frames_used,
                "count_axis": count_axis,
                "density_axis": count_axis / voxel_volume,
                "observed_counts": observed_average,
                "model_counts": model_counts,
                "gas_counts": gas_counts,
                "liquid_counts": liquid_counts,
                "interface_counts": interface_counts,
                "n_voxels_per_frame": nbins**3,
                "voxel_volume": voxel_volume,
                "box_volume": box_volume,
                "interface_points": points,
                "rho_liquid": liquid_mean / voxel_volume,
                "rho_liquid_unc": _standard_uncertainty(
                    liquid_gradient, covariance
                ),
                "rho_gas": gas_mean / voxel_volume,
                "rho_gas_unc": _standard_uncertainty(gas_gradient, covariance),
                "liquid_scale_density": liquid_scale / voxel_volume,
                "liquid_shape_alpha": float(alpha),
                "liquid_shape_alpha_unc": _standard_uncertainty(
                    alpha_gradient, covariance
                ),
                "gas_weight": float(weights[0]),
                "liquid_weight": float(weights[1]),
                "interface_weight": float(weights[2]),
                "uncertainty_method": covariance_method,
                "parameter_covariance": covariance,
                "log_likelihood": log_likelihood,
                "AIC": float(12 - 2 * log_likelihood),
                "BIC": float(
                    6 * np.log(nbins**3 * frames_used) - 2 * log_likelihood
                ),
                "delta_AIC_vs_normal": float(
                    12 - 2 * log_likelihood - float(normal_fit["AIC"])
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("voxel_nbins").reset_index(
        drop=True
    )


def plot_phase_nbins_mixtures(
    fit_results,
    *,
    title: str,
    liquid_label: str,
    density_xlim=None,
):
    """Plot phase-mixture components with raw and standardized residuals."""

    import matplotlib.pyplot as plt

    selected = fit_results.sort_values("voxel_nbins")
    if selected.empty:
        raise ValueError("fit_results is empty")
    columns = min(3, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(
        3 * rows,
        columns,
        figsize=(5.4 * columns, 6.8 * rows),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [value for _ in range(rows) for value in (3, 1, 1)]
        },
    )
    for panel_index, (_, fit) in enumerate(selected.iterrows()):
        panel_row, column = divmod(panel_index, columns)
        fit_axis = axes[3 * panel_row, column]
        raw_axis = axes[3 * panel_row + 1, column]
        standard_axis = axes[3 * panel_row + 2, column]
        density = np.asarray(fit["density_axis"], dtype=float)
        observed = np.asarray(fit["observed_counts"], dtype=float)
        predicted = np.asarray(fit["model_counts"], dtype=float)
        residual = observed - predicted
        frames_used = int(fit["frames_used"])
        voxels = int(fit["n_voxels_per_frame"])
        probability = predicted / voxels
        variance = voxels * probability * (1.0 - probability) / frames_used
        expected_total = predicted * frames_used
        standardized = np.full_like(residual, np.nan)
        valid = (
            (expected_total >= 5.0)
            & np.isfinite(variance)
            & (variance > 0.0)
        )
        standardized[valid] = residual[valid] / np.sqrt(variance[valid])

        fit_axis.step(density, observed, where="mid", color="black", label="data")
        fit_axis.plot(density, predicted, color="tab:purple", label="total fit")
        fit_axis.plot(density, fit["gas_counts"], ":", label="Poisson vapor")
        fit_axis.plot(density, fit["liquid_counts"], "--", label=liquid_label)
        fit_axis.plot(
            density,
            fit["interface_counts"],
            "-.",
            label="interface",
        )
        panel_title = (
            f"nbins={int(fit['voxel_nbins'])}, "
            rf"$\rho_l$={fit['rho_liquid']:.4g}"
        )
        if "liquid_shape_alpha" in fit.index:
            panel_title += rf", $\alpha$={fit['liquid_shape_alpha']:.3g}"
        fit_axis.set(title=panel_title, ylabel="Mean number of voxels")
        fit_axis.grid(alpha=0.25)
        fit_axis.legend(fontsize="small")

        raw_axis.step(density, residual, where="mid", color="tab:blue")
        raw_axis.axhline(0.0, color="black", linewidth=0.8)
        raw_axis.set(ylabel="Data − fit\n(voxels)")
        raw_axis.grid(alpha=0.25)

        standard_axis.step(density, standardized, where="mid", color="tab:green")
        standard_axis.axhline(0.0, color="black", linewidth=0.8)
        for level in (-2.0, 2.0):
            standard_axis.axhline(
                level,
                color="tab:orange",
                linestyle="--",
                linewidth=0.9,
            )
        standard_axis.set(
            xlabel="Voxel density",
            ylabel="Standardized\nresidual",
        )
        standard_axis.grid(alpha=0.25)
        if density_xlim is not None:
            for axis in (fit_axis, raw_axis, standard_axis):
                axis.set_xlim(*density_xlim)

    for panel_index in range(len(selected), rows * columns):
        panel_row, column = divmod(panel_index, columns)
        for offset in range(3):
            axes[3 * panel_row + offset, column].set_visible(False)
    figure.suptitle(title)
    return figure, axes


def plot_phase_liquid_density_vs_nbins(fit_results, *, title: str):
    """Plot fitted liquid-component mean density against voxel resolution."""

    import matplotlib.pyplot as plt

    selected = fit_results.sort_values("voxel_nbins")
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.errorbar(
        selected["voxel_nbins"],
        selected["rho_liquid"],
        yerr=selected["rho_liquid_unc"],
        marker="o",
        capsize=3,
    )
    axis.set(
        xlabel="nbins (per box dimension)",
        ylabel="Liquid-component mean density",
        title=title,
    )
    axis.grid(alpha=0.3)
    return figure, axis


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
