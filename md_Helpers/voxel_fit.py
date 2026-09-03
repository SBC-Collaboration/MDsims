"""Conditional final-frame voxel mixture fit for every V4 workflow."""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import voxel_bins_for_ncells


PHASE_FIT_METHOD = "final_frame_voxel_mixture"
PHASE_FIT_METHOD_VERSION = "final_frame_voxel_mixture_v1"

PHASE_FIT_SQL_FIELDS = (
    "rho_liquid",
    "rho_liquid_unc",
    "rho_gas",
    "rho_gas_unc",
    "V_liquid",
    "V_liquid_unc",
    "V_gas",
    "V_gas_unc",
)


def _finite_difference_hessian(function, parameters, relative_step=1e-4):
    parameters = np.asarray(parameters, dtype=float)
    steps = relative_step * np.maximum(1.0, np.abs(parameters))
    hessian = np.zeros((len(parameters), len(parameters)), dtype=float)
    center = float(function(parameters))
    for i in range(len(parameters)):
        step_i = np.zeros_like(parameters)
        step_i[i] = steps[i]
        hessian[i, i] = (
            function(parameters + step_i)
            - 2.0 * center
            + function(parameters - step_i)
        ) / steps[i] ** 2
        for j in range(i + 1, len(parameters)):
            step_j = np.zeros_like(parameters)
            step_j[j] = steps[j]
            value = (
                function(parameters + step_i + step_j)
                - function(parameters + step_i - step_j)
                - function(parameters - step_i + step_j)
                + function(parameters - step_i - step_j)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return 0.5 * (hessian + hessian.T)


def _standard_uncertainty(gradient, covariance) -> float:
    variance = float(gradient @ covariance @ gradient)
    return float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else np.nan


def fit_final_frame_voxel_mixture(
    positions: np.ndarray,
    box: np.ndarray,
    n_cells: int,
    interface_void_fraction: float = 0.5,
    interface_points: int = 40,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """Fit V3's gas/liquid/interface count model to the exact final frame."""

    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from scipy.stats import norm, poisson

    interface_void_fraction = float(interface_void_fraction)
    interface_points = int(interface_points)
    if not 0 <= interface_void_fraction <= 1:
        raise ValueError("interface_void_fraction must be between 0 and 1")
    if interface_points <= 0 or int(max_iterations) <= 0:
        raise ValueError("interface_points and max_iterations must be positive")

    positions = np.asarray(positions, dtype=np.float64)
    box_lengths = np.asarray(box, dtype=np.float64)[:3]
    nbins = voxel_bins_for_ncells(n_cells)
    wrapped = (positions + box_lengths / 2.0) % box_lengths - box_lengths / 2.0
    bounds = [[-length / 2.0, length / 2.0] for length in box_lengths]
    voxel_counts, _ = np.histogramdd(wrapped, bins=nbins, range=bounds)
    voxel_counts = voxel_counts.astype(int).ravel()
    voxel_volume = float(np.prod(box_lengths / nbins))
    box_volume = float(np.prod(box_lengths))
    count_axis = np.arange(int(voxel_counts.max()) + 1)
    observed = np.bincount(voxel_counts, minlength=len(count_axis)).astype(float)

    liquid_guess = max(1.0, float(np.argmax(observed)))
    low_mask = count_axis < max(1.0, 0.35 * liquid_guess)
    gas_guess = (
        float(np.average(count_axis[low_mask], weights=observed[low_mask]))
        if observed[low_mask].sum() > 0
        else max(0.1, 0.05 * liquid_guess)
    )
    gas_guess = max(0.05, min(gas_guess, 0.5 * liquid_guess))
    dense = np.repeat(count_axis, observed.astype(int))
    dense = dense[dense > 0.5 * liquid_guess]
    sigma_guess = float(np.std(dense, ddof=1)) if len(dense) > 1 else np.sqrt(liquid_guess)
    sigma_guess = max(0.5, sigma_guess)
    gas_weight = max(0.02, observed[low_mask].sum() / observed.sum())
    interface_weight = max(0.05, min(0.3, 2.0 * gas_weight))
    liquid_weight = max(0.05, 1.0 - gas_weight - interface_weight)
    weights = np.array([gas_weight, liquid_weight, interface_weight])
    weights /= weights.sum()
    initial = np.array([
        np.log(gas_guess),
        np.log(max(0.1, liquid_guess - gas_guess)),
        np.log(sigma_guess),
        np.log(weights[0] / weights[2]),
        np.log(weights[1] / weights[2]),
    ])

    def unpack(parameters):
        gas_mean = np.exp(parameters[0])
        gap = np.exp(parameters[1])
        liquid_mean = gas_mean + gap
        liquid_sigma = np.exp(parameters[2])
        logits = np.array([parameters[3], parameters[4], 0.0])
        fitted_weights = np.exp(logits - logsumexp(logits))
        return gas_mean, gap, liquid_mean, liquid_sigma, fitted_weights

    def discrete_normal(mean, sigma):
        lower = count_axis - 0.5
        lower[count_axis == 0] = -np.inf
        return norm.cdf(count_axis + 0.5, mean, sigma) - norm.cdf(
            lower,
            mean,
            sigma,
        )

    def component_probabilities(gas_mean, liquid_mean, liquid_sigma):
        gas = poisson.pmf(count_axis, gas_mean)
        liquid = discrete_normal(liquid_mean, liquid_sigma)
        interface = np.zeros_like(gas)
        fractions = (np.arange(interface_points) + 0.5) / interface_points
        for gas_fraction in fractions:
            gas_part = poisson.pmf(count_axis, gas_fraction * gas_mean)
            liquid_fraction = 1.0 - gas_fraction
            lower = count_axis - 0.5
            lower[count_axis == 0] = -np.inf
            liquid_part = norm.cdf(
                count_axis + 0.5,
                liquid_fraction * liquid_mean,
                np.sqrt(liquid_fraction) * liquid_sigma,
            ) - norm.cdf(
                lower,
                liquid_fraction * liquid_mean,
                np.sqrt(liquid_fraction) * liquid_sigma,
            )
            interface += np.convolve(gas_part, liquid_part)[: len(count_axis)]
        return gas, liquid, interface / interface_points

    def objective(parameters):
        gas_mean, _, liquid_mean, liquid_sigma, fitted_weights = unpack(parameters)
        components = component_probabilities(gas_mean, liquid_mean, liquid_sigma)
        mixture = sum(weight * component for weight, component in zip(fitted_weights, components))
        return float(-np.dot(observed, np.log(np.clip(mixture, 1e-300, None))))

    maximum_count = max(2.0, float(count_axis[-1]))
    optimum = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[
            (np.log(1e-3), np.log(maximum_count)),
            (np.log(1e-2), np.log(2.0 * maximum_count)),
            (np.log(0.1), np.log(maximum_count)),
            (-12.0, 12.0),
            (-12.0, 12.0),
        ],
        options={"maxiter": int(max_iterations)},
    )
    gas_mean, gap, liquid_mean, _, fitted_weights = unpack(optimum.x)
    hessian = _finite_difference_hessian(objective, optimum.x)
    try:
        covariance = np.linalg.inv(hessian)
        covariance_method = "inverse_hessian"
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
        covariance_method = "pseudo_inverse_hessian"
    covariance = 0.5 * (covariance + covariance.T)

    gas_density_gradient = np.array([gas_mean / voxel_volume, 0, 0, 0, 0])
    liquid_density_gradient = np.array([
        gas_mean / voxel_volume,
        gap / voxel_volume,
        0,
        0,
        0,
    ])
    weight_jacobian = np.empty((3, 2), dtype=float)
    for component in range(3):
        weight_jacobian[component, 0] = fitted_weights[component] * (
            (1.0 if component == 0 else 0.0) - fitted_weights[0]
        )
        weight_jacobian[component, 1] = fitted_weights[component] * (
            (1.0 if component == 1 else 0.0) - fitted_weights[1]
        )
    gas_volume_gradient = np.zeros(5)
    liquid_volume_gradient = np.zeros(5)
    gas_volume_gradient[3:] = box_volume * (
        weight_jacobian[0] + interface_void_fraction * weight_jacobian[2]
    )
    liquid_volume_gradient[3:] = box_volume * (
        weight_jacobian[1]
        + (1.0 - interface_void_fraction) * weight_jacobian[2]
    )

    gas_volume = box_volume * (
        fitted_weights[0] + interface_void_fraction * fitted_weights[2]
    )
    liquid_volume = box_volume - gas_volume
    log_likelihood = -float(optimum.fun)
    return {
        "success": bool(optimum.success),
        "message": str(optimum.message),
        "method": PHASE_FIT_METHOD,
        "method_version": PHASE_FIT_METHOD_VERSION,
        "voxel_nbins": int(nbins),
        "n_voxels": int(len(voxel_counts)),
        "voxel_volume": voxel_volume,
        "box_volume": box_volume,
        "interface_void_fraction": interface_void_fraction,
        "interface_points": interface_points,
        "max_iterations": int(max_iterations),
        "rho_liquid": float(liquid_mean / voxel_volume),
        "rho_liquid_unc": _standard_uncertainty(liquid_density_gradient, covariance),
        "rho_gas": float(gas_mean / voxel_volume),
        "rho_gas_unc": _standard_uncertainty(gas_density_gradient, covariance),
        "V_liquid": float(liquid_volume),
        "V_liquid_unc": _standard_uncertainty(liquid_volume_gradient, covariance),
        "V_gas": float(gas_volume),
        "V_gas_unc": _standard_uncertainty(gas_volume_gradient, covariance),
        "gas_weight": float(fitted_weights[0]),
        "liquid_weight": float(fitted_weights[1]),
        "interface_weight": float(fitted_weights[2]),
        "uncertainty_method": covariance_method,
        "parameter_covariance": covariance,
        "log_likelihood": log_likelihood,
        "AIC": float(10 - 2 * log_likelihood),
        "BIC": float(5 * np.log(len(voxel_counts)) - 2 * log_likelihood),
    }


def conditional_phase_fit(
    voxel_classification: dict[str, Any],
    positions: np.ndarray,
    box: np.ndarray,
    n_cells: int,
    **fit_options: Any,
) -> dict[str, Any]:
    """Fit only states classified as separated by the voxel classifier.

    This is the mandatory V4 gate for thermalization, cavitation, and
    excitation. Homogeneous states retain NULL for all eight SQL fit values.
    """

    empty_values = {field: None for field in PHASE_FIT_SQL_FIELDS}
    if not bool(voxel_classification.get("phase_separated", False)):
        return {
            "status": "Skipped_Homogeneous",
            "method": None,
            "method_version": None,
            "message": "Voxel classification did not identify phase separation.",
            **empty_values,
        }

    try:
        fit = fit_final_frame_voxel_mixture(
            positions,
            box,
            n_cells,
            **fit_options,
        )
    except Exception as error:
        return {
            "status": "Failed",
            "method": PHASE_FIT_METHOD,
            "method_version": PHASE_FIT_METHOD_VERSION,
            "message": f"{type(error).__name__}: {error}",
            **empty_values,
        }
    if not fit["success"]:
        return {
            "status": "Failed",
            "method": PHASE_FIT_METHOD,
            "method_version": PHASE_FIT_METHOD_VERSION,
            "message": fit["message"],
            **empty_values,
        }
    return {"status": "Complete", **fit}


def phase_fit_sql_values(fit: dict[str, Any]) -> dict[str, Any]:
    """Select the exact Phase-Fit columns stored in SQL."""

    return {
        **{field: fit.get(field) for field in PHASE_FIT_SQL_FIELDS},
        "Phase_Fit_Status": fit["status"],
        "Phase_Fit_Method": fit.get("method"),
        "Phase_Fit_Method_Version": fit.get("method_version"),
    }

