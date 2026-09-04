"""Conditional, time-averaged voxel mixture fit for every V4 workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import voxel_bins_for_ncells


PHASE_FIT_METHOD = "averaged_voxel_histogram_mixture"
PHASE_FIT_METHOD_VERSION = "terminal_5_saved_frames_log_stride_10_v2"
PHASE_FIT_NUM_FRAMES = 5
PHASE_FIT_FRAME_STRIDE = 5

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


def phase_fit_frame_indices(
    trajectory_length: int,
    num_frames: int = PHASE_FIT_NUM_FRAMES,
    frame_stride: int = PHASE_FIT_FRAME_STRIDE,
) -> list[int]:
    """Return final-first, zero-based frame indices for the averaged fit."""

    trajectory_length = int(trajectory_length)
    num_frames = int(num_frames)
    frame_stride = int(frame_stride)
    if trajectory_length <= 0:
        raise ValueError("trajectory must contain at least one frame")
    if num_frames <= 0 or frame_stride <= 0:
        raise ValueError("num_frames and frame_stride must be positive")
    final_index = trajectory_length - 1
    return [
        final_index - offset
        for offset in range(0, num_frames * frame_stride, frame_stride)
        if final_index - offset >= 0
    ]


def _voxel_count_histogram(
    positions: np.ndarray,
    box: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, float, float]:
    positions = np.asarray(positions, dtype=np.float64)
    box_lengths = np.asarray(box, dtype=np.float64)[:3]
    wrapped = (positions + box_lengths / 2.0) % box_lengths - box_lengths / 2.0
    bounds = [[-length / 2.0, length / 2.0] for length in box_lengths]
    voxel_counts, _ = np.histogramdd(wrapped, bins=nbins, range=bounds)
    voxel_counts = voxel_counts.astype(int).ravel()
    observed = np.bincount(voxel_counts).astype(float)
    return (
        observed,
        float(np.prod(box_lengths / nbins)),
        float(np.prod(box_lengths)),
    )


def averaged_trajectory_voxel_histogram(
    trajectory_path: str | Path,
    n_cells: int,
    num_frames: int = PHASE_FIT_NUM_FRAMES,
    frame_stride: int = PHASE_FIT_FRAME_STRIDE,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return the exact multi-frame histogram used as phase-fit input."""

    import gsd.hoomd

    nbins = voxel_bins_for_ncells(n_cells)
    histograms = []
    voxel_volumes = []
    box_volumes = []
    with gsd.hoomd.open(name=str(trajectory_path), mode="r") as trajectory:
        trajectory_length = len(trajectory)
        indices = (
            phase_fit_frame_indices(trajectory_length, num_frames, frame_stride)
            if frame_indices is None
            else [int(index) for index in frame_indices]
        )
        if not indices:
            raise ValueError("At least one phase-fit frame is required")
        if any(index < 0 or index >= trajectory_length for index in indices):
            raise IndexError("A phase-fit frame index is outside the trajectory")
        for index in indices:
            frame = trajectory[index]
            histogram, voxel_volume, box_volume = _voxel_count_histogram(
                frame.particles.position,
                frame.configuration.box,
                nbins,
            )
            histograms.append(histogram)
            voxel_volumes.append(voxel_volume)
            box_volumes.append(box_volume)

    max_count_bins = max(len(histogram) for histogram in histograms)
    padded = np.zeros((len(histograms), max_count_bins), dtype=float)
    for row, histogram in enumerate(histograms):
        padded[row, : len(histogram)] = histogram
    voxel_volume = float(np.mean(voxel_volumes))
    count_axis = np.arange(max_count_bins)
    return {
        "frame_indices": indices,
        "negative_frame_indices": [
            index - trajectory_length for index in indices
        ],
        "requested_frames": (
            int(num_frames) if frame_indices is None else len(indices)
        ),
        "frames_used": len(indices),
        "frame_stride": int(frame_stride) if frame_indices is None else None,
        "voxel_nbins": int(nbins),
        "voxel_volume": voxel_volume,
        "box_volume": float(np.mean(box_volumes)),
        "count_axis": count_axis,
        "density_axis": count_axis / voxel_volume,
        "individual_histograms": padded,
        "observed_counts": np.mean(padded, axis=0),
    }


def fit_averaged_voxel_mixture(
    positions_by_frame: Sequence[np.ndarray],
    boxes_by_frame: Sequence[np.ndarray],
    n_cells: int,
    frame_indices: Sequence[int] | None = None,
    interface_void_fraction: float = 0.5,
    interface_points: int = 40,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """Fit V3's model once to the average of several voxel histograms."""

    from scipy.optimize import minimize
    from scipy.special import logsumexp
    from scipy.stats import norm, poisson

    interface_void_fraction = float(interface_void_fraction)
    interface_points = int(interface_points)
    if not 0 <= interface_void_fraction <= 1:
        raise ValueError("interface_void_fraction must be between 0 and 1")
    if interface_points <= 0 or int(max_iterations) <= 0:
        raise ValueError("interface_points and max_iterations must be positive")

    positions_by_frame = list(positions_by_frame)
    boxes_by_frame = list(boxes_by_frame)
    if not positions_by_frame or len(positions_by_frame) != len(boxes_by_frame):
        raise ValueError("positions_by_frame and boxes_by_frame must have equal length")

    nbins = voxel_bins_for_ncells(n_cells)
    histograms = []
    voxel_volumes = []
    box_volumes = []
    for positions, box in zip(positions_by_frame, boxes_by_frame):
        histogram, voxel_volume, box_volume = _voxel_count_histogram(
            positions,
            box,
            nbins,
        )
        histograms.append(histogram)
        voxel_volumes.append(voxel_volume)
        box_volumes.append(box_volume)

    max_count_bins = max(len(histogram) for histogram in histograms)
    padded = np.zeros((len(histograms), max_count_bins), dtype=float)
    for row, histogram in enumerate(histograms):
        padded[row, : len(histogram)] = histogram
    observed_average = np.mean(padded, axis=0)
    # Scaling the averaged histogram by the number of frames leaves the best-fit
    # parameters unchanged while making covariance reflect all sampled voxels.
    observed = observed_average * len(histograms)
    voxel_volume = float(np.mean(voxel_volumes))
    box_volume = float(np.mean(box_volumes))
    count_axis = np.arange(max_count_bins)

    liquid_guess = max(1.0, float(np.argmax(observed)))
    low_mask = count_axis < max(1.0, 0.35 * liquid_guess)
    gas_guess = (
        float(np.average(count_axis[low_mask], weights=observed[low_mask]))
        if observed[low_mask].sum() > 0
        else max(0.1, 0.05 * liquid_guess)
    )
    gas_guess = max(0.05, min(gas_guess, 0.5 * liquid_guess))
    dense_mask = count_axis > 0.5 * liquid_guess
    dense_weights = observed_average[dense_mask]
    if dense_weights.sum() > 1:
        dense_counts = count_axis[dense_mask]
        dense_mean = np.average(dense_counts, weights=dense_weights)
        sigma_guess = float(np.sqrt(np.average(
            (dense_counts - dense_mean) ** 2,
            weights=dense_weights,
        )))
    else:
        sigma_guess = np.sqrt(liquid_guess)
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
    component_probabilities_at_optimum = component_probabilities(
        gas_mean,
        liquid_mean,
        float(np.exp(optimum.x[2])),
    )
    expected_voxels_per_frame = float(nbins**3)
    gas_counts = (
        expected_voxels_per_frame
        * fitted_weights[0]
        * component_probabilities_at_optimum[0]
    )
    liquid_counts = (
        expected_voxels_per_frame
        * fitted_weights[1]
        * component_probabilities_at_optimum[1]
    )
    interface_counts = (
        expected_voxels_per_frame
        * fitted_weights[2]
        * component_probabilities_at_optimum[2]
    )
    model_counts = gas_counts + liquid_counts + interface_counts
    log_likelihood = -float(optimum.fun)
    return {
        "success": bool(optimum.success),
        "message": str(optimum.message),
        "method": PHASE_FIT_METHOD,
        "method_version": PHASE_FIT_METHOD_VERSION,
        "voxel_nbins": int(nbins),
        "frames_used": int(len(histograms)),
        "frame_indices": list(frame_indices) if frame_indices is not None else None,
        "frame_stride": PHASE_FIT_FRAME_STRIDE,
        "histogram_aggregation": "arithmetic_mean",
        "count_axis": count_axis,
        "density_axis": count_axis / voxel_volume,
        "individual_histograms": padded,
        "observed_counts": observed_average,
        "model_counts": model_counts,
        "gas_counts": gas_counts,
        "liquid_counts": liquid_counts,
        "interface_counts": interface_counts,
        "n_voxels_per_frame": int(nbins**3),
        "n_voxel_samples": int(nbins**3 * len(histograms)),
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
        "BIC": float(5 * np.log(nbins**3 * len(histograms)) - 2 * log_likelihood),
    }


def fit_trajectory_voxel_mixture(
    trajectory_path: str | Path,
    n_cells: int,
    num_frames: int = PHASE_FIT_NUM_FRAMES,
    frame_stride: int = PHASE_FIT_FRAME_STRIDE,
    frame_indices: Sequence[int] | None = None,
    **fit_options: Any,
) -> dict[str, Any]:
    """Read selected GSD frames and fit their averaged voxel histogram."""

    import gsd.hoomd

    with gsd.hoomd.open(name=str(trajectory_path), mode="r") as trajectory:
        indices = (
            phase_fit_frame_indices(len(trajectory), num_frames, frame_stride)
            if frame_indices is None
            else [int(index) for index in frame_indices]
        )
        if not indices:
            raise ValueError("At least one phase-fit frame is required")
        if any(index < 0 or index >= len(trajectory) for index in indices):
            raise IndexError("A phase-fit frame index is outside the trajectory")
        positions = []
        boxes = []
        for index in indices:
            frame = trajectory[index]
            positions.append(np.asarray(frame.particles.position, dtype=np.float64))
            boxes.append(np.asarray(frame.configuration.box, dtype=np.float64))
    fit = fit_averaged_voxel_mixture(
        positions,
        boxes,
        n_cells,
        frame_indices=indices,
        **fit_options,
    )
    fit["requested_frames"] = (
        int(num_frames) if frame_indices is None else len(indices)
    )
    fit["frame_stride"] = (
        int(frame_stride) if frame_indices is None else None
    )
    return fit


def conditional_phase_fit(
    voxel_classification: dict[str, Any],
    trajectory_path: str | Path,
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
        fit = fit_trajectory_voxel_mixture(
            trajectory_path,
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
