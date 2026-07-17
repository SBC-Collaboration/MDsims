from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, poisson

from .spatial import compute_voxel_densities


def _discrete_normal_pmf(counts, mean, sigma):
    counts = np.asarray(counts, dtype=int)
    if sigma <= 1e-8:
        pmf = np.zeros_like(counts, dtype=float)
        nearest = int(np.clip(round(mean), counts[0], counts[-1]))
        pmf[nearest - counts[0]] = 1.0
        return pmf

    lower = counts - 0.5
    lower[counts == 0] = -np.inf
    upper = counts + 0.5
    return norm.cdf(upper, mean, sigma) - norm.cdf(lower, mean, sigma)


def voxel_mixture_components(
    counts,
    gas_mean,
    liquid_mean,
    liquid_sigma,
    interface_points=40,
):
    """Return gas, liquid, and gas-liquid interface count PMFs."""

    counts = np.asarray(counts, dtype=int)
    support_max = int(counts.max())
    support = np.arange(support_max + 1)

    gas = poisson.pmf(support, gas_mean)
    liquid = _discrete_normal_pmf(support, liquid_mean, liquid_sigma)

    interface = np.zeros_like(gas)
    gas_fractions = (
        np.arange(int(interface_points), dtype=float) + 0.5
    ) / int(interface_points)

    for gas_fraction in gas_fractions:
        gas_part = poisson.pmf(support, gas_fraction * gas_mean)
        liquid_fraction = 1.0 - gas_fraction
        liquid_part = _discrete_normal_pmf(
            support,
            liquid_fraction * liquid_mean,
            np.sqrt(liquid_fraction) * liquid_sigma,
        )
        interface += np.convolve(gas_part, liquid_part)[:len(support)]

    interface /= len(gas_fractions)
    return gas[counts], liquid[counts], interface[counts]


def _unpack_parameters(parameters):
    gas_mean = np.exp(parameters[0])
    liquid_mean = gas_mean + np.exp(parameters[1])
    liquid_sigma = np.exp(parameters[2])
    logits = np.array([parameters[3], parameters[4], 0.0])
    weights = np.exp(logits - logsumexp(logits))
    return gas_mean, liquid_mean, liquid_sigma, weights


def _finite_difference_hessian(function, parameters, relative_step=1e-4):
    """Numerically estimate the Hessian of a scalar function."""

    parameters = np.asarray(parameters, dtype=float)
    n_parameters = len(parameters)
    steps = relative_step * np.maximum(1.0, np.abs(parameters))
    hessian = np.zeros((n_parameters, n_parameters), dtype=float)
    center_value = float(function(parameters))

    for i in range(n_parameters):
        step_i = np.zeros(n_parameters, dtype=float)
        step_i[i] = steps[i]
        forward = float(function(parameters + step_i))
        backward = float(function(parameters - step_i))
        hessian[i, i] = (
            forward - 2.0 * center_value + backward
        ) / steps[i] ** 2

        for j in range(i + 1, n_parameters):
            step_j = np.zeros(n_parameters, dtype=float)
            step_j[j] = steps[j]
            f_pp = float(function(parameters + step_i + step_j))
            f_pm = float(function(parameters + step_i - step_j))
            f_mp = float(function(parameters - step_i + step_j))
            f_mm = float(function(parameters - step_i - step_j))
            mixed = (
                f_pp - f_pm - f_mp + f_mm
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = mixed
            hessian[j, i] = mixed

    return 0.5 * (hessian + hessian.T)


def _covariance_from_hessian(hessian):
    """Return an approximate covariance matrix from a Hessian."""

    try:
        covariance = np.linalg.inv(hessian)
        method = "inverse_hessian"
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
        method = "pseudo_inverse_hessian"

    covariance = 0.5 * (covariance + covariance.T)
    return covariance, method


def _liquid_mean_gradient(parameters):
    """Gradient of fitted liquid count mean in transformed parameter space."""

    parameters = np.asarray(parameters, dtype=float)
    gradient = np.zeros_like(parameters, dtype=float)
    gradient[0] = np.exp(parameters[0])
    gradient[1] = np.exp(parameters[1])
    return gradient


def _fit_uncertainty_summary(objective, parameters, voxel_volume):
    """
    Estimate fitted liquid-mean uncertainty from the likelihood Hessian.

    The optimizer works in transformed parameters.  The covariance from the
    negative-log-likelihood Hessian is propagated to the fitted liquid Gaussian
    mean and then divided by voxel volume to get density uncertainty.
    """

    hessian = _finite_difference_hessian(objective, parameters)
    covariance, method = _covariance_from_hessian(hessian)
    liquid_gradient = _liquid_mean_gradient(parameters)
    liquid_variance = float(liquid_gradient @ covariance @ liquid_gradient)

    if liquid_variance >= 0.0 and np.isfinite(liquid_variance):
        liquid_mean_uncertainty = float(np.sqrt(liquid_variance))
        liquid_density_uncertainty = (
            liquid_mean_uncertainty / float(voxel_volume)
        )
    else:
        liquid_mean_uncertainty = np.nan
        liquid_density_uncertainty = np.nan

    parameter_variances = np.diag(covariance)
    parameter_uncertainties = np.where(
        parameter_variances >= 0.0,
        np.sqrt(np.clip(parameter_variances, 0.0, None)),
        np.nan,
    )

    return {
        "parameter_hessian": hessian,
        "parameter_covariance": covariance,
        "parameter_uncertainties": parameter_uncertainties,
        "uncertainty_method": method,
        "liquid_mean_count_uncertainty": liquid_mean_uncertainty,
        "liquid_density_uncertainty": liquid_density_uncertainty,
    }


def _initial_parameters(observed):
    count_axis = np.arange(len(observed))
    liquid_mean = max(1.0, float(np.argmax(observed)))
    low_mask = count_axis < max(1.0, 0.35 * liquid_mean)
    low_counts = observed[low_mask]
    low_axis = count_axis[low_mask]
    gas_mean = (
        float(np.average(low_axis, weights=low_counts))
        if low_counts.sum() > 0
        else max(0.1, 0.05 * liquid_mean)
    )
    gas_mean = max(0.05, min(gas_mean, 0.5 * liquid_mean))

    dense_samples = np.repeat(count_axis, observed.astype(int))
    dense_samples = dense_samples[dense_samples > 0.5 * liquid_mean]
    liquid_sigma = (
        float(np.std(dense_samples, ddof=1))
        if len(dense_samples) > 1
        else np.sqrt(liquid_mean)
    )
    liquid_sigma = max(0.5, liquid_sigma)

    gas_weight = max(0.02, observed[low_mask].sum() / observed.sum())
    interface_weight = max(0.05, min(0.3, 2.0 * gas_weight))
    liquid_weight = max(0.05, 1.0 - gas_weight - interface_weight)
    weights = np.array([gas_weight, liquid_weight, interface_weight])
    weights /= weights.sum()

    return np.array([
        np.log(gas_mean),
        np.log(max(0.1, liquid_mean - gas_mean)),
        np.log(liquid_sigma),
        np.log(weights[0] / weights[2]),
        np.log(weights[1] / weights[2]),
    ])


def fit_voxel_count_mixture(
    voxel_counts,
    voxel_volume,
    interface_points=40,
    max_iterations=500,
):
    """Fit the whiteboard gas + liquid + interface voxel-count model."""

    voxel_counts = np.asarray(voxel_counts, dtype=int)
    if len(voxel_counts) == 0 or np.any(voxel_counts < 0):
        raise ValueError("voxel_counts must be a non-empty nonnegative array")

    count_axis = np.arange(int(voxel_counts.max()) + 1)
    observed = np.bincount(voxel_counts, minlength=len(count_axis)).astype(float)
    initial = _initial_parameters(observed)
    max_count = max(2.0, float(count_axis[-1]))

    def objective(parameters):
        gas_mean, liquid_mean, liquid_sigma, weights = _unpack_parameters(
            parameters
        )
        gas, liquid, interface = voxel_mixture_components(
            count_axis,
            gas_mean,
            liquid_mean,
            liquid_sigma,
            interface_points=interface_points,
        )
        mixture = weights[0] * gas + weights[1] * liquid + weights[2] * interface
        return float(-np.dot(observed, np.log(np.clip(mixture, 1e-300, None))))

    bounds = [
        (np.log(1e-3), np.log(max_count)),
        (np.log(1e-2), np.log(2.0 * max_count)),
        (np.log(0.1), np.log(max_count)),
        (-12.0, 12.0),
        (-12.0, 12.0),
    ]
    optimum = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iterations)},
    )

    gas_mean, liquid_mean, liquid_sigma, weights = _unpack_parameters(
        optimum.x
    )
    uncertainty = _fit_uncertainty_summary(
        objective,
        optimum.x,
        voxel_volume=voxel_volume,
    )
    gas_pmf, liquid_pmf, interface_pmf = voxel_mixture_components(
        count_axis,
        gas_mean,
        liquid_mean,
        liquid_sigma,
        interface_points=interface_points,
    )
    component_pmfs = {
        "gas": weights[0] * gas_pmf,
        "liquid": weights[1] * liquid_pmf,
        "interface": weights[2] * interface_pmf,
    }
    mixture_pmf = sum(component_pmfs.values())
    n_voxels = len(voxel_counts)
    parameter_count = 5
    log_likelihood = -float(optimum.fun)

    return {
        "success": bool(optimum.success),
        "message": str(optimum.message),
        "count_axis": count_axis,
        "density_axis": count_axis / float(voxel_volume),
        "observed_counts": observed,
        "model_counts": n_voxels * mixture_pmf,
        "gas_counts": n_voxels * component_pmfs["gas"],
        "liquid_counts": n_voxels * component_pmfs["liquid"],
        "interface_counts": n_voxels * component_pmfs["interface"],
        "gas_mean_count": float(gas_mean),
        "liquid_mean_count": float(liquid_mean),
        "liquid_mean_count_uncertainty": float(
            uncertainty["liquid_mean_count_uncertainty"]
        ),
        "liquid_sigma_count": float(liquid_sigma),
        "gas_density": float(gas_mean / voxel_volume),
        "liquid_density": float(liquid_mean / voxel_volume),
        "liquid_density_uncertainty": float(
            uncertainty["liquid_density_uncertainty"]
        ),
        "liquid_sigma_density": float(liquid_sigma / voxel_volume),
        "gas_weight": float(weights[0]),
        "liquid_weight": float(weights[1]),
        "interface_weight": float(weights[2]),
        "voxel_volume": float(voxel_volume),
        "n_voxels": n_voxels,
        "log_likelihood": log_likelihood,
        "parameter_hessian": uncertainty["parameter_hessian"],
        "parameter_covariance": uncertainty["parameter_covariance"],
        "parameter_uncertainties": uncertainty["parameter_uncertainties"],
        "uncertainty_method": uncertainty["uncertainty_method"],
        "AIC": float(2 * parameter_count - 2 * log_likelihood),
        "BIC": float(
            parameter_count * np.log(n_voxels) - 2 * log_likelihood
        ),
    }


def _trajectory_path(result, explicit_path=None):
    if explicit_path is not None:
        return Path(explicit_path)
    if isinstance(result, dict) and "trajectory_path" in result.get("paths", {}):
        return Path(result["paths"]["trajectory_path"])
    raise ValueError("Pass an evolution result or trajectory_path")


def fit_last_frame_voxel_histogram(
    evolution=None,
    trajectory_path=None,
    frame_index=-1,
    voxel_nbins=10,
    interface_points=40,
    max_iterations=500,
):
    """Load one trajectory frame and fit its voxel-count distribution."""

    import gsd.hoomd

    path = _trajectory_path(evolution, trajectory_path)
    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        frame = trajectory[int(frame_index)]
        timestep = int(frame.configuration.step)

    _, voxel_counts, voxel_volume = compute_voxel_densities(
        frame,
        nbins=voxel_nbins,
    )
    result = fit_voxel_count_mixture(
        voxel_counts,
        voxel_volume,
        interface_points=interface_points,
        max_iterations=max_iterations,
    )
    result.update({
        "trajectory_path": str(path),
        "frame_index": int(frame_index),
        "timestep": timestep,
        "voxel_nbins": int(voxel_nbins),
    })
    return result


def bubble_size_from_voxel_fit(
    fit,
    box_volume,
    interface_void_fraction=0.5,
):
    """Convert voxel-mixture weights to an equivalent spherical bubble size.

    ``interface_void_fraction`` specifies how much of an interface voxel is
    assigned to the bubble.  A value of 0.5 matches the uniformly sampled
    interface fractions used by :func:`voxel_mixture_components`.
    """

    interface_void_fraction = float(interface_void_fraction)
    if not 0.0 <= interface_void_fraction <= 1.0:
        raise ValueError("interface_void_fraction must be between 0 and 1")

    box_volume = float(box_volume)
    if box_volume <= 0.0:
        raise ValueError("box_volume must be positive")

    bubble_volume_fraction = (
        float(fit["gas_weight"])
        + interface_void_fraction * float(fit["interface_weight"])
    )
    bubble_volume = bubble_volume_fraction * box_volume
    bubble_radius = (
        3.0 * bubble_volume / (4.0 * np.pi)
    ) ** (1.0 / 3.0)

    return {
        "bubble_volume_fraction": float(bubble_volume_fraction),
        "bubble_volume_estimate": float(bubble_volume),
        "bubble_radius_estimate": float(bubble_radius),
        "interface_void_fraction": interface_void_fraction,
        "box_volume": box_volume,
    }


def fit_trajectory_tail_voxel_histogram(
    evolution=None,
    trajectory_path=None,
    voxel_nbins=12,
    nframes=None,
    skip=1,
    tail_fraction=0.5,
    interface_void_fraction=0.5,
    interface_points=40,
    max_iterations=500,
):
    """Pool tail-frame voxel counts and estimate the final bubble size.

    Frames are selected backward from the final trajectory frame, separated
    by ``skip``, and constrained to the final ``tail_fraction`` of the
    trajectory.  If ``nframes`` is supplied, only that many of the most recent
    eligible frames are pooled.  Cavitation trajectories have fixed boxes;
    this function raises an error if the selected voxel volumes differ.
    """

    import gsd.hoomd

    voxel_nbins = int(voxel_nbins)
    skip = int(skip)
    tail_fraction = float(tail_fraction)
    if voxel_nbins <= 0:
        raise ValueError("voxel_nbins must be positive")
    if skip <= 0:
        raise ValueError("skip must be positive")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must satisfy 0 < value <= 1")
    if nframes is not None and int(nframes) <= 0:
        raise ValueError("nframes must be positive or None")

    path = _trajectory_path(evolution, trajectory_path)
    frame_counts = []
    timesteps = []
    box_volumes = []

    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        total_frames = len(trajectory)
        if total_frames == 0:
            raise ValueError(f"No frames found in trajectory: {path}")

        first_tail_frame = int(np.floor((1.0 - tail_fraction) * total_frames))
        selected_indices = list(range(
            total_frames - 1,
            first_tail_frame - 1,
            -skip,
        ))
        if nframes is not None:
            selected_indices = selected_indices[:int(nframes)]
        selected_indices.reverse()

        for frame_index in selected_indices:
            frame = trajectory[frame_index]
            _, counts, voxel_volume = compute_voxel_densities(
                frame,
                nbins=voxel_nbins,
            )
            frame_counts.append(np.asarray(counts, dtype=int))
            timesteps.append(int(frame.configuration.step))
            box_volumes.append(float(voxel_volume) * voxel_nbins ** 3)

    if not np.allclose(box_volumes, box_volumes[0]):
        raise ValueError(
            "Selected frames have different box volumes; pooled integer-count "
            "histograms require a fixed box."
        )

    voxel_volume = box_volumes[0] / voxel_nbins ** 3
    pooled_counts = np.concatenate(frame_counts)
    pooled_fit = fit_voxel_count_mixture(
        pooled_counts,
        voxel_volume=voxel_volume,
        interface_points=interface_points,
        max_iterations=max_iterations,
    )

    average_fit = pooled_fit.copy()
    for key in [
        "observed_counts",
        "model_counts",
        "gas_counts",
        "liquid_counts",
        "interface_counts",
    ]:
        average_fit[key] = pooled_fit[key] / len(selected_indices)

    average_fit.update({
        "trajectory_path": str(path),
        "frame_indices": selected_indices,
        "timesteps": timesteps,
        "nframes": len(selected_indices),
        "voxel_nbins": voxel_nbins,
        "tail_fraction": tail_fraction,
        "skip": skip,
        "timestep": f"average of {len(selected_indices)} tail frames",
    })
    average_fit.update(bubble_size_from_voxel_fit(
        average_fit,
        box_volume=box_volumes[0],
        interface_void_fraction=interface_void_fraction,
    ))

    return average_fit
