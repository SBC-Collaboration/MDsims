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
        "liquid_sigma_count": float(liquid_sigma),
        "gas_density": float(gas_mean / voxel_volume),
        "liquid_density": float(liquid_mean / voxel_volume),
        "liquid_sigma_density": float(liquid_sigma / voxel_volume),
        "gas_weight": float(weights[0]),
        "liquid_weight": float(weights[1]),
        "interface_weight": float(weights[2]),
        "voxel_volume": float(voxel_volume),
        "n_voxels": n_voxels,
        "log_likelihood": log_likelihood,
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
