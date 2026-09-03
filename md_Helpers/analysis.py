"""Post-run analyses that can be reused by every simulation stage."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


VOXEL_METHOD_VERSION = "voxel_low_density_fraction_v1"
PE_DROP_METHOD_VERSION = "pe_drop_v1"
COMBINED_METHOD_VERSION = "phase_combination_v1"


def voxel_bins_for_ncells(n_cells: int) -> int:
    """Preserve the V3 voxel-resolution rule."""

    return round(0.3 * int(n_cells) + 3)


def classify_voxel_histogram(
    positions: np.ndarray,
    box: np.ndarray,
    n_cells: int,
    density_threshold: float = 0.2,
    voxel_fraction_threshold: float = 0.01,
) -> dict[str, Any]:
    """Classify a final state from its fraction of low-density voxels."""

    positions = np.asarray(positions, dtype=np.float64)
    box_lengths = np.asarray(box, dtype=np.float64)[:3]
    nbins = voxel_bins_for_ncells(n_cells)
    wrapped = (positions + box_lengths / 2.0) % box_lengths - box_lengths / 2.0
    bounds = [[-length / 2.0, length / 2.0] for length in box_lengths]
    counts, _ = np.histogramdd(wrapped, bins=nbins, range=bounds)
    voxel_volume = float(np.prod(box_lengths / nbins))
    densities = counts.ravel() / voxel_volume
    low_density_fraction = float(np.mean(densities < density_threshold))

    return {
        "phase_separated": low_density_fraction > voxel_fraction_threshold,
        "method": "voxel_histogram",
        "method_version": VOXEL_METHOD_VERSION,
        "nbins": int(nbins),
        "density_threshold": float(density_threshold),
        "voxel_fraction_threshold": float(voxel_fraction_threshold),
        "low_density_fraction": low_density_fraction,
        "voxel_volume": voxel_volume,
        "minimum_density": float(np.min(densities)),
        "maximum_density": float(np.max(densities)),
        "mean_density": float(np.mean(densities)),
        "density_std": float(np.std(densities, ddof=1)),
    }


def classify_pe_drop(
    potential_energy: np.ndarray,
    n_particles: int,
    n_last: int = 100,
    drop_threshold: float = -0.15,
    z_limit: float = 5.0,
    decision_rule: str = "either",
) -> dict[str, Any]:
    """Preserve the V3 potential-energy-drop classification."""

    values = np.asarray(potential_energy, dtype=np.float64) / int(n_particles)
    if values.size == 0:
        raise ValueError("potential_energy cannot be empty")

    window = values[-min(int(n_last), len(values)) :]
    last_mean = float(np.mean(window))
    last_std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
    pe_drop = last_mean - float(values[0])
    if last_std > 0:
        z_score = pe_drop / last_std
    elif pe_drop < 0:
        z_score = -math.inf
    elif pe_drop > 0:
        z_score = math.inf
    else:
        z_score = 0.0

    raw_pass = pe_drop <= float(drop_threshold)
    z_pass = z_score <= -float(z_limit)
    rules = {
        "raw": raw_pass,
        "z_score": z_pass,
        "either": raw_pass or z_pass,
        "both": raw_pass and z_pass,
    }
    if decision_rule not in rules:
        raise ValueError("decision_rule must be raw, z_score, either, or both")

    return {
        "phase_separated": bool(rules[decision_rule]),
        "method": "PE_drop",
        "method_version": PE_DROP_METHOD_VERSION,
        "decision_rule": decision_rule,
        "n_last": min(int(n_last), len(values)),
        "drop_threshold": float(drop_threshold),
        "z_limit": float(z_limit),
        "starting_PE_per_particle": float(values[0]),
        "last_PE_per_particle_mean": last_mean,
        "last_PE_per_particle_std": last_std,
        "PE_drop": float(pe_drop),
        "PE_drop_z_score": float(z_score),
        "passes_drop_threshold": bool(raw_pass),
        "passes_z_limit": bool(z_pass),
    }


def select_phase_classification(
    voxel: dict[str, Any],
    pe_drop: dict[str, Any],
    method: str = "voxel_histogram",
) -> dict[str, Any]:
    """Select the canonical SQL decision while retaining both HDF5 results."""

    decisions = {
        "voxel_histogram": bool(voxel["phase_separated"]),
        "PE_drop": bool(pe_drop["phase_separated"]),
        "combined_either": bool(
            voxel["phase_separated"] or pe_drop["phase_separated"]
        ),
        "combined_both": bool(
            voxel["phase_separated"] and pe_drop["phase_separated"]
        ),
    }
    if method not in decisions:
        raise ValueError(
            "phase_method must be voxel_histogram, PE_drop, "
            "combined_either, or combined_both"
        )

    version = (
        voxel["method_version"]
        if method == "voxel_histogram"
        else pe_drop["method_version"]
        if method == "PE_drop"
        else COMBINED_METHOD_VERSION
    )
    separated = decisions[method]
    return {
        "phase_separated": separated,
        "status": "Separated" if separated else "Not_Separated",
        "method": method,
        "method_version": version,
    }


def _stats(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None, None, None
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, std, std / math.sqrt(len(finite))


def thermodynamic_summary(
    run_steps: np.ndarray,
    pressure: np.ndarray,
    potential_energy: np.ndarray,
    n_particles: int,
    n_last: int = 100,
) -> dict[str, Any]:
    """Summarize the last logged samples for the SQL result table."""

    run_steps = np.asarray(run_steps, dtype=np.int64)
    if len(run_steps) == 0:
        raise ValueError("At least one logged sample is required")
    count = min(int(n_last), len(run_steps))
    selection = slice(len(run_steps) - count, None)
    pressure_mean, pressure_std, pressure_sem = _stats(
        np.asarray(pressure)[selection]
    )
    pe_mean, pe_std, pe_sem = _stats(
        np.asarray(potential_energy)[selection] / int(n_particles)
    )
    return {
        "Summary_Start_Step": int(run_steps[selection][0]),
        "Summary_End_Step": int(run_steps[selection][-1]),
        "Summary_Num_Samples": int(count),
        "Pressure_Mean": pressure_mean,
        "Pressure_Std": pressure_std,
        "Pressure_SEM": pressure_sem,
        "PE_Per_Particle_Mean": pe_mean,
        "PE_Per_Particle_Std": pe_std,
        "PE_Per_Particle_SEM": pe_sem,
    }

