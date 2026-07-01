"""Parameter sweeps for cavitation survival versus FCC system size."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import cavitation_analysis


def summarize_bubble_survival(
    measurements,
    tail_fraction=0.2,
    stabilized_radius_ratio=0.5,
    collapsed_radius_ratio=0.1,
):
    """Summarize whether a bubble persists near its constructed location.

    The outcome is based on the median estimated radius over the final portion
    of the trajectory, rather than on one noisy final frame.
    """

    required = {
        "bubble_radius_estimate",
        "initial_bubble_radius",
        "bulk_density",
        "density_inside_initial_radius",
    }
    missing = required.difference(measurements.columns)
    if missing:
        raise ValueError(
            "measurements are missing required columns: "
            + ", ".join(sorted(missing))
        )
    if measurements.empty:
        raise ValueError("measurements must contain at least one frame")
    if not 0 < float(tail_fraction) <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not 0 <= collapsed_radius_ratio < stabilized_radius_ratio:
        raise ValueError(
            "require 0 <= collapsed_radius_ratio < "
            "stabilized_radius_ratio"
        )

    initial_radius = float(measurements["initial_bubble_radius"].iloc[0])
    if initial_radius <= 0:
        raise ValueError("initial_bubble_radius must be positive")

    tail_count = max(1, int(np.ceil(len(measurements) * tail_fraction)))
    tail = measurements.tail(tail_count)
    final = measurements.iloc[-1]

    tail_radius = float(tail["bubble_radius_estimate"].median())
    tail_radius_ratio = tail_radius / initial_radius
    tail_inside_density = float(
        tail["density_inside_initial_radius"].median()
    )
    tail_bulk_density = float(tail["bulk_density"].median())
    density_recovery_ratio = (
        tail_inside_density / tail_bulk_density
        if tail_bulk_density > 0
        else np.nan
    )

    if tail_radius_ratio >= stabilized_radius_ratio:
        radius_outcome = "persisted"
    elif tail_radius_ratio <= collapsed_radius_ratio:
        radius_outcome = "collapsed"
    else:
        radius_outcome = "intermediate"

    return {
        "radius_outcome": radius_outcome,
        "n_frames": int(len(measurements)),
        "tail_frames": tail_count,
        "initial_bubble_radius": initial_radius,
        "initial_measured_radius": float(
            measurements["bubble_radius_estimate"].iloc[0]
        ),
        "final_bubble_radius": float(final["bubble_radius_estimate"]),
        "tail_median_bubble_radius": tail_radius,
        "tail_radius_ratio": tail_radius_ratio,
        "tail_inside_density": tail_inside_density,
        "tail_bulk_density": tail_bulk_density,
        "tail_density_recovery_ratio": density_recovery_ratio,
        "final_void_fraction": float(final["void_fraction_estimate"]),
        "stabilized_radius_ratio_threshold": float(
            stabilized_radius_ratio
        ),
        "collapsed_radius_ratio_threshold": float(collapsed_radius_ratio),
    }


def _normalize_conditions(conditions):
    normalized = []
    for condition in conditions:
        if isinstance(condition, dict):
            density = condition.get("density", condition.get("rho"))
            temperature = condition.get("temperature", condition.get("kT"))
            label = condition.get("label")
        else:
            try:
                density, temperature = condition
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "each condition must be a (density, temperature) pair "
                    "or a dictionary"
                ) from error
            label = None

        if density is None or temperature is None:
            raise ValueError("each condition requires density and temperature")
        density = float(density)
        temperature = float(temperature)
        if density <= 0:
            raise ValueError("condition density must be positive")
        normalized.append({
            "density": density,
            "temperature": temperature,
            "condition_label": label or f"rho={density:g}, kT={temperature:g}",
        })
    if not normalized:
        raise ValueError("conditions must not be empty")
    return normalized


def run_cavitation_size_sweep(
    n_fcc_cells_values,
    conditions,
    source_nsteps,
    evolve_nsteps,
    radius,
    evolve_seeds=(1,),
    source_seed=1,
    trajectory_period=1_000,
    log_period=1_000,
    summary_path="cavitation_size_sweep.csv",
    tail_fraction=0.2,
    stabilized_radius_ratio=0.5,
    collapsed_radius_ratio=0.1,
    overwrite=False,
    **cavitation_kwargs,
):
    """Run a fixed-absolute-radius cavitation sweep and save one row per run.

    ``conditions`` is an explicit sequence of ``(density, temperature)``
    pairs, so nearby state points can be compared without accidentally taking
    their Cartesian product. ``radius`` is held fixed in simulation length
    units while the FCC system size changes.
    """

    # Keep HOOMD/GSD as run-time dependencies so the summary helper remains
    # importable in lightweight analysis and test environments.
    from . import cavitation
    from . import classification

    cells_values = [int(value) for value in n_fcc_cells_values]
    if not cells_values or any(value <= 0 for value in cells_values):
        raise ValueError("n_fcc_cells_values must contain positive integers")
    conditions = _normalize_conditions(conditions)
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    evolve_seeds = [int(seed) for seed in evolve_seeds]
    if not evolve_seeds:
        raise ValueError("evolve_seeds must not be empty")

    forbidden = {
        "radius",
        "evolve_kT",
        "target_rho",
        "kT",
        "reject_phase_separated_source",
    }
    conflicts = forbidden.intersection(cavitation_kwargs)
    if conflicts:
        raise ValueError(
            "sweep-controlled arguments cannot be passed in cavitation_kwargs: "
            + ", ".join(sorted(conflicts))
        )

    rows = []
    summary_path = Path(summary_path) if summary_path is not None else None

    for condition in conditions:
        for n_fcc_cells in cells_values:
            source_particle_count = 4 * n_fcc_cells ** 3
            box_length = (
                source_particle_count / condition["density"]
            ) ** (1.0 / 3.0)
            if radius >= 0.5 * box_length:
                raise ValueError(
                    f"radius={radius:g} must be smaller than half the "
                    f"box length ({0.5 * box_length:g}) for "
                    f"n_fcc_cells={n_fcc_cells}, "
                    f"density={condition['density']:g}"
                )
            for evolve_seed in evolve_seeds:
                result = cavitation.get_or_create_cavitation(
                    n_fcc_cells=n_fcc_cells,
                    target_rho=condition["density"],
                    kT=condition["temperature"],
                    source_nsteps=int(source_nsteps),
                    radius=radius,
                    evolve_nsteps=int(evolve_nsteps),
                    evolve_kT=condition["temperature"],
                    evolve_seed=evolve_seed,
                    source_seed=int(source_seed),
                    trajectory_period=int(trajectory_period),
                    log_period=int(log_period),
                    overwrite=overwrite,
                    reject_phase_separated_source=True,
                    **cavitation_kwargs,
                )
                base_row = {
                    **condition,
                    "n_fcc_cells": n_fcc_cells,
                    "N_source": source_particle_count,
                    "evolve_seed": evolve_seed,
                    "radius": radius,
                }
                if result.get("status") == "source_phase_separated":
                    source_phase = result["initial_result"][
                        "source_phase_separation"
                    ]
                    source_paths = result["initial_result"]["source_result"][
                        "paths"
                    ]
                    rows.append({
                        **base_row,
                        "run_status": "thermalization_failed_phase_separated",
                        "thermalization_passed": False,
                        "source_phase_separated": True,
                        "source_low_density_fraction": source_phase.get(
                            "low_density_fraction"
                        ),
                        "outcome": "not_cavitated",
                        "source_state_path": str(source_paths["state_path"]),
                        "source_log_path": str(source_paths["log_path"]),
                    })
                    if summary_path is not None:
                        summary_path.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).to_csv(summary_path, index=False)
                    continue
                if result.get("status") == "missing_source":
                    raise RuntimeError(
                        "thermalized source state is missing for "
                        f"n_fcc_cells={n_fcc_cells}, "
                        f"density={condition['density']}, "
                        f"temperature={condition['temperature']}"
                    )

                measurements = cavitation_analysis.measure_cavitation_trajectory(
                    evolution=result,
                )
                survival = summarize_bubble_survival(
                    measurements,
                    tail_fraction=tail_fraction,
                    stabilized_radius_ratio=stabilized_radius_ratio,
                    collapsed_radius_ratio=collapsed_radius_ratio,
                )
                voxel, _ = classification.read_phase_method_attrs(
                    result["paths"]["log_path"],
                    "voxel",
                )
                source_paths = result["initial_result"]["source_result"][
                    "paths"
                ]

                row = {
                    **base_row,
                    "run_status": "cavitation_completed",
                    "thermalization_passed": True,
                    "source_phase_separated": False,
                    "source_low_density_fraction": result["initial_result"]
                    .get("source_phase_separation", {})
                    .get("low_density_fraction"),
                    "source_state_path": str(source_paths["state_path"]),
                    "source_log_path": str(source_paths["log_path"]),
                    "post_cavitation_N": int(measurements["N"].iloc[0]),
                    "box_length": float(measurements["BoxLength_x"].iloc[0]),
                    "final_phase_separated": voxel.get("phase_separated"),
                    "final_low_density_fraction": voxel.get(
                        "low_density_fraction"
                    ),
                    "trajectory_path": str(result["paths"]["trajectory_path"]),
                    "log_path": str(result["paths"]["log_path"]),
                    **survival,
                }
                row["outcome"] = (
                    "stabilized"
                    if bool(row["final_phase_separated"])
                    else "rethermalized"
                )
                rows.append(row)

                if summary_path is not None:
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(summary_path, index=False)

    return pd.DataFrame(rows)
