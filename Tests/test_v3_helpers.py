import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from md_Helpers import paths, spatial
from md_Helpers import cavitation_sweep as cavitation_sweep_module
from md_Helpers.cavitation_analysis import estimate_bubble_from_radial_density
from md_Helpers.cavitation_sweep import (
    run_cavitation_size_sweep,
    summarize_bubble_survival,
)
from md_Helpers.voxel_fit import (
    fit_voxel_count_mixture,
    voxel_mixture_components,
)


def make_frame(positions, box_lengths):
    return SimpleNamespace(
        configuration=SimpleNamespace(
            box=[*box_lengths, 0.0, 0.0, 0.0],
        ),
        particles=SimpleNamespace(
            position=np.asarray(positions, dtype=float),
            N=len(positions),
        ),
    )


class PathTests(unittest.TestCase):
    def test_thermalized_path(self):
        result = paths.thermalized_run_paths(
            n_fcc_cells=30,
            target_rho=0.64,
            kT=0.8,
            nsteps=1_000_000,
            seed=1,
        )
        self.assertEqual(result["state_path"].name, "randomization.gsd")
        self.assertIn("rho_0.640", str(result["state_path"]))
        self.assertIn("kT_0.800", str(result["state_path"]))

    def test_cavitation_paths_are_distinct(self):
        initial = paths.cavitation_state_paths(
            30, 0.8, 0.8, 1_000_000, 1, 0.1,
        )
        evolved = paths.cavitation_evolved_paths(
            30, 0.8, 0.8, 1_000_000, 1, 0.1, 0.8, 100_000, 1,
        )
        self.assertEqual(initial["state_path"].name, "cavitation_initial.gsd")
        self.assertEqual(evolved["final_state_path"].name, "cavitation_final.gsd")


class SpatialTests(unittest.TestCase):
    def test_periodic_distance(self):
        distance = spatial.periodic_distances(
            positions=[[0.49, 0.0, 0.0]],
            center=[-0.49, 0.0, 0.0],
            box_lengths=[1.0, 1.0, 1.0],
        )
        self.assertAlmostEqual(distance[0], 0.02)

    def test_voxel_counts_conserve_particles_after_wrapping(self):
        frame = make_frame(
            positions=[[0.6, 0.0, 0.0], [-0.6, 0.0, 0.0]],
            box_lengths=[1.0, 1.0, 1.0],
        )
        densities, counts, voxel_volume = spatial.compute_voxel_densities(
            frame,
            nbins=2,
        )
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(len(densities), 8)
        self.assertAlmostEqual(voxel_volume, 0.125)


class CavitationAnalysisTests(unittest.TestCase):
    def test_radial_density_recovers_near_known_radius(self):
        rng = np.random.default_rng(12)
        inner_radius = 2.0
        outer_radius = 5.0
        uniform_volume = rng.uniform(
            inner_radius ** 3,
            outer_radius ** 3,
            size=100_000,
        )
        distances = uniform_volume ** (1.0 / 3.0)

        result = estimate_bubble_from_radial_density(
            distances=distances,
            box_lengths=[10.0, 10.0, 10.0],
            bulk_density=100.0,
            n_radial_bins=50,
            density_threshold_fraction=0.5,
            recovery_bins=3,
        )
        self.assertAlmostEqual(
            result["bubble_radius_estimate"],
            inner_radius,
            delta=0.2,
        )

    def test_summarizes_tail_as_stabilized(self):
        measurements = pd.DataFrame({
            "bubble_radius_estimate": [2.0, 1.8, 1.7, 1.6],
            "initial_bubble_radius": [2.0] * 4,
            "bulk_density": [0.7] * 4,
            "density_inside_initial_radius": [0.0, 0.1, 0.1, 0.1],
            "void_fraction_estimate": [0.03, 0.02, 0.02, 0.02],
        })
        summary = summarize_bubble_survival(
            measurements,
            tail_fraction=0.5,
        )
        self.assertEqual(summary["radius_outcome"], "persisted")
        self.assertAlmostEqual(summary["tail_median_bubble_radius"], 1.65)

    def test_summarizes_tail_as_collapsed(self):
        measurements = pd.DataFrame({
            "bubble_radius_estimate": [2.0, 0.4, 0.1, 0.0],
            "initial_bubble_radius": [2.0] * 4,
            "bulk_density": [0.7] * 4,
            "density_inside_initial_radius": [0.0, 0.4, 0.65, 0.7],
            "void_fraction_estimate": [0.03, 0.005, 0.0, 0.0],
        })
        summary = summarize_bubble_survival(
            measurements,
            tail_fraction=0.5,
        )
        self.assertEqual(summary["radius_outcome"], "collapsed")


class CavitationSweepTests(unittest.TestCase):
    def test_phase_separated_source_is_recorded_without_measurement(self):
        fake_result = {
            "status": "source_phase_separated",
            "initial_result": {
                "source_phase_separation": {
                    "phase_separated": True,
                    "low_density_fraction": 0.25,
                },
                "source_result": {
                    "paths": {
                        "state_path": "source.gsd",
                        "log_path": "source.hdf5",
                    },
                },
            },
        }
        fake_cavitation = SimpleNamespace(
            get_or_create_cavitation=lambda **kwargs: fake_result,
        )

        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "md_Helpers.cavitation": fake_cavitation,
            "md_Helpers.classification": SimpleNamespace(),
        }), patch.object(
            cavitation_sweep_module.cavitation_analysis,
            "measure_cavitation_trajectory",
            side_effect=AssertionError("failed source must not be measured"),
        ):
            summary = run_cavitation_size_sweep(
                n_fcc_cells_values=[10],
                conditions=[(0.71, 0.8)],
                source_nsteps=100,
                evolve_nsteps=100,
                summary_path=Path(tmp) / "summary.csv",
            )

        self.assertEqual(
            summary.loc[0, "run_status"],
            "thermalization_failed_phase_separated",
        )
        self.assertFalse(bool(summary.loc[0, "thermalization_passed"]))
        self.assertEqual(summary.loc[0, "outcome"], "not_cavitated")


class VoxelMixtureFitTests(unittest.TestCase):
    def test_recovers_synthetic_three_component_model(self):
        count_axis = np.arange(81)
        gas, liquid, interface = voxel_mixture_components(
            count_axis,
            gas_mean=2.0,
            liquid_mean=50.0,
            liquid_sigma=5.0,
            interface_points=15,
        )
        probability = 0.08 * gas + 0.75 * liquid + 0.17 * interface
        probability /= probability.sum()

        samples = np.random.default_rng(5).choice(
            count_axis,
            size=4_000,
            p=probability,
        )
        fit = fit_voxel_count_mixture(
            samples,
            voxel_volume=1.0,
            interface_points=15,
            max_iterations=300,
        )

        self.assertTrue(fit["success"])
        self.assertAlmostEqual(fit["gas_mean_count"], 2.0, delta=0.7)
        self.assertAlmostEqual(fit["liquid_mean_count"], 50.0, delta=1.5)
        self.assertAlmostEqual(fit["liquid_weight"], 0.75, delta=0.12)


if __name__ == "__main__":
    unittest.main()
