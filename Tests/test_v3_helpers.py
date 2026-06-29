import unittest
from types import SimpleNamespace

import numpy as np

from md_Helpers import paths, spatial
from md_Helpers.cavitation_analysis import estimate_bubble_from_radial_density


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


if __name__ == "__main__":
    unittest.main()
