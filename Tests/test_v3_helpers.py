import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from md_Helpers import eos_sweep, master_csv, paths, seitz, spatial
from md_Helpers import cavitation_sweep as cavitation_sweep_module
from md_Helpers.cavitation_analysis import estimate_bubble_from_radial_density
from md_Helpers.cavitation_sweep import (
    run_cavitation_size_sweep,
    summarize_bubble_survival,
)
from md_Helpers.voxel_fit import (
    bubble_size_from_voxel_fit,
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
            30, 0.8, 0.8, 1_000_000, 1, 2.0,
        )
        evolved = paths.cavitation_evolved_paths(
            30, 0.8, 0.8, 1_000_000, 1, 2.0, 0.8, 100_000, 1,
        )
        self.assertEqual(initial["state_path"].name, "cavitation_initial.gsd")
        self.assertEqual(evolved["final_state_path"].name, "cavitation_final.gsd")
        self.assertIn("radius_2.000", str(initial["state_path"]))


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
        call_kwargs = {}
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
        def fake_get_or_create_cavitation(**kwargs):
            call_kwargs.update(kwargs)
            return fake_result

        fake_cavitation = SimpleNamespace(
            get_or_create_cavitation=fake_get_or_create_cavitation,
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
                radius=2.0,
                summary_path=Path(tmp) / "summary.csv",
            )

        self.assertEqual(
            summary.loc[0, "run_status"],
            "thermalization_failed_phase_separated",
        )
        self.assertFalse(bool(summary.loc[0, "thermalization_passed"]))
        self.assertEqual(summary.loc[0, "outcome"], "not_cavitated")
        self.assertEqual(call_kwargs["radius"], 2.0)


class VoxelMixtureFitTests(unittest.TestCase):
    def test_converts_mixture_weights_to_equivalent_radius(self):
        result = bubble_size_from_voxel_fit(
            {
                "gas_weight": 0.02,
                "interface_weight": 0.06,
            },
            box_volume=1_000.0,
        )

        self.assertAlmostEqual(result["bubble_volume_fraction"], 0.05)
        self.assertAlmostEqual(result["bubble_volume_estimate"], 50.0)
        self.assertAlmostEqual(
            result["bubble_radius_estimate"],
            (3.0 * 50.0 / (4.0 * np.pi)) ** (1.0 / 3.0),
        )

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


class SeitzTests(unittest.TestCase):
    def test_seitz_threshold_matches_whiteboard_formula(self):
        result = seitz.seitz_threshold(
            volume=10.0,
            n_cavity=6.0,
            u_cavity=-12.0,
            rho0=0.8,
            u0=-2.0,
            p0=0.5,
        )

        n0 = 8.0
        u0_total = -16.0
        expected = (-12.0 - u0_total) + ((n0 - 6.0) / n0) * (
            u0_total + 0.5 * 10.0
        )

        self.assertAlmostEqual(result, expected)

    def test_interpolates_liquid_reference_by_density(self):
        result = seitz.interpolate_liquid_reference(
            target_rho=0.75,
            rho=[0.8, 0.7],
            u_per_particle=[-3.0, -2.0],
            pressure=[0.4, 0.2],
        )

        self.assertAlmostEqual(result["rho0"], 0.75)
        self.assertAlmostEqual(result["u0"], -2.5)
        self.assertAlmostEqual(result["p0"], 0.3)

    def test_liquid_reference_from_eos_filters_and_interpolates(self):
        eos = pd.DataFrame({
            "status": ["completed", "completed", "completed"],
            "phase_separated": [False, False, True],
            "kT": [0.8, 0.8, 0.8],
            "actual_rho": [0.7, 0.8, 0.9],
            "PE_per_particle_mean_last100": [-2.0, -3.0, -4.0],
            "pressure_mean_last100": [0.2, 0.4, 0.6],
        })

        result = seitz.liquid_reference_from_eos(
            eos,
            kT=0.8,
            target_rho=0.75,
        )

        self.assertAlmostEqual(result["rho0"], 0.75)
        self.assertAlmostEqual(result["u0"], -2.5)
        self.assertAlmostEqual(result["p0"], 0.3)

    def test_cavity_terms_from_frame_sums_lj_per_particle_energy(self):
        frame = make_frame(
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            box_lengths=[10.0, 10.0, 10.0],
        )

        result = seitz.cavity_terms_from_frame(
            frame,
            radius=0.2,
            center=[0.0, 0.0, 0.0],
            r_cut_LJ=2.5,
            lj_mode="none",
        )

        self.assertEqual(result["n_cavity"], 1)
        self.assertAlmostEqual(result["u_cavity"], 0.0)


class MasterCsvTests(unittest.TestCase):
    def test_builds_thermalization_master_csv_from_hdf5_logs(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Thermalized_States_v3"
            folder = (
                root / "FCC" / "n_cells_30" / "rho_0.700"
                / "kT_0.800" / "nsteps_1000" / "seed_1"
            )
            folder.mkdir(parents=True)
            log_path = folder / "randomization_log.hdf5"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["state_kind"] = "thermalized"
                state.attrs["n_fcc_cells"] = 30
                state.attrs["N"] = 108000
                state.attrs["target_rho"] = 0.7
                state.attrs["actual_rho"] = 0.7
                state.attrs["kT"] = 0.8
                state.attrs["BoxLength"] = 10.0
                state.attrs["volume"] = 1000.0

                run = hdf.require_group("metadata/run")
                run.attrs["nsteps"] = 1000
                run.attrs["seed"] = 1

                phase = hdf.require_group(
                    "metadata/classification/phase_separation"
                )
                phase.attrs["phase_separated"] = False

                hdf.create_dataset(
                    "hoomd-data/Simulation/timestep",
                    data=np.array([0, 1, 2]),
                )
                thermo = hdf.require_group(
                    "hoomd-data/md/compute/ThermodynamicQuantities"
                )
                thermo.create_dataset(
                    "pressure",
                    data=np.array([1.0, 2.0, 3.0]),
                )
                thermo.create_dataset(
                    "potential_energy",
                    data=np.array([-108000.0, -216000.0, -324000.0]),
                )

            output_path = Path(tmp) / "master.csv"
            table = master_csv.build_thermalization_master_csv(
                root=root,
                output_path=output_path,
                n_last=2,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(table), 1)
            self.assertAlmostEqual(
                table.loc[0, "pressure_mean_last2"],
                2.5,
            )
            self.assertAlmostEqual(
                table.loc[0, "PE_per_particle_mean_last2"],
                -2.5,
            )


class EosSweepTests(unittest.TestCase):
    def test_pressure_region_uses_old_window_defaults(self):
        self.assertEqual(
            eos_sweep.pressure_region(-0.01),
            "below_window",
        )
        self.assertEqual(
            eos_sweep.pressure_region(0.10),
            "inside_window",
        )
        self.assertEqual(
            eos_sweep.pressure_region(0.20),
            "above_window",
        )

    def test_liquid_eos_table_filters_completed_liquid_rows(self):
        table = pd.DataFrame({
            "status": ["completed", "failed", "completed"],
            "phase_separated": [False, False, True],
            "n_fcc_cells": [28, 28, 28],
            "kT": [0.8, 0.8, 0.8],
            "actual_rho": [0.7, 0.71, 0.72],
            "pressure_mean_last100": [0.1, 0.2, 0.3],
            "PE_per_particle_mean_last100": [-4.0, -4.1, -4.2],
        })

        liquid = eos_sweep.liquid_eos_table(table, n_fcc_cells=28)

        self.assertEqual(len(liquid), 1)
        self.assertAlmostEqual(liquid.loc[0, "actual_rho"], 0.7)


if __name__ == "__main__":
    unittest.main()
