import unittest
import sys
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from md_Helpers import (
    classification,
    dt_validation,
    eos_sweep,
    excitation_evolution,
    master_csv,
    paths,
    run_logs,
    seitz,
    spatial,
)
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


class RunLogTests(unittest.TestCase):
    def tearDown(self):
        run_logs.configure_run_logging(False)

    def test_progress_is_printed_and_immediately_written(self):
        with TemporaryDirectory() as folder:
            log_path = run_logs.configure_run_logging(
                True,
                notebook_name="My Sweep.ipynb",
                log_dir=folder,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                with run_logs.simulation_progress(
                    "Cavitation",
                    ncells=30,
                    source_rho=0.71,
                    kT=0.8,
                    radius=4.0,
                    nsteps=100_000,
                ):
                    pass

            cell_text = output.getvalue()
            file_text = log_path.read_text(encoding="utf-8")
            self.assertTrue(log_path.name.startswith("My_Sweep_"))
            self.assertRegex(
                log_path.name,
                r"^My_Sweep_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log$",
            )
            start_line = (
                "Starting Cavitation Evolution (ncells = 30, "
                "source_rho = 0.71, kT = 0.8, radius = 4.0, "
                "nsteps = 100000)"
            )
            self.assertIn(start_line, cell_text)
            self.assertIn(start_line, file_text)
            self.assertIn("simulation done, simulation time =", cell_text)
            self.assertIn("simulation done, simulation time =", file_text)

    def test_disabled_logging_still_prints_progress(self):
        run_logs.configure_run_logging(False)
        output = io.StringIO()
        with redirect_stdout(output):
            run_logs.progress_print("still visible")
        self.assertEqual(output.getvalue(), "still visible\n")

    def test_requested_thermalization_and_excitation_lines(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with run_logs.simulation_progress(
                "Thermalization",
                ncells=30,
                rho=0.71,
                kT=0.8,
                nsteps=1_000,
            ):
                pass
            with run_logs.simulation_progress(
                "Excitation",
                ncells=30,
                rho=0.71,
                Source_kT=0.8,
                nsteps=2_000,
            ):
                pass

        cell_text = output.getvalue()
        self.assertIn(
            "Starting Thermalization Evolution (ncells = 30, rho = 0.71, "
            "kT = 0.8, nsteps = 1000)",
            cell_text,
        )
        self.assertIn(
            "Starting Excitation Evolution (ncells = 30, rho = 0.71, "
            "Source_kT = 0.8, nsteps = 2000)",
            cell_text,
        )


class PathTests(unittest.TestCase):
    def test_dt_validation_path_is_isolated_and_explicit(self):
        result = dt_validation.timestep_validation_paths(
            n_fcc_cells=30,
            target_rho=0.71,
            kT=0.8,
            dt=0.005,
            physical_time=5000.0,
            seed=2,
            base_folder="/tmp/dt_validation",
        )
        path_text = str(result["state_path"])
        self.assertIn("/tmp/dt_validation/", path_text)
        self.assertIn("dt_0.00500", path_text)
        self.assertIn("physical_time_5000.000", path_text)
        self.assertIn("seed_2", path_text)
        self.assertIn("physical_time_5000.000/sweep_manifest.json", str(
            result["manifest_path"]
        ))

    def test_dt_validation_preserves_physical_time(self):
        self.assertEqual(
            dt_validation.nsteps_for_physical_time(5000.0, 0.005),
            1_000_000,
        )
        self.assertEqual(
            dt_validation.nsteps_for_physical_time(5000.0, 0.01),
            500_000,
        )

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

    def test_excitation_paths_include_two_dt_segments(self):
        evolved = paths.excitation_evolved_paths(
            n_fcc_cells=30,
            source_rho=0.71,
            kT=0.8,
            source_nsteps=1_000_000,
            source_seed=1,
            method="velocity_rescale_com",
            radius=3.0,
            energy=4000.0,
            evolve_seed=1,
            dt2=0.005,
            nsteps2=100_000,
        )
        path_text = str(evolved["final_state_path"])
        self.assertEqual(evolved["final_state_path"].name, "excitation_final.gsd")
        self.assertIn("source_kT_0.800", path_text)
        self.assertIn("rho_0.710", path_text)
        self.assertNotIn("source_rho_0.710", path_text)
        self.assertNotIn("/kT_0.800/", path_text)
        self.assertIn("method_velocity_rescale_com", path_text)
        self.assertIn("energy_4000.000", path_text)
        self.assertIn("segment_1_dt_0.0005", path_text)
        self.assertIn("nsteps_200000", path_text)
        self.assertIn("segment_2_dt_0.005", path_text)
        self.assertIn("nsteps_100000", path_text)
        self.assertEqual(
            evolved["final_state_path"],
            evolved["segment_2"]["final_state_path"],
        )
        self.assertEqual(
            evolved["manifest_path"].name,
            "evolution_manifest.hdf5",
        )
        self.assertNotIn("source_phase_randomization", path_text)

    def test_excitation_paths_require_second_segment(self):
        with self.assertRaisesRegex(ValueError, "dt2 is required"):
            paths.excitation_evolved_paths(
                n_fcc_cells=30,
                source_rho=0.71,
                kT=0.8,
                source_nsteps=1_000_000,
                source_seed=1,
                method="velocity_rescale_com",
                radius=3.0,
                energy=4000.0,
                nsteps2=100_000,
                evolve_seed=1,
            )


class ExcitationEvolutionTests(unittest.TestCase):
    def _paths(self, root):
        return paths.excitation_evolved_paths(
            n_fcc_cells=10,
            source_rho=0.71,
            kT=0.8,
            source_nsteps=1_000,
            source_seed=1,
            method="velocity_rescale_com",
            radius=2.0,
            energy=100.0,
            dt1=0.0005,
            nsteps1=4,
            dt2=0.002,
            nsteps2=3,
            evolve_seed=1,
            base_folder=root,
        )

    @staticmethod
    def _write_log(path, timesteps, values):
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, mode="w") as hdf:
            hdf.require_group("hoomd-data/Simulation").create_dataset(
                "timestep",
                data=np.asarray(timesteps, dtype=np.int64),
            )
            hdf.require_group(
                "hoomd-data/md/compute/ThermodynamicQuantities"
            ).create_dataset(
                "pressure",
                data=np.asarray(values, dtype=float),
            )

    def test_stitched_log_deduplicates_boundary_and_builds_time(self):
        with TemporaryDirectory() as tmp:
            evolved = self._paths(tmp)
            self._write_log(
                evolved["segment_1"]["log_path"],
                [10, 12, 14],
                [1.0, 2.0, 3.0],
            )
            self._write_log(
                evolved["segment_2"]["log_path"],
                [14, 15, 17],
                [30.0, 4.0, 5.0],
            )
            excitation_evolution.write_evolution_manifest(
                evolved,
                status="complete",
                evolve_seed=1,
                segment_timing={
                    1: {"start_timestep": 10, "final_timestep": 14},
                    2: {"start_timestep": 14, "final_timestep": 17},
                },
            )

            stitched = excitation_evolution.read_stitched_log(evolved)

            np.testing.assert_array_equal(
                stitched["stitched"]["timestep"],
                [10, 12, 14, 15, 17],
            )
            np.testing.assert_allclose(
                stitched["stitched"]["elapsed_time"],
                [0.0, 0.001, 0.002, 0.004, 0.008],
            )
            np.testing.assert_allclose(
                stitched["hoomd-data"]["md"]["compute"][
                    "ThermodynamicQuantities"
                ]["pressure"],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            )
            self.assertTrue(
                stitched["stitched"]["boundary_duplicate_removed"]
            )

    def test_legacy_archive_is_dry_run_then_collision_safe_move(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "Excitation_Evolved_v3"
            archive = Path(tmp) / "Excitation_Evolved_v3_legacy_single_dt"
            source.mkdir()
            (source / "old_result.hdf5").touch()

            preview = excitation_evolution.archive_legacy_excitation_evolved(
                source_root=source,
                archive_root=archive,
                dry_run=True,
            )
            self.assertEqual(preview["status"], "would_move")
            self.assertTrue(source.exists())
            self.assertFalse(archive.exists())

            moved = excitation_evolution.archive_legacy_excitation_evolved(
                source_root=source,
                archive_root=archive,
                dry_run=False,
            )
            self.assertTrue(moved["moved"])
            self.assertTrue(source.is_dir())
            self.assertTrue((archive / "old_result.hdf5").exists())

    def test_active_root_rejects_unarchived_single_dt_logs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Excitation_Evolved_v3"
            legacy_folder = root / "FCC" / "old_run" / "seed_1"
            legacy_folder.mkdir(parents=True)
            (legacy_folder / "excitation_log.hdf5").touch()

            with self.assertRaisesRegex(
                RuntimeError,
                "still contains legacy single-dt results",
            ):
                excitation_evolution.ensure_two_segment_root(root)


class SpatialTests(unittest.TestCase):
    def test_result_dict_can_resolve_path_frame(self):
        fake_frame = make_frame(
            positions=[[0.0, 0.0, 0.0]],
            box_lengths=[1.0, 1.0, 1.0],
        )

        with patch.object(
            spatial,
            "_load_last_gsd_frame",
            return_value=fake_frame,
        ) as load_frame:
            snapshot = spatial.as_snapshot({
                "paths": {
                    "final_state_path": "final.gsd",
                },
            })

        self.assertIs(snapshot, fake_frame)
        load_frame.assert_called_once_with("final.gsd")

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


class ClassificationTests(unittest.TestCase):
    def test_voxel_classifier_uses_ncells_rule(self):
        frame = object()
        with patch(
            "md_Helpers.classification.compute_voxel_densities",
            return_value=(
                np.asarray([0.0, 0.5]),
                np.asarray([0, 1]),
                1.0,
            ),
        ) as compute_densities:
            result = classification.compute_voxel_fraction_phase_separation(
                frame,
                n_fcc_cells=30,
            )

        compute_densities.assert_called_once_with(frame, 12)
        self.assertEqual(result["nbins"], 12)
        self.assertEqual(result["nbins_source"], "n_fcc_cells_rule")

    def test_explicit_voxel_nbins_overrides_ncells_rule(self):
        frame = object()
        with patch(
            "md_Helpers.classification.compute_voxel_densities",
            return_value=(
                np.asarray([0.0, 0.5]),
                np.asarray([0, 1]),
                1.0,
            ),
        ) as compute_densities:
            result = classification.compute_voxel_fraction_phase_separation(
                frame,
                nbins=8,
                n_fcc_cells=30,
            )

        compute_densities.assert_called_once_with(frame, 8)
        self.assertEqual(result["nbins"], 8)
        self.assertEqual(result["nbins_source"], "explicit")


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

    def test_rethermalized_cavitation_can_skip_measurement(self):
        fake_result = {
            "status": "loaded_evolution",
            "paths": {
                "log_path": "cavitation_log.hdf5",
                "trajectory_path": "cavitation_trajectory.gsd",
            },
            "initial_result": {
                "source_phase_separation": {
                    "phase_separated": False,
                    "low_density_fraction": 0.0,
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
        fake_classification = SimpleNamespace(
            read_phase_method_attrs=lambda log_path, method: (
                {
                    "phase_separated": False,
                    "low_density_fraction": 0.0,
                },
                "metadata/classification/phase_separation/voxel",
            ),
        )

        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "md_Helpers.cavitation": fake_cavitation,
            "md_Helpers.classification": fake_classification,
        }), patch.object(
            cavitation_sweep_module.cavitation_analysis,
            "measure_cavitation_trajectory",
            side_effect=AssertionError(
                "rethermalized cavitation must not be measured"
            ),
        ):
            summary = run_cavitation_size_sweep(
                n_fcc_cells_values=[10],
                conditions=[(0.71, 0.8)],
                source_nsteps=100,
                evolve_nsteps=100,
                radius=2.0,
                summary_path=Path(tmp) / "summary.csv",
                summary_mode="interesting_only",
            )

        self.assertEqual(
            summary.loc[0, "run_status"],
            "cavitation_rethermalized",
        )
        self.assertTrue(bool(summary.loc[0, "thermalization_passed"]))
        self.assertEqual(summary.loc[0, "outcome"], "rethermalized")
        self.assertFalse(bool(summary.loc[0, "final_phase_separated"]))


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
    def test_nbins_for_ncells_uses_linear_rule(self):
        self.assertEqual(seitz.nbins_for_ncells(10), 6)
        self.assertEqual(seitz.nbins_for_ncells(20), 9)
        self.assertEqual(seitz.nbins_for_ncells(30), 12)

    def test_seitz_threshold_matches_intensive_formula(self):
        result = seitz.seitz_threshold(
            nc=75.0,
            uc=-140.0 / 75.0,
            u0=-2.0,
            p0=0.5,
            rho_c=0.75,
            rho_0=0.8,
        )

        expected = 75.0 * (
            (-140.0 / 75.0 - -2.0) + 0.5 * (1.0 / 0.75 - 1.0 / 0.8)
        )

        self.assertAlmostEqual(result, expected)

    def test_estimates_u0_and_p0_from_eos(self):
        eos = pd.DataFrame({
            "status": ["completed", "completed"],
            "phase_separated": [False, False],
            "kT": [0.8, 0.8],
            "actual_rho": [0.7, 0.8],
            "PE_per_particle_mean_last100": [-2.0, -3.0],
            "pressure_mean_last100": [0.2, 0.4],
        })

        self.assertAlmostEqual(
            seitz.estimate_u0_from_eos(
                eos,
                kT=0.8,
                target_rho=0.75,
            ),
            -2.5,
        )
        self.assertAlmostEqual(
            seitz.estimate_p0_from_eos(
                eos,
                kT=0.8,
                target_rho=0.75,
            ),
            0.3,
        )

    def test_eos_estimator_averages_duplicate_density_rows(self):
        eos = pd.DataFrame({
            "status": ["completed", "completed", "completed"],
            "phase_separated": [False, False, False],
            "kT": [0.8, 0.8, 0.8],
            "actual_rho": [0.7, 0.7, 0.8],
            "PE_per_particle_mean_last100": [-2.0, -4.0, -5.0],
            "pressure_mean_last100": [0.2, 0.4, 0.8],
        })

        self.assertAlmostEqual(
            seitz.estimate_u0_from_eos(
                eos,
                kT=0.8,
                target_rho=0.75,
            ),
            -4.0,
        )
        self.assertAlmostEqual(
            seitz.estimate_p0_from_eos(
                eos,
                kT=0.8,
                target_rho=0.75,
            ),
            0.55,
        )

    def test_estimates_use_default_eos_csv_when_table_is_omitted(self):
        eos = pd.DataFrame({
            "status": ["completed", "completed"],
            "phase_separated": [False, False],
            "kT": [0.8, 0.8],
            "actual_rho": [0.7, 0.8],
            "PE_per_particle_mean_last100": [-2.0, -3.0],
            "pressure_mean_last100": [0.2, 0.4],
        })

        with TemporaryDirectory() as tmp, patch.object(
            paths,
            "MASTER_CSVS_V3_ROOT",
            Path(tmp),
        ), patch.object(
            seitz,
            "DEFAULT_EOS_TABLE_NAME",
            "default_eos.csv",
        ):
            eos.to_csv(Path(tmp) / "default_eos.csv", index=False)

            self.assertAlmostEqual(
                seitz.estimate_u0_from_eos(
                    kT=0.8,
                    target_rho=0.75,
                ),
                -2.5,
            )
            self.assertAlmostEqual(
                seitz.estimate_p0_from_eos(
                    kT=0.8,
                    target_rho=0.75,
                ),
                0.3,
            )

    def test_eos_estimator_rejects_unknown_method(self):
        eos = pd.DataFrame({
            "status": ["completed", "completed"],
            "phase_separated": [False, False],
            "kT": [0.8, 0.8],
            "actual_rho": [0.7, 0.8],
            "PE_per_particle_mean_last100": [-2.0, -3.0],
            "pressure_mean_last100": [0.2, 0.4],
        })

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            seitz.estimate_u0_from_eos(
                eos,
                kT=0.8,
                target_rho=0.75,
                method="quadratic",
            )

    def test_extract_bubble_state_terms_uses_metadata_log_and_check(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        fake_check = {
            "fit": {
                "liquid_density": 0.8,
                "liquid_sigma_density": 0.04,
                "gas_density": 0.02,
                "bubble_radius_estimate": 3.0,
                "bubble_volume_estimate": 100.0,
            },
        }

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "bubble_log.hdf5"
            trajectory_path = Path(tmp) / "bubble_trajectory.gsd"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["kT"] = 0.8
                state.attrs["N"] = 999
                state.attrs["volume"] = 100.0

                creation = hdf.require_group("metadata/creation")
                creation.attrs["N_after"] = 75

                paths = hdf.require_group("metadata/paths")
                paths.attrs["trajectory_path"] = str(trajectory_path)
                paths.attrs["log_path"] = str(log_path)

                thermo = hdf.require_group(
                    "hoomd-data/md/compute/ThermodynamicQuantities"
                )
                thermo.create_dataset(
                    "potential_energy",
                    data=np.array([-120.0, -140.0]),
                )

            with patch(
                "md_Helpers.visualization.fit_and_animate_final_bubble",
                return_value=fake_check,
            ) as check_mock:
                result = seitz.extract_bubble_state_terms(
                    metadata_path=log_path,
                    n_last=None,
                    nbins=8,
                    nframes=3,
                    nskip=2,
                    plot=True,
                    estimate_reference=False,
                )

        check_mock.assert_called_once()
        self.assertEqual(
            check_mock.call_args.args[0],
            str(trajectory_path),
        )
        self.assertEqual(check_mock.call_args.kwargs["nbins"], 8)
        self.assertEqual(check_mock.call_args.kwargs["nframes"], 3)
        self.assertEqual(check_mock.call_args.kwargs["skip"], 2)
        self.assertTrue(check_mock.call_args.kwargs["show_histogram"])
        self.assertTrue(check_mock.call_args.kwargs["show_residuals"])
        self.assertAlmostEqual(result["Nc"], 75.0)
        self.assertAlmostEqual(result["Uc"], -140.0)
        self.assertAlmostEqual(result["uc"], -140.0 / 75.0)
        self.assertAlmostEqual(result["rho_c"], 0.75)
        self.assertAlmostEqual(result["rho_0"], 0.8)
        self.assertAlmostEqual(result["V"], 100.0)
        self.assertAlmostEqual(result["kT"], 0.8)
        self.assertEqual(result["voxel_nbins"], 8)
        self.assertEqual(result["voxel_nbins_source"], "explicit")

    def test_extract_bubble_state_terms_infers_nbins_from_ncells(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        fake_check = {"fit": {"liquid_density": 0.8}}

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "bubble_log.hdf5"
            trajectory_path = Path(tmp) / "bubble_trajectory.gsd"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["kT"] = 0.8
                state.attrs["volume"] = 100.0
                state.attrs["n_fcc_cells"] = 10

                creation = hdf.require_group("metadata/creation")
                creation.attrs["N_after"] = 75

                paths = hdf.require_group("metadata/paths")
                paths.attrs["trajectory_path"] = str(trajectory_path)
                paths.attrs["log_path"] = str(log_path)

                thermo = hdf.require_group(
                    "hoomd-data/md/compute/ThermodynamicQuantities"
                )
                thermo.create_dataset("potential_energy", data=np.array([-140.0]))

            with patch(
                "md_Helpers.visualization.fit_and_animate_final_bubble",
                return_value=fake_check,
            ) as check_mock:
                result = seitz.extract_bubble_state_terms(
                    metadata_path=log_path,
                    plot=False,
                    estimate_reference=False,
                )

        self.assertEqual(check_mock.call_args.kwargs["nbins"], 6)
        self.assertEqual(result["voxel_nbins"], 6)
        self.assertEqual(result["voxel_nbins_source"], "n_fcc_cells_rule")

    def test_extract_bubble_state_terms_adds_reference_values_and_q(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        eos = pd.DataFrame({
            "status": ["completed", "completed"],
            "phase_separated": [False, False],
            "kT": [0.8, 0.8],
            "actual_rho": [0.7, 0.9],
            "PE_per_particle_mean_last100": [-2.0, -4.0],
            "pressure_mean_last100": [0.2, 0.6],
        })
        fake_check = {"fit": {"liquid_density": 0.8}}

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "bubble_log.hdf5"
            trajectory_path = Path(tmp) / "bubble_trajectory.gsd"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["kT"] = 0.8
                state.attrs["volume"] = 100.0

                creation = hdf.require_group("metadata/creation")
                creation.attrs["N_after"] = 75

                paths_group = hdf.require_group("metadata/paths")
                paths_group.attrs["trajectory_path"] = str(trajectory_path)

                thermo = hdf.require_group(
                    "hoomd-data/md/compute/ThermodynamicQuantities"
                )
                thermo.create_dataset("potential_energy", data=np.array([-140.0]))

            with patch(
                "md_Helpers.visualization.fit_and_animate_final_bubble",
                return_value=fake_check,
            ):
                result = seitz.extract_bubble_state_terms(
                    metadata_path=log_path,
                    eos_table=eos,
                    plot=False,
                )

        expected_q = seitz.seitz_threshold(
            nc=75.0,
            uc=-140.0 / 75.0,
            u0=-3.0,
            p0=0.4,
            rho_c=0.75,
            rho_0=0.8,
        )

        self.assertAlmostEqual(result["u0"], -3.0)
        self.assertAlmostEqual(result["P0"], 0.4)
        self.assertAlmostEqual(result["p0"], 0.4)
        self.assertAlmostEqual(result["Q"], expected_q)
        self.assertAlmostEqual(result["q_seitz"], expected_q)

    def test_extract_bubble_state_terms_requires_n_after(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "bubble_log.hdf5"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["kT"] = 0.8
                state.attrs["N"] = 75
                state.attrs["volume"] = 100.0

                hdf.require_group("metadata/creation")

            with self.assertRaisesRegex(ValueError, "N_after"):
                seitz.extract_bubble_state_terms(
                    metadata_path=log_path,
                    plot=False,
                    estimate_reference=False,
                )

    def test_extract_bubble_state_terms_reports_missing_evolution_log(self):
        with TemporaryDirectory() as tmp:
            missing_log = Path(tmp) / "cavitation_log.hdf5"

            with self.assertRaisesRegex(
                FileNotFoundError,
                "cavitation evolution did not complete",
            ):
                seitz.extract_bubble_state_terms(
                    metadata_path=missing_log,
                    plot=False,
                    estimate_reference=False,
                )

    def test_extract_cavitation_result_terms_handles_skipped_cavitation(self):
        result = seitz.extract_bubble_state_terms({
            "status": "source_phase_separated",
            "paths": {
                "log_path": "missing_log.hdf5",
                "trajectory_path": "missing_trajectory.gsd",
            },
            "initial_result": {
                "source_phase_separation": {
                    "phase_separated": True,
                    "low_density_fraction": 0.044,
                },
            },
        })

        self.assertEqual(result["status"], "seitz_not_computed")
        self.assertEqual(
            result["cavitation_status"],
            "source_phase_separated",
        )
        self.assertTrue(np.isnan(result["Q"]))
        self.assertAlmostEqual(result["source_low_density_fraction"], 0.044)

    def test_extract_bubble_state_terms_defaults_to_five_frames_skip_five(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        fake_check = {"fit": {"liquid_density": 0.8}}

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "bubble_log.hdf5"
            trajectory_path = Path(tmp) / "bubble_trajectory.gsd"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["kT"] = 0.8
                state.attrs["volume"] = 100.0

                creation = hdf.require_group("metadata/creation")
                creation.attrs["N_after"] = 75

                paths = hdf.require_group("metadata/paths")
                paths.attrs["trajectory_path"] = str(trajectory_path)

                thermo = hdf.require_group(
                    "hoomd-data/md/compute/ThermodynamicQuantities"
                )
                thermo.create_dataset("potential_energy", data=np.array([-140.0]))

            with patch(
                "md_Helpers.visualization.fit_and_animate_final_bubble",
                return_value=fake_check,
            ) as check_mock:
                seitz.extract_bubble_state_terms(
                    metadata_path=log_path,
                    plot=False,
                    estimate_reference=False,
                )

        self.assertEqual(check_mock.call_args.kwargs["nframes"], 5)
        self.assertEqual(check_mock.call_args.kwargs["skip"], 5)
        self.assertFalse(check_mock.call_args.kwargs["show_histogram"])


class VisualizationTests(unittest.TestCase):
    def test_fit_and_animate_final_bubble_reports_no_bubble_when_rethermalized(self):
        try:
            from md_Helpers import visualization
        except ImportError as error:
            raise unittest.SkipTest("visualization dependencies are missing") from error

        fake_fit = {
            "frame_indices": [1, 2],
            "bubble_radius_estimate": 0.0,
            "bubble_volume_estimate": 0.0,
            "bubble_volume_fraction": 0.0,
        }
        fake_frame = make_frame(
            positions=[[0, 0, 0], [1, 1, 1]],
            box_lengths=[10, 10, 10],
        )
        fake_trajectory = [fake_frame]

        with patch(
            "md_Helpers.voxel_fit.fit_trajectory_tail_voxel_histogram",
            return_value=fake_fit,
        ), patch(
            "md_Helpers.visualization.plot_voxel_mixture_fit",
        ), patch(
            "gsd.hoomd.open",
            return_value=SimpleNamespace(
                __enter__=lambda self: fake_trajectory,
                __exit__=lambda self, exc_type, exc, tb: False,
            ),
        ), patch(
            "md_Helpers.classification.compute_voxel_fraction_phase_separation",
            return_value={"phase_separated": False},
        ):
            result = visualization.fit_and_animate_final_bubble(
                "trajectory.gsd",
                show_histogram=True,
            )

        self.assertFalse(result["has_bubble"])
        self.assertEqual(result["outcome"], "rethermalized")
        self.assertIsNone(result["animation"])
        self.assertEqual(
            result["message"],
            "Final state is not phase separated; no bubble found.",
        )


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

    def test_builds_results_inventory_csv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Cavitation_Evolved_v3"
            run = root / "FCC" / "n_cells_20" / "run_1"
            run.mkdir(parents=True)

            trajectory_path = run / "cavitation_trajectory.gsd"
            log_path = run / "cavitation_log.hdf5"
            trajectory_path.write_bytes(b"gsd")
            log_path.write_bytes(b"hdf5")

            output_path = Path(tmp) / "inventory.csv"
            table = master_csv.build_results_inventory_csv(
                roots={
                    "cavitation_evolved": root,
                    "missing_family": Path(tmp) / "missing",
                },
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(table), 3)
            self.assertEqual(
                set(table["file_role"]),
                {"trajectory", "log", "missing_root"},
            )
            self.assertEqual(
                table.loc[
                    table["filename"] == "cavitation_trajectory.gsd",
                    "result_family",
                ].item(),
                "cavitation_evolved",
            )

            summary = master_csv.summarize_results_inventory(table)
            self.assertEqual(int(summary["n_files"].sum()), 2)
            self.assertIn("trajectory", set(summary["file_role"]))

    def test_seitz_master_leaves_terms_empty_for_rethermalized_run(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Cavitation_Evolved_v3"
            run = root / "FCC" / "run_1"
            run.mkdir(parents=True)
            log_path = run / "cavitation_log.hdf5"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["n_fcc_cells"] = 30
                state.attrs["source_rho"] = 0.6
                state.attrs["kT"] = 1.0

                run_group = hdf.require_group("metadata/run")
                run_group.attrs["nsteps"] = 100_000
                run_group.attrs["seed"] = 1

                source = hdf.require_group("metadata/source")
                source.attrs["source_N"] = 108000
                source.attrs["source_rho"] = 0.6
                source.attrs["source_kT"] = 1.0
                source.attrs["source_nsteps"] = 1_000_000
                source.attrs["source_seed"] = 1

                creation = hdf.require_group("metadata/creation")
                creation.attrs["radius"] = 7.0
                creation.attrs["bubble_seed"] = 1

                paths_group = hdf.require_group("metadata/paths")
                paths_group.attrs["trajectory_path"] = str(
                    run / "cavitation_trajectory.gsd"
                )

                voxel = hdf.require_group(
                    "metadata/classification/phase_separation/voxel"
                )
                voxel.attrs["phase_separated"] = False

            table = master_csv.build_seitz_master_csv(
                root=root,
                output_path=Path(tmp) / "seitz.csv",
            )

        self.assertEqual(len(table), 1)
        self.assertEqual(table.loc[0, "status"], "rethermalized")
        self.assertFalse(bool(table.loc[0, "final_phase_separated"]))
        self.assertTrue(pd.isna(table.loc[0, "Q"]))
        self.assertTrue(pd.isna(table.loc[0, "N_cav"]))

    def test_seitz_master_computes_terms_for_phase_separated_run(self):
        try:
            import h5py
        except ImportError as error:
            raise unittest.SkipTest("h5py is not installed") from error

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "Cavitation_Evolved_v3"
            run = root / "FCC" / "run_1"
            run.mkdir(parents=True)
            log_path = run / "cavitation_log.hdf5"

            with h5py.File(log_path, mode="w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["n_fcc_cells"] = 30
                state.attrs["source_rho"] = 0.6
                state.attrs["kT"] = 1.0

                run_group = hdf.require_group("metadata/run")
                run_group.attrs["nsteps"] = 100_000
                run_group.attrs["seed"] = 2

                source = hdf.require_group("metadata/source")
                source.attrs["source_N"] = 108000
                source.attrs["source_rho"] = 0.6
                source.attrs["source_kT"] = 1.0
                source.attrs["source_nsteps"] = 1_000_000
                source.attrs["source_seed"] = 1

                creation = hdf.require_group("metadata/creation")
                creation.attrs["radius"] = 7.0
                creation.attrs["bubble_seed"] = 3

                paths_group = hdf.require_group("metadata/paths")
                paths_group.attrs["trajectory_path"] = str(
                    run / "cavitation_trajectory.gsd"
                )

                voxel = hdf.require_group(
                    "metadata/classification/phase_separation/voxel"
                )
                voxel.attrs["phase_separated"] = True

            with patch(
                "md_Helpers.seitz.extract_bubble_state_terms",
                return_value={
                    "Nc": 100.0,
                    "uc": -1.2,
                    "rho_0": 0.61,
                    "rho_c": 0.02,
                    "P0": 0.4,
                    "u0": -2.0,
                    "Q": 15.0,
                },
            ) as seitz_mock:
                table = master_csv.build_seitz_master_csv(
                    root=root,
                    output_path=Path(tmp) / "seitz.csv",
                )

        seitz_mock.assert_called_once()
        self.assertEqual(table.loc[0, "status"], "seitz_computed")
        self.assertTrue(bool(table.loc[0, "final_phase_separated"]))
        self.assertAlmostEqual(table.loc[0, "N_cav"], 100.0)
        self.assertAlmostEqual(table.loc[0, "Q"], 15.0)
        self.assertEqual(table.loc[0, "bubble_seed"], 3)


class EosSweepTests(unittest.TestCase):
    def test_pressure_stop_region_uses_old_stop_defaults(self):
        self.assertEqual(
            eos_sweep.pressure_stop_region(-0.04),
            "below_stop",
        )
        self.assertEqual(
            eos_sweep.pressure_stop_region(0.10),
            "inside_stops",
        )
        self.assertEqual(
            eos_sweep.pressure_stop_region(0.20),
            "above_stop",
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
