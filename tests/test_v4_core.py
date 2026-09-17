from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np

from md_Helpers.database import (
    SQLiteRunDatabase,
    cavitation_dataframe,
    display_master_table,
    master_dataframe,
    thermalization_dataframe,
)
from md_Helpers.cavitation import (
    CavitationConfig,
    _source_context,
    cavitation_frame_schedule,
    shift_and_mask_positions,
)
from md_Helpers.analysis import thermodynamic_summary
from md_Helpers.lattices import build_fcc_lattice
from md_Helpers.paths import ProjectPaths
from md_Helpers.run_analysis import RunAnalysis, open_run
from md_Helpers.run_management import delete_run
from md_Helpers.signatures import create_run_signature
from md_Helpers.storage import StateData
from md_Helpers.thermalization import (
    CloneRescaleThermalizationConfig,
    ThermalizationConfig,
    _clone_request_context,
    _inherited_clone_config,
    _single_axis_final_box,
    clone_final_density_is_acceptable,
    thermalization_log_steps,
    thermalization_phase_frame_schedule,
)
from md_Helpers.voxel_fit import (
    conditional_phase_fit,
    phase_fit_frame_indices,
    phase_fit_sql_values,
)


class SignatureTests(unittest.TestCase):
    def test_thermalization_default_timestep(self):
        config = ThermalizationConfig(4, 0.5, 100, seed=1)
        self.assertEqual(config.dt, 0.002)

    def test_signature_is_stable_and_order_independent(self):
        first = create_run_signature({"n_cells": 45, "rho": 0.5, "kT": 0.9})
        second = create_run_signature({"kT": 0.9, "rho": 0.5, "n_cells": 45})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_simulation_input_changes_signature(self):
        first = ThermalizationConfig(4, 0.5, 100, seed=1)
        second = ThermalizationConfig(4, 0.5, 100, seed=2)
        self.assertNotEqual(first.run_signature, second.run_signature)

    def test_clone_signature_uses_source_frame_density_and_duration(self):
        request = CloneRescaleThermalizationConfig(
            source_run_id="20260903214936",
            final_density=0.4,
            nsteps=200_000,
        )
        signature = request.run_signature(source_frame_id=2000)
        self.assertNotEqual(signature, request.run_signature(source_frame_id=1999))
        self.assertNotEqual(
            signature,
            CloneRescaleThermalizationConfig(
                source_run_id=request.source_run_id,
                final_density=0.5,
                nsteps=request.nsteps,
            ).run_signature(source_frame_id=2000),
        )

    def test_clone_signature_distinguishes_nvt_and_nve(self):
        nvt = CloneRescaleThermalizationConfig("source", 0.4, 200_000)
        nve = CloneRescaleThermalizationConfig(
            "source", 0.4, 200_000, ensemble="nve"
        )
        self.assertNotEqual(nvt.run_signature(5), nve.run_signature(5))
        self.assertEqual(nve.signature_parameters(5)["ensemble"], "NVE")

    def test_clone_rejects_unknown_ensemble(self):
        config = CloneRescaleThermalizationConfig(
            "source", 0.4, 200_000, ensemble="NPT"
        )
        with self.assertRaisesRegex(ValueError, "NVT.*NVE"):
            config.validate()

    def test_constant_volume_rate_signature_records_axis(self):
        request = CloneRescaleThermalizationConfig(
            "source",
            0.4,
            200_000,
            ensemble="NVE",
            resize_mode="linear_volume_axis",
            resize_axis="z",
        )
        request.validate()
        parameters = request.signature_parameters(5)
        self.assertEqual(parameters["resize_axis"], "z")
        self.assertEqual(
            parameters["density_schedule"],
            "linear_volume_single_axis_v2",
        )

    def test_constant_volume_rate_signature_distinguishes_axes(self):
        x_axis = CloneRescaleThermalizationConfig(
            "source", 0.4, 200_000,
            resize_mode="linear_volume_axis", resize_axis="x",
        )
        y_axis = CloneRescaleThermalizationConfig(
            "source", 0.4, 200_000,
            resize_mode="linear_volume_axis", resize_axis="y",
        )
        self.assertNotEqual(x_axis.run_signature(5), y_axis.run_signature(5))

    def test_clone_rejects_unknown_resize_axis(self):
        request = CloneRescaleThermalizationConfig(
            "source", 0.4, 200_000,
            resize_mode="linear_volume_axis", resize_axis="q",
        )
        with self.assertRaisesRegex(ValueError, "'x'.*'y'.*'z'"):
            request.validate()

    def test_cavitation_signature_records_location_seed(self):
        centered = CavitationConfig("source", 2.0, 100_000)
        random = CavitationConfig(
            "source", 2.0, 100_000,
            random_location=True, location_seed=7,
        )
        self.assertNotEqual(centered.run_signature(5), random.run_signature(5))

    def test_random_cavitation_requires_seed(self):
        with self.assertRaisesRegex(ValueError, "location_seed is required"):
            CavitationConfig(
                "source", 2.0, 100_000, random_location=True
            ).validate()


class CavitationScheduleAndMaskTests(unittest.TestCase):
    def test_bulk_and_terminal_frame_schedule_is_a_union(self):
        schedule = cavitation_frame_schedule(100_000, 1_000)
        self.assertEqual(
            [item["log_ordinal"] for item in schedule],
            [20, 40, 60, 70, 80, 90, 100],
        )
        self.assertEqual(
            [item["log_ordinal"] for item in schedule if item["phase_frame"]],
            [60, 70, 80, 90, 100],
        )

    def test_centered_mask_removes_inside_and_keeps_velocities_addressable(self):
        result = shift_and_mask_positions(
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            box=np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
            mask_radius=1.0,
        )
        self.assertEqual(result["particles_removed"], 1)
        np.testing.assert_array_equal(result["keep_mask"], [False, True])
        np.testing.assert_allclose(result["sampled_center"], [0.0, 0.0, 0.0])

    def test_random_center_is_reproducible_and_shifted_to_origin(self):
        positions = np.array([[1.0, 1.0, 1.0], [-3.0, -3.0, -3.0]])
        kwargs = dict(
            positions=positions,
            box=np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
            mask_radius=0.5,
            random_location=True,
            location_seed=17,
        )
        # Put one particle at the generated center so the mask is non-empty.
        center = np.random.default_rng(17).uniform(-5.0, 5.0, 3)
        kwargs["positions"] = np.vstack([center, positions[1]])
        first = shift_and_mask_positions(**kwargs)
        second = shift_and_mask_positions(**kwargs)
        np.testing.assert_allclose(first["sampled_center"], center)
        np.testing.assert_allclose(
            first["sampled_center"], second["sampled_center"]
        )
        np.testing.assert_allclose(first["shifted_positions"][0], 0.0)

    def test_mask_diameter_cannot_exceed_eighty_five_percent(self):
        with self.assertRaisesRegex(ValueError, "85%"):
            shift_and_mask_positions(
                positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
                box=np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]),
                mask_radius=4.26,
            )


class LatticeTests(unittest.TestCase):
    def test_fcc_count_and_density(self):
        lattice = build_fcc_lattice(n_cells=3, density=0.5)
        self.assertEqual(lattice.n_particles, 4 * 3**3)
        self.assertEqual(lattice.positions.shape, (4 * 3**3, 3))
        self.assertAlmostEqual(lattice.actual_density, 0.5)


class SingleAxisResizeTests(unittest.TestCase):
    def test_changes_only_selected_length_and_reaches_volume(self):
        initial = [8.0, 9.0, 10.0, 0.1, 0.2, 0.3]
        final = _single_axis_final_box(initial, final_volume=900.0, axis="y")
        self.assertEqual(final, [8.0, 11.25, 10.0, 0.1, 0.2, 0.3])
        self.assertAlmostEqual(np.prod(final[:3]), 900.0)


class PhaseFitPolicyTests(unittest.TestCase):
    def test_selects_five_terminal_saved_frames(self):
        self.assertEqual(phase_fit_frame_indices(100), [95, 96, 97, 98, 99])

    def test_six_frame_trajectory_excludes_initial_frame(self):
        self.assertEqual(phase_fit_frame_indices(6), [1, 2, 3, 4, 5])

    def test_incomplete_trajectory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one initial frame"):
            phase_fit_frame_indices(5)

    @patch("md_Helpers.voxel_fit.fit_trajectory_voxel_mixture")
    def test_homogeneous_state_skips_fit_and_leaves_values_null(self, fit):
        result = conditional_phase_fit(
            {"phase_separated": False},
            trajectory_path="unused.gsd",
            n_cells=4,
        )
        fit.assert_not_called()
        self.assertEqual(result["status"], "Skipped_Homogeneous")
        sql_values = phase_fit_sql_values(result)
        for column in [
            "rho_liquid",
            "rho_liquid_unc",
            "rho_gas",
            "rho_gas_unc",
            "V_liquid",
            "V_liquid_unc",
            "V_gas",
            "V_gas_unc",
        ]:
            self.assertIsNone(sql_values[column])


class ThermalizationFrameScheduleTests(unittest.TestCase):
    def test_clone_final_density_accepts_point_one_percent_tolerance(self):
        self.assertTrue(clone_final_density_is_acceptable(0.6000004, 0.6))
        self.assertFalse(clone_final_density_is_acceptable(0.6007, 0.6))

    def test_selects_logs_60_70_80_90_100(self):
        schedule = thermalization_phase_frame_schedule(
            nsteps=100_000,
            log_period=1_000,
        )
        self.assertEqual(
            [item["log_ordinal"] for item in schedule],
            [60, 70, 80, 90, 100],
        )
        self.assertEqual(
            [item["run_step"] for item in schedule],
            [60_000, 70_000, 80_000, 90_000, 100_000],
        )

    def test_final_step_is_always_a_log_point(self):
        self.assertEqual(
            thermalization_log_steps(2_500, 1_000),
            [1_000, 2_000, 2_500],
        )

    def test_rejects_too_few_evolved_logs(self):
        with self.assertRaisesRegex(ValueError, "at least 41"):
            thermalization_phase_frame_schedule(
                nsteps=40_000,
                log_period=1_000,
            )


class ThermodynamicSummaryTests(unittest.TestCase):
    def test_pressure_statistics_are_finite(self):
        summary = thermodynamic_summary(
            run_steps=np.array([0, 100, 200]),
            pressure=np.array([np.nan, 5.0, 7.0]),
            potential_energy=np.array([-10.0, -12.0, -14.0]),
            n_particles=2,
            n_last=3,
        )
        self.assertEqual(summary["Pressure_Mean"], 6.0)
        self.assertAlmostEqual(summary["Pressure_Std"], np.sqrt(2.0))

    def test_missing_pressure_fails_instead_of_writing_sql_null(self):
        with self.assertRaisesRegex(RuntimeError, "no finite samples"):
            thermodynamic_summary(
                run_steps=np.array([0, 100]),
                pressure=np.array([np.nan, np.nan]),
                potential_energy=np.array([-10.0, -11.0]),
                n_particles=2,
            )


class PathTests(unittest.TestCase):
    def test_two_canonical_run_files(self):
        paths = ProjectPaths("/tmp/example").for_run(
            "Thermalization",
            "20260902120000",
        )
        self.assertEqual(
            paths.relative_directory,
            Path("Thermalization/20260902120000"),
        )
        self.assertEqual(paths.trajectory.name, "trajectory.gsd")
        self.assertEqual(paths.hdf5.name, "run.hdf5")


class RunPlotPolicyTests(unittest.TestCase):
    @patch("md_Helpers.run_analysis.plot_log_dataframe", return_value="figure")
    def test_fresh_lattice_skips_first_ten_pressure_and_pe_points(self, plot):
        run = RunAnalysis.__new__(RunAnalysis)
        run.sim_type = "Thermalization"
        run.state_row = {"Clone_Run_ID": None}
        run.logs_dataframe = lambda: "logs"

        result = run.plot_logs(
            quantities=["pressure", "potential_energy_per_particle"]
        )

        self.assertEqual(result, "figure")
        self.assertEqual(
            plot.call_args.kwargs["skip_initial_by_quantity"],
            {
                "pressure": 10,
                "potential_energy_per_particle": 10,
                "PE_per_particle": 10,
            },
        )

    @patch("md_Helpers.run_analysis.plot_log_dataframe", return_value="figure")
    def test_cloned_state_keeps_all_log_points(self, plot):
        run = RunAnalysis.__new__(RunAnalysis)
        run.sim_type = "Thermalization"
        run.state_row = {"Clone_Run_ID": "20260903214936"}
        run.logs_dataframe = lambda: "logs"

        run.plot_logs(quantities=["pressure"])

        self.assertEqual(
            plot.call_args.kwargs["skip_initial_by_quantity"],
            {},
        )


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database = SQLiteRunDatabase(
            Path(self.temp_directory.name) / "mdsims.sqlite3"
        )
        self.database.initialize()

    def tearDown(self):
        self.temp_directory.cleanup()

    def _create_complete_thermalization(self) -> tuple[str, ProjectPaths]:
        run_id = self.database.reserve_run_id()
        self.database.update_master(
            run_id,
            Run_Signature="e" * 64,
            N_Cells=4,
            Nsteps=100,
            Sim_Type="Thermalization",
            Status="Running",
        )
        self.database.complete_thermalization(
            run_id,
            thermalization={
                "File_Location": f"Thermalization/{run_id}",
                "N_Cells": 4,
                "Therm_kT": 0.9,
                "Therm_Seed": 1,
                "Density_Start": 0.5,
                "Density_End": 0.5,
                "BoxLength_Start": 10.0,
                "BoxLength_End": 10.0,
                "dt": 0.005,
                "Nsteps": 100,
                "This_LJ_Time": 0.5,
                "Cumulative_LJ_Time": 0.5,
                "Ensemble": "NVT",
                "T_Set": 0.9,
                "LJ_r_cut": 2.5,
                "LJ_r_on": 2.0,
                "LJ_Mode": "xplor",
                "Phase_Separation_Status": "Not_Separated",
                "Phase_Separation_Method": "voxel_histogram",
                "Phase_Separation_Method_Version": "test",
                "Phase_Fit_Status": "Not_Run",
                "Summary_Start_Step": 0,
                "Summary_End_Step": 100,
                "Summary_Num_Samples": 2,
                "Num_Frames": 6,
            },
            master={"Status": "Complete", "Current_Nstep": 100},
        )
        paths = ProjectPaths(self.temp_directory.name)
        run_paths = paths.for_run("Thermalization", run_id)
        run_paths.directory.mkdir(parents=True)
        run_paths.trajectory.write_bytes(b"trajectory")
        run_paths.hdf5.write_bytes(b"hdf5")
        return run_id, paths

    def _create_complete_cavitation(
        self,
    ) -> tuple[str, str, ProjectPaths]:
        source_run_id, paths = self._create_complete_thermalization()
        run_id = f"{int(source_run_id) + 1:014d}"
        with self.database.connection() as connection:
            connection.execute(
                "INSERT INTO MD_Master (Run_ID) VALUES (?)",
                (run_id,),
            )
        self.database.update_master(
            run_id,
            Run_Signature="9" * 64,
            N_Cells=4,
            Nsteps=41_000,
            Sim_Type="Cavitation",
            Status="Running",
        )
        self.database.complete_cavitation(
            run_id,
            cavitation={
                "File_Location": f"Cavitation/{run_id}",
                "Source_Run_ID": source_run_id,
                "Source_Frame_ID": 5,
                "N_Cells": 4,
                "Therm_kT": 0.9,
                "Therm_Seed": 1,
                "Source_Density": 0.5,
                "Initial_Density": 0.49,
                "BoxLength": 10.0,
                "Mask_Radius": 1.0,
                "Random_Location": 0,
                "Location_Seed": None,
                "dt": 0.005,
                "Nsteps": 41_000,
                "This_LJ_Time": 205.0,
                "Cumulative_LJ_Time": 205.5,
                "Ensemble": "NVT",
                "T_Set": 0.9,
                "P_Set": None,
                "LJ_r_cut": 2.5,
                "LJ_r_on": 2.0,
                "LJ_Mode": "xplor",
                "Phase_Separation_Status": "Not_Separated",
                "Phase_Separation_Method": "voxel_histogram",
                "Phase_Separation_Method_Version": "test",
                "Phase_Fit_Status": "Skipped_Homogeneous",
                "Summary_Start_Step": 0,
                "Summary_End_Step": 41_000,
                "Summary_Num_Samples": 42,
                "Num_Frames": 7,
            },
            master={"Status": "Complete", "Current_Nstep": 41_000},
        )
        run_paths = paths.for_run("Cavitation", run_id)
        run_paths.directory.mkdir(parents=True)
        run_paths.trajectory.write_bytes(b"trajectory")
        run_paths.hdf5.write_bytes(b"hdf5")
        return source_run_id, run_id, paths

    def test_reserve_then_populate_master(self):
        run_id = self.database.reserve_run_id()
        reserved = self.database.get_run(run_id)
        self.assertIsNone(reserved["Run_Signature"])
        self.assertIsNone(reserved["Status"])

        self.database.update_master(
            run_id,
            Run_Signature="a" * 64,
            N_Cells=4,
            Nsteps=100,
            Current_Nstep=0,
            ElapsedTime=0.0,
            Sim_Type="Thermalization",
            Status="Initializing",
        )
        match = self.database.check_run_exists("a" * 64)
        self.assertEqual(match["Run_ID"], run_id)

        table = master_dataframe(self.database)
        self.assertEqual(table.loc[0, "Run_ID"], run_id)
        self.assertEqual(len(table.columns), 14)

    def test_initialize_adds_n_cells_to_legacy_thermalization_table(self):
        legacy_path = Path(self.temp_directory.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute(
                "CREATE TABLE Thermalization (Run_ID TEXT PRIMARY KEY)"
            )

        legacy_database = SQLiteRunDatabase(legacy_path)
        legacy_database.initialize()

        with legacy_database.connection() as connection:
            columns = [
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(Thermalization)"
                ).fetchall()
            ]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertIn("N_Cells", columns)
        self.assertEqual(version, 3)

    def test_complete_and_query_cavitation(self):
        source_run_id, run_id, _ = self._create_complete_cavitation()
        row = self.database.get_cavitation(run_id)
        self.assertEqual(row["Source_Run_ID"], source_run_id)
        self.assertEqual(row["Initial_Density"], 0.49)
        table = cavitation_dataframe(
            self.database,
            Source_Run_ID=source_run_id,
            Mask_Radius=(0.5, 1.5),
        )
        self.assertEqual(table["Run_ID"].tolist(), [run_id])

    def test_phase_separated_source_returns_structured_skip(self):
        source_run_id, _ = self._create_complete_thermalization()
        self.database.update_thermalization(
            source_run_id,
            Phase_Separation_Status="Separated",
        )
        result = _source_context(
            CavitationConfig(source_run_id, 1.0, 41_000),
            self.database,
        )
        self.assertTrue(result["skipped"])
        self.assertEqual(result["skip_reason"], "source_phase_separated")

    def test_source_deletion_is_blocked_by_cavitation_dependency(self):
        source_run_id, cavitation_run_id, paths = (
            self._create_complete_cavitation()
        )
        with self.assertRaisesRegex(
            RuntimeError,
            f"Cavitation:{cavitation_run_id}",
        ):
            delete_run(
                source_run_id,
                dry_run=False,
                confirm_run_id=source_run_id,
                project_paths=paths,
                database=self.database,
            )
        self.assertIsNotNone(self.database.get_run(source_run_id))

    def test_open_run_loads_cavitation_sql_row(self):
        _, run_id, paths = self._create_complete_cavitation()
        run = open_run(run_id, project_paths=paths, database=self.database)
        self.assertEqual(run.sim_type, "Cavitation")
        self.assertEqual(run.state_row["Mask_Radius"], 1.0)

    def test_complete_thermalization_updates_both_tables(self):
        run_id = self.database.reserve_run_id()
        self.database.update_master(
            run_id,
            Run_Signature="b" * 64,
            N_Cells=4,
            Nsteps=100,
            Sim_Type="Thermalization",
            Status="Running",
        )
        self.database.complete_thermalization(
            run_id,
            thermalization={
                "File_Location": f"Thermalization/{run_id}",
                "N_Cells": 4,
                "Therm_kT": 0.9,
                "Therm_Seed": 1,
                "Density_Start": 0.5,
                "Density_End": 0.5,
                "BoxLength_Start": 10.0,
                "BoxLength_End": 10.0,
                "dt": 0.005,
                "Nsteps": 100,
                "This_LJ_Time": 0.5,
                "Cumulative_LJ_Time": 0.5,
                "Ensemble": "NVT",
                "T_Set": 0.9,
                "LJ_r_cut": 2.5,
                "LJ_r_on": 2.0,
                "LJ_Mode": "xplor",
                "Phase_Separation_Status": "Not_Separated",
                "Phase_Separation_Method": "voxel_histogram",
                "Phase_Separation_Method_Version": "test",
                "Phase_Fit_Status": "Not_Run",
                "Summary_Start_Step": 0,
                "Summary_End_Step": 100,
                "Summary_Num_Samples": 2,
                "Num_Frames": 2,
            },
            master={"Status": "Complete", "Current_Nstep": 100},
        )
        self.assertEqual(self.database.get_run(run_id)["Status"], "Complete")
        self.database.update_thermalization(
            run_id,
            Pressure_Mean=6.0,
            Pressure_Std=0.2,
            Pressure_SEM=0.02,
        )
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*), Pressure_Mean FROM Thermalization WHERE Run_ID = ?",
                (run_id,),
            ).fetchone()
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], 6.0)

        table = thermalization_dataframe(
            self.database,
            Therm_kT=(0.8, 1.0),
            Nsteps=[100, 200],
            Phase_Separation_Status="Not_Separated",
        )
        self.assertEqual(len(table), 1)
        self.assertEqual(table.loc[0, "Run_ID"], run_id)
        self.assertEqual(
            list(table.columns)[2:5],
            ["Clone_Run_ID", "Clone_Frame_ID", "N_Cells"],
        )

    def test_backfills_thermalization_n_cells_from_master(self):
        run_id, _ = self._create_complete_thermalization()
        self.database.update_thermalization(run_id, N_Cells=3)

        result = self.database.backfill_thermalization_n_cells()

        self.assertEqual(result["thermalization_rows"], 1)
        self.assertEqual(result["rows_copied"], 1)
        self.assertEqual(result["rows_remaining_null"], 0)
        self.assertEqual(
            self.database.get_thermalization(run_id)["N_Cells"],
            4,
        )

    def test_delete_run_previews_then_deletes_files_and_both_rows(self):
        run_id, paths = self._create_complete_thermalization()
        run_directory = paths.for_run("Thermalization", run_id).directory

        preview = delete_run(
            run_id,
            project_paths=paths,
            database=self.database,
        )
        self.assertTrue(preview["dry_run"])
        self.assertTrue(preview["trajectory_exists"])
        self.assertTrue(preview["hdf5_exists"])
        self.assertTrue(run_directory.exists())
        self.assertIsNotNone(self.database.get_run(run_id))

        result = delete_run(
            run_id,
            dry_run=False,
            confirm_run_id=run_id,
            project_paths=paths,
            database=self.database,
        )
        self.assertTrue(result["directory_deleted"])
        self.assertEqual(result["thermalization_rows_deleted"], 1)
        self.assertEqual(result["master_rows_deleted"], 1)
        self.assertFalse(run_directory.exists())
        self.assertIsNone(self.database.get_thermalization(run_id))
        self.assertIsNone(self.database.get_run(run_id))

    def test_delete_run_requires_exact_confirmation(self):
        run_id, paths = self._create_complete_thermalization()

        with self.assertRaisesRegex(ValueError, "confirm_run_id"):
            delete_run(
                run_id,
                dry_run=False,
                confirm_run_id="wrong",
                project_paths=paths,
                database=self.database,
            )

        self.assertTrue(
            paths.for_run("Thermalization", run_id).directory.exists()
        )
        self.assertIsNotNone(self.database.get_run(run_id))

    def test_delete_failed_run_without_directory_deletes_sql_record(self):
        run_id = self.database.reserve_run_id()
        self.database.update_master(
            run_id,
            Run_Signature="f" * 64,
            N_Cells=40,
            Nsteps=25_000,
            Current_Nstep=0,
            Sim_Type="Thermalization",
            Status="Failed",
            Stop_Reason="exception",
            Status_Message="ValueError: invalid thermalization schedule",
        )
        paths = ProjectPaths(self.temp_directory.name)

        preview = delete_run(
            run_id,
            project_paths=paths,
            database=self.database,
        )
        self.assertFalse(preview["directory_exists"])

        result = delete_run(
            run_id,
            dry_run=False,
            confirm_run_id=run_id,
            project_paths=paths,
            database=self.database,
        )

        self.assertFalse(result["directory_deleted"])
        self.assertEqual(result["thermalization_rows_deleted"], 0)
        self.assertEqual(result["master_rows_deleted"], 1)
        self.assertIsNone(self.database.get_run(run_id))

    def test_delete_complete_run_without_directory_still_refuses(self):
        run_id, paths = self._create_complete_thermalization()
        run_directory = paths.for_run("Thermalization", run_id).directory
        for path in run_directory.iterdir():
            path.unlink()
        run_directory.rmdir()

        with self.assertRaisesRegex(FileNotFoundError, "no SQL rows"):
            delete_run(
                run_id,
                dry_run=False,
                confirm_run_id=run_id,
                project_paths=paths,
                database=self.database,
            )

        self.assertIsNotNone(self.database.get_run(run_id))
        self.assertIsNotNone(self.database.get_thermalization(run_id))

    def test_open_run_is_a_lazy_sql_and_path_lookup(self):
        run_id = self.database.reserve_run_id()
        self.database.update_master(
            run_id,
            Run_Signature="c" * 64,
            Sim_Type="Thermalization",
            Status="Running",
        )
        paths = ProjectPaths(self.temp_directory.name)
        run = open_run(run_id, project_paths=paths, database=self.database)
        self.assertEqual(run.run_id, run_id)
        self.assertEqual(
            run.trajectory_path,
            paths.top_directory / "Thermalization" / run_id / "trajectory.gsd",
        )
        self.assertFalse(run.trajectory_path.exists())

    def test_clone_request_uses_completed_source_and_builds_note(self):
        source_run_id = self.database.reserve_run_id()
        self.database.update_master(
            source_run_id,
            Run_Signature="d" * 64,
            N_Cells=4,
            Nsteps=1_000,
            Sim_Type="Thermalization",
            Status="Running",
        )
        self.database.complete_thermalization(
            source_run_id,
            thermalization={
                "File_Location": f"Thermalization/{source_run_id}",
                "N_Cells": 4,
                "Therm_kT": 0.9,
                "Therm_Seed": 7,
                "Density_Start": 0.5,
                "Density_End": 0.5,
                "BoxLength_Start": 8.0,
                "BoxLength_End": 8.0,
                "dt": 0.005,
                "Nsteps": 1_000,
                "This_LJ_Time": 5.0,
                "Cumulative_LJ_Time": 5.0,
                "Ensemble": "NVT",
                "T_Set": 0.9,
                "LJ_r_cut": 2.5,
                "LJ_r_on": 2.0,
                "LJ_Mode": "xplor",
                "Phase_Separation_Status": "Not_Separated",
                "Phase_Separation_Method": "voxel_histogram",
                "Phase_Separation_Method_Version": "test",
                "Phase_Fit_Status": "Not_Run",
                "Summary_Start_Step": 0,
                "Summary_End_Step": 1_000,
                "Summary_Num_Samples": 2,
                "Num_Frames": 11,
            },
            master={"Status": "Complete", "Current_Nstep": 1_000},
        )
        request = CloneRescaleThermalizationConfig(
            source_run_id=source_run_id,
            final_density=0.4,
            nsteps=20_000,
            notes="slow expansion",
        )
        master, thermal, frame_id, note = _clone_request_context(
            request,
            self.database,
        )
        self.assertEqual(master["N_Cells"], 4)
        self.assertEqual(thermal["Therm_Seed"], 7)
        self.assertEqual(frame_id, 10)
        self.assertIn("density changing linearly from 0.500000 to 0.400000", note)
        self.assertTrue(note.endswith("User note: slow expansion"))

        inherited = _inherited_clone_config(
            request,
            master,
            thermal,
            {
                "mdsims/output/Log_Period": 250,
                "mdsims/protocol/Device": "GPU",
            },
            StateData(
                positions=np.zeros((4 * 4**3, 3)),
                velocities=np.zeros((4 * 4**3, 3)),
                box=np.array([8.0, 8.0, 8.0, 0.0, 0.0, 0.0]),
                n_particles=4 * 4**3,
                particle_types=("A",),
            ),
        )
        self.assertEqual(inherited.kT, 0.9)
        self.assertEqual(inherited.seed, 7)
        self.assertEqual(inherited.log_period, 250)
        self.assertEqual(inherited.target_rho, 0.4)

    @patch(
        "md_Helpers.database._display_dataframe",
        side_effect=lambda table, **_: table,
    )
    def test_master_display_can_hide_clock_times(self, _display):
        table = display_master_table(
            self.database,
            show_run_signature=False,
            show_clock_times=False,
        )
        self.assertNotIn("Run_Signature", table.columns)
        self.assertNotIn("StartTime", table.columns)
        self.assertNotIn("EndTime", table.columns)
        self.assertNotIn("Last_Update_Time", table.columns)
        self.assertIn("ElapsedTime", table.columns)

    @patch(
        "md_Helpers.database._display_dataframe",
        side_effect=lambda table, **_: table,
    )
    def test_master_display_limit_returns_latest_rows(self, _display):
        with self.database.connection() as connection:
            connection.executemany(
                "INSERT INTO MD_Master (Run_ID) VALUES (?)",
                [
                    ("20260910000001",),
                    ("20260910000002",),
                    ("20260910000003",),
                ],
            )

        table = display_master_table(self.database, limit=2)

        self.assertEqual(
            table["Run_ID"].tolist(),
            ["20260910000002", "20260910000003"],
        )

    @patch(
        "md_Helpers.database._display_dataframe",
        side_effect=lambda table, **_: table,
    )
    def test_master_display_rejects_nonpositive_limit(self, _display):
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            display_master_table(self.database, limit=0)


if __name__ == "__main__":
    unittest.main()
