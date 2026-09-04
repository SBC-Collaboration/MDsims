from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from md_Helpers.database import (
    SQLiteRunDatabase,
    display_master_table,
    master_dataframe,
    thermalization_dataframe,
)
from md_Helpers.analysis import thermodynamic_summary
from md_Helpers.lattices import build_fcc_lattice
from md_Helpers.paths import ProjectPaths
from md_Helpers.run_analysis import RunAnalysis, open_run
from md_Helpers.signatures import create_run_signature
from md_Helpers.storage import StateData
from md_Helpers.thermalization import (
    CloneRescaleThermalizationConfig,
    ThermalizationConfig,
    _clone_request_context,
    _inherited_clone_config,
    thermalization_log_steps,
    thermalization_phase_frame_schedule,
)
from md_Helpers.voxel_fit import (
    conditional_phase_fit,
    phase_fit_frame_indices,
    phase_fit_sql_values,
)


class SignatureTests(unittest.TestCase):
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


class LatticeTests(unittest.TestCase):
    def test_fcc_count_and_density(self):
        lattice = build_fcc_lattice(n_cells=3, density=0.5)
        self.assertEqual(lattice.n_particles, 4 * 3**3)
        self.assertEqual(lattice.positions.shape, (4 * 3**3, 3))
        self.assertAlmostEqual(lattice.actual_density, 0.5)


class PhaseFitPolicyTests(unittest.TestCase):
    def test_selects_five_backward_spaced_frames(self):
        self.assertEqual(phase_fit_frame_indices(100), [99, 94, 89, 84, 79])

    def test_short_trajectory_uses_available_frames_without_duplicates(self):
        self.assertEqual(phase_fit_frame_indices(12), [11, 6, 1])

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


if __name__ == "__main__":
    unittest.main()
