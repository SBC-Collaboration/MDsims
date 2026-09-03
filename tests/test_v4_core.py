from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from md_Helpers.database import (
    SQLiteRunDatabase,
    master_dataframe,
    thermalization_dataframe,
)
from md_Helpers.lattices import build_fcc_lattice
from md_Helpers.paths import ProjectPaths
from md_Helpers.signatures import create_run_signature
from md_Helpers.thermalization import ThermalizationConfig
from md_Helpers.voxel_fit import conditional_phase_fit, phase_fit_sql_values


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


class LatticeTests(unittest.TestCase):
    def test_fcc_count_and_density(self):
        lattice = build_fcc_lattice(n_cells=3, density=0.5)
        self.assertEqual(lattice.n_particles, 4 * 3**3)
        self.assertEqual(lattice.positions.shape, (4 * 3**3, 3))
        self.assertAlmostEqual(lattice.actual_density, 0.5)


class PhaseFitPolicyTests(unittest.TestCase):
    @patch("md_Helpers.voxel_fit.fit_final_frame_voxel_mixture")
    def test_homogeneous_state_skips_fit_and_leaves_values_null(self, fit):
        result = conditional_phase_fit(
            {"phase_separated": False},
            positions=np.zeros((1, 3)),
            box=np.array([10, 10, 10, 0, 0, 0]),
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
        with self.database.connection() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM Thermalization WHERE Run_ID = ?",
                (run_id,),
            ).fetchone()[0]
        self.assertEqual(count, 1)

        table = thermalization_dataframe(
            self.database,
            Therm_kT=(0.8, 1.0),
            Nsteps=[100, 200],
            Phase_Separation_Status="Not_Separated",
        )
        self.assertEqual(len(table), 1)
        self.assertEqual(table.loc[0, "Run_ID"], run_id)


if __name__ == "__main__":
    unittest.main()
