from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py

from md_Helpers.database import SQLiteRunDatabase
from md_Helpers.paths import ProjectPaths
from md_Helpers.phase_fit_backfill import backfill_skew_phase_fits
from md_Helpers.storage import replace_versioned_phase_fit_metadata
from md_Helpers.voxel_fit import (
    PHASE_FIT_METHOD,
    PHASE_FIT_METHOD_VERSION,
    conditional_phase_fit,
)


RESULT_TABLE_SQL = """
CREATE TABLE {name} (
    Run_ID TEXT PRIMARY KEY,
    File_Location TEXT NOT NULL,
    N_Cells INTEGER NOT NULL,
    Phase_Separation_Status TEXT NOT NULL,
    Phase_Fit_Status TEXT NOT NULL,
    Phase_Fit_Method TEXT,
    Phase_Fit_Method_Version TEXT,
    rho_liquid REAL,
    rho_liquid_unc REAL,
    rho_gas REAL,
    rho_gas_unc REAL,
    V_liquid REAL,
    V_liquid_unc REAL,
    V_gas REAL,
    V_gas_unc REAL
)
"""


class SkewDefaultTests(unittest.TestCase):
    @patch("md_Helpers.voxel_fit.fit_trajectory_voxel_skew_mixture")
    def test_conditional_fit_uses_skew_model_without_nbins_override(self, fit):
        fit.return_value = {
            "success": True,
            "message": "ok",
            "method": PHASE_FIT_METHOD,
            "method_version": PHASE_FIT_METHOD_VERSION,
        }
        result = conditional_phase_fit(
            {"phase_separated": True},
            "trajectory.gsd",
            40,
            interface_points=12,
        )
        self.assertEqual(result["status"], "Complete")
        fit.assert_called_once_with(
            "trajectory.gsd",
            40,
            interface_points=12,
        )


class PhaseFitHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = SQLiteRunDatabase(self.root / "runs.sqlite3")
        self.database.initialize()
        with self.database.connection() as connection:
            connection.execute(
                "INSERT INTO MD_Master (Run_ID) VALUES ('20261001000001')"
            )
            for table in ("Results_A", "Results_B"):
                connection.execute(RESULT_TABLE_SQL.format(name=table))
                connection.execute(
                    f"""
                    INSERT INTO {table} VALUES (
                        '20261001000001', 'Thermalization/20261001000001', 40,
                        'Separated',
                        'Complete', 'normal', 'normal-v1',
                        0.6, 0.01, 0.05, 0.01, 80, 1, 20, 1
                    )
                    """
                )

    def tearDown(self):
        self.temporary.cleanup()

    def test_updates_every_matching_table_and_preserves_legacy(self):
        updated = self.database.apply_phase_fit_everywhere(
            "20261001000001",
            active_values={
                "rho_liquid": 0.61,
                "rho_liquid_unc": 0.001,
                "rho_gas": 0.04,
                "rho_gas_unc": 0.002,
                "V_liquid": 81.0,
                "V_liquid_unc": 0.5,
                "V_gas": 19.0,
                "V_gas_unc": 0.5,
                "Phase_Fit_Status": "Complete",
                "Phase_Fit_Method": PHASE_FIT_METHOD,
                "Phase_Fit_Method_Version": PHASE_FIT_METHOD_VERSION,
            },
            history_values={
                "Method": PHASE_FIT_METHOD,
                "Method_Version": PHASE_FIT_METHOD_VERSION,
                "Status": "Complete",
                "Attempt_Count": 1,
            },
        )
        self.assertEqual(updated, ["Results_A", "Results_B"])
        with self.database.connection() as connection:
            for table in updated:
                row = connection.execute(
                    f"SELECT * FROM {table} WHERE Run_ID = '20261001000001'"
                ).fetchone()
                self.assertEqual(row["Phase_Fit_Method_Version"], PHASE_FIT_METHOD_VERSION)
                self.assertAlmostEqual(row["rho_liquid"], 0.61)
        history = self.database.query_phase_fit_history(run_id="20261001000001")
        self.assertEqual(len(history), 4)
        self.assertEqual(
            {(row["Sim_Table"], row["Method_Version"]) for row in history},
            {
                ("Results_A", "normal-v1"),
                ("Results_A", PHASE_FIT_METHOD_VERSION),
                ("Results_B", "normal-v1"),
                ("Results_B", PHASE_FIT_METHOD_VERSION),
            },
        )

    def test_dry_run_groups_duplicate_table_rows_into_one_fit(self):
        result = backfill_skew_phase_fits(
            self.database,
            project_paths=ProjectPaths(self.root),
            dry_run=True,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Status"], "Planned")
        self.assertEqual(result.iloc[0]["Tables"], "Results_A,Results_B")
        self.assertEqual(result.iloc[0]["Message"], "nbins=15")


class PhaseFitMetadataTests(unittest.TestCase):
    def test_active_metadata_is_archived_by_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.hdf5"
            with h5py.File(path, mode="w") as hdf5:
                group = hdf5.require_group("mdsims/analysis/phase_fit")
                group.attrs["method"] = "normal"
                group.attrs["method_version"] = "normal-v1"
                group.attrs["rho_liquid"] = 0.6
            replace_versioned_phase_fit_metadata(path, {
                "method": PHASE_FIT_METHOD,
                "method_version": PHASE_FIT_METHOD_VERSION,
                "status": "Complete",
                "rho_liquid": 0.61,
            })
            with h5py.File(path, mode="r") as hdf5:
                active = hdf5["mdsims/analysis/phase_fit"].attrs
                self.assertEqual(active["method_version"], PHASE_FIT_METHOD_VERSION)
                self.assertAlmostEqual(active["rho_liquid"], 0.61)
                legacy = hdf5[
                    "mdsims/analysis/phase_fit_history/normal-v1"
                ].attrs
                self.assertAlmostEqual(legacy["rho_liquid"], 0.6)
                versioned = hdf5[
                    "mdsims/analysis/phase_fit_history/"
                    + PHASE_FIT_METHOD_VERSION
                ].attrs
                self.assertAlmostEqual(versioned["rho_liquid"], 0.61)


if __name__ == "__main__":
    unittest.main()
