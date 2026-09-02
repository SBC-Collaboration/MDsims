import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py

from md_Helpers import metadata


class TestThermalizedMetadataCleanup(unittest.TestCase):
    def test_thermalized_writer_schema_is_lean(self):
        groups = metadata.split_simulation_metadata({
            "state_kind": "thermalized",
            "data_version": "v3",
            "lattice_type": "fcc",
            "density_mode": "fixed_N_variable_L",
            "n_fcc_cells": 30,
            "N": 108000,
            "target_rho": 0.755,
            "actual_rho": 0.755,
            "kT": 0.8,
            "BoxLength": 52.3,
            "volume": 143000.0,
            "fcc_cell_size": 1.74,
            "phase_name": "randomization",
            "nsteps": 1_000_000,
            "final_timestep": 1_000_000,
            "seed": 1,
            "dt": 0.005,
            "log_period": 1000,
            "state_path": "/tmp/randomization.gsd",
            "log_path": "/tmp/randomization_log.hdf5",
            "starting_state_path": "/tmp/lattice.gsd",
        })

        self.assertNotIn("data_version", groups["metadata/state"])
        self.assertNotIn("fcc_cell_size", groups["metadata/state"])
        self.assertNotIn("final_timestep", groups["metadata/run"])
        self.assertNotIn("metadata/paths", groups)
        self.assertEqual(
            groups["metadata/state"]["BoxLength"],
            52.3,
        )

    def test_cleanup_reports_and_removes_only_retired_fields(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "randomization_log.hdf5"
            with h5py.File(log_path, "w") as hdf:
                state = hdf.require_group("metadata/state")
                state.attrs["state_kind"] = "thermalized"
                state.attrs["BoxLength"] = 52.3
                state.attrs["data_version"] = "v3"
                state.attrs["fcc_cell_size"] = 1.74

                run = hdf.require_group("metadata/run")
                run.attrs["nsteps"] = 1_000_000
                run.attrs["final_timestep"] = 1_000_000

                paths = hdf.require_group("metadata/paths")
                paths.attrs["log_path"] = str(log_path)
                paths.attrs["state_path"] = str(
                    log_path.with_name("randomization.gsd")
                )

                voxel = hdf.require_group(
                    "metadata/classification/phase_separation/voxel"
                )
                voxel.attrs["low_density_fraction"] = 0.0
                voxel.attrs["max_voxel_density"] = 0.9

            preview = metadata.cleanup_thermalized_metadata_file(
                log_path,
                dry_run=True,
            )
            self.assertEqual(preview["status"], "would_clean")
            self.assertEqual(preview["removed_count"], 6)

            with h5py.File(log_path, "r") as hdf:
                self.assertIn("data_version", hdf["metadata/state"].attrs)

            applied = metadata.cleanup_thermalized_metadata_file(
                log_path,
                dry_run=False,
            )
            self.assertEqual(applied["status"], "cleaned")

            with h5py.File(log_path, "r") as hdf:
                self.assertEqual(
                    hdf["metadata/state"].attrs["BoxLength"],
                    52.3,
                )
                self.assertNotIn(
                    "data_version",
                    hdf["metadata/state"].attrs,
                )
                self.assertNotIn(
                    "final_timestep",
                    hdf["metadata/run"].attrs,
                )
                self.assertNotIn("metadata/paths", hdf)
                self.assertIn(
                    "low_density_fraction",
                    hdf[
                        "metadata/classification/phase_separation/voxel"
                    ].attrs,
                )


if __name__ == "__main__":
    unittest.main()
