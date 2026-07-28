import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py

from md_Helpers import metadata


class TestCavitationCreationMetadataCleanup(unittest.TestCase):
    def _make_file(self, path, random_location):
        with h5py.File(path, "w") as hdf:
            state = hdf.require_group("metadata/state")
            state.attrs["state_kind"] = "cavitation_initial"

            creation = hdf.require_group("metadata/creation")
            creation.attrs["bubble_center"] = [0.0, 0.0, 0.0]
            creation.attrs["bubble_center_x"] = 0.0
            creation.attrs["bubble_method"] = "remove_particles_in_sphere"
            creation.attrs["bubble_radius"] = 2.5
            creation.attrs["bubble_seed"] = 7
            creation.attrs["particles_removed"] = 46
            creation.attrs["periodic_distance"] = True
            creation.attrs["radius"] = 2.5
            creation.attrs["random_location"] = random_location
            creation.attrs["rho_after"] = 0.75
            creation.create_dataset(
                "removed_particle_indices",
                data=[1, 2, 3],
            )

    def test_cleanup_centered_creation_metadata(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cavitation_creation.hdf5"
            self._make_file(path, random_location=False)

            report = metadata.cleanup_cavitation_creation_metadata_file(
                path,
                dry_run=False,
            )
            self.assertEqual(report["status"], "cleaned")

            with h5py.File(path, "r") as hdf:
                creation = hdf["metadata/creation"]
                self.assertEqual(
                    set(creation.attrs),
                    {
                        "bubble_center",
                        "bubble_method",
                        "particles_removed",
                        "periodic_distance",
                        "radius",
                        "random_location",
                    },
                )
                self.assertNotIn("removed_particle_indices", creation)

    def test_cleanup_random_creation_keeps_seed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cavitation_creation.hdf5"
            self._make_file(path, random_location=True)

            metadata.cleanup_cavitation_creation_metadata_file(
                path,
                dry_run=False,
            )

            with h5py.File(path, "r") as hdf:
                self.assertEqual(
                    hdf["metadata/creation"].attrs["bubble_seed"],
                    7,
                )


if __name__ == "__main__":
    unittest.main()
