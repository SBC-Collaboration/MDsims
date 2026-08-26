import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

matplotlib.use("Agg")

from md_Helpers.phase_density import (
    density_uncertainties_from_fit,
    discover_phase_separated_thermalized_states,
    plot_density_ratio_with_uncertainty,
    select_phase_density_results,
)


class DensityUncertaintyTests(unittest.TestCase):
    def test_propagates_gas_liquid_and_ratio_covariance(self):
        fit = {
            "parameter_covariance": np.diag([0.01, 0.04, 0.0, 0.0, 0.0]),
            "voxel_volume": 2.0,
            "gas_mean_count": 4.0,
            "liquid_mean_count": 12.0,
        }

        result = density_uncertainties_from_fit(fit)

        self.assertAlmostEqual(result["gas_density"]["value"], 2.0)
        self.assertAlmostEqual(result["gas_density"]["se"], 0.2)
        self.assertAlmostEqual(result["liquid_density"]["value"], 6.0)
        self.assertAlmostEqual(
            result["liquid_density"]["se"],
            np.sqrt(0.68),
        )
        self.assertAlmostEqual(
            result["liquid_to_gas_density_ratio"]["value"],
            3.0,
        )
        self.assertAlmostEqual(
            result["liquid_to_gas_density_ratio"]["se"],
            np.sqrt(0.20),
        )

    def test_uses_parameter_correlation(self):
        covariance = np.zeros((5, 5))
        covariance[0, 0] = 0.01
        covariance[1, 1] = 0.01
        covariance[0, 1] = covariance[1, 0] = 0.009
        fit = {
            "parameter_covariance": covariance,
            "voxel_volume": 1.0,
            "gas_mean_count": 2.0,
            "liquid_mean_count": 10.0,
        }

        correlated = density_uncertainties_from_fit(fit)
        independent_fit = dict(fit)
        independent_fit["parameter_covariance"] = np.diag(np.diag(covariance))
        independent = density_uncertainties_from_fit(independent_fit)

        self.assertLess(
            correlated["liquid_to_gas_density_ratio"]["se"],
            independent["liquid_to_gas_density_ratio"]["se"],
        )


class ResultSelectionTests(unittest.TestCase):
    def test_filters_ncells_temperature_density_and_status(self):
        table = pd.DataFrame({
            "status": ["completed", "completed", "fit_failed", "completed"],
            "n_fcc_cells": [20, 20, 20, 30],
            "kT": [0.7, 0.8, 0.8, 0.8],
            "target_rho": [0.70, 0.70, 0.72, 0.70],
            "seed": [1, 1, 1, 2],
        })

        selected = select_phase_density_results(
            table,
            ncells=20,
            temperatures=[0.8],
            target_densities=0.70,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["n_fcc_cells"], 20)
        self.assertAlmostEqual(selected.iloc[0]["kT"], 0.8)

    def test_single_ncell_ratio_plot_uses_temperature_axis(self):
        table = pd.DataFrame({
            "status": ["completed", "completed"],
            "n_fcc_cells": [30, 30],
            "target_rho": [0.70, 0.70],
            "kT": [0.7, 0.8],
            "liquid_to_gas_density_ratio": [12.0, 8.0],
            "liquid_to_gas_density_ratio_se": [0.5, 0.4],
            "liquid_to_gas_density_ratio_ci95_low": [11.0, 7.2],
            "liquid_to_gas_density_ratio_ci95_high": [13.1, 8.9],
        })

        axis = plot_density_ratio_with_uncertainty(table)

        self.assertEqual(axis.get_xlabel(), "kT")
        self.assertIn("ncell=30", axis.get_title())
        self.assertIn("rho=0.7", axis.get_legend_handles_labels()[1])

        ci_axis = plot_density_ratio_with_uncertainty(
            table,
            uncertainty="ci95",
        )
        self.assertEqual(ci_axis.get_xlabel(), "kT")


class DiscoveryTests(unittest.TestCase):
    @unittest.skipIf(h5py is None, "h5py is not installed")
    def test_reads_voxel_phase_classification(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = root / "run" / "randomization_log.hdf5"
            log_path.parent.mkdir()
            with h5py.File(log_path, mode="w") as hdf:
                group = hdf.create_group(
                    "metadata/classification/phase_separation/voxel"
                )
                group.attrs["phase_separated"] = True
                group.attrs["low_density_fraction"] = 0.025

            summary = {
                "n_fcc_cells": 20,
                "target_rho": 0.7,
                "actual_rho": 0.7,
                "kT": 0.8,
                "nsteps": 1_000_000,
                "seed": 1,
                "phase_name": "randomization",
            }
            with patch(
                "md_Helpers.phase_density.summarize_thermalization_log",
                return_value=summary,
            ):
                states = discover_phase_separated_thermalized_states(root)

        self.assertEqual(len(states), 1)
        self.assertTrue(bool(states.iloc[0]["phase_separated"]))
        self.assertAlmostEqual(
            states.iloc[0]["phase_sep_low_density_fraction"],
            0.025,
        )


if __name__ == "__main__":
    unittest.main()
