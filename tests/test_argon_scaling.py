from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from md_Helpers.argon_scaling import (
    ARGON_ATOMIC_MASS_U,
    ArgonLJScale,
    fit_argon_lj_scale,
)


class ArgonScalingTests(unittest.TestCase):
    def test_base_units_generate_expected_derived_conversions(self):
        scale = ArgonLJScale(
            epsilon_over_kb_K=120.0,
            sigma_nm=0.35,
        )
        self.assertEqual(scale.mass_u, ARGON_ATOMIC_MASS_U)
        self.assertAlmostEqual(scale.temperature(0.8), 96.0)
        self.assertAlmostEqual(scale.length(2.0, "nm"), 0.7)
        self.assertAlmostEqual(
            scale.energy(1.0, "eV"),
            scale.epsilon_J / 1.602176634e-19,
        )
        self.assertGreater(scale.time_ps, 0)
        self.assertGreater(scale.pressure(1.0, "bar"), 0)

    def test_ratio_and_absolute_density_fits_recover_known_scale(self):
        argon = pd.DataFrame({
            "temperature_K": [70, 80, 100, 120, 140, 150],
            "density_liquid_mol_L": [32, 28, 24, 20, 16, 14],
            "density_vapor_mol_L": [0.2, 0.4, 0.8, 1.25, 2.0, 2.8],
        })
        selected = argon.set_index("temperature_K").loc[[80, 100, 120]]
        states = pd.DataFrame({
            "Run_ID": ["1", "2", "3"],
            "Therm_kT": [0.8, 1.0, 1.2],
            "rho_liquid": selected["density_liquid_mol_L"].to_numpy() / 2,
            "rho_liquid_unc": [0.05, 0.05, 0.05],
            "rho_gas": selected["density_vapor_mol_L"].to_numpy() / 2,
            "rho_gas_unc": [0.002, 0.002, 0.002],
        })
        result = fit_argon_lj_scale(
            states,
            argon_table=argon,
            initial_temperature_scale_K=100,
        )
        self.assertAlmostEqual(result.scale.epsilon_over_kb_K, 100.0, places=6)
        self.assertAlmostEqual(result.scale.density_mol_L, 2.0, places=6)
        expected_sigma = (1 / (0.602214076 * 2.0)) ** (1 / 3)
        self.assertAlmostEqual(result.scale.sigma_nm, expected_sigma, places=6)
        self.assertAlmostEqual(result.temperature_chi2, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
