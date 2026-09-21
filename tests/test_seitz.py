from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from md_Helpers.seitz import (
    calculate_cavitation_seitz,
    seitz_threshold,
    seitz_threshold_uncertainty,
)


class SeitzCalculationTests(unittest.TestCase):
    def test_v3_threshold_expression(self):
        self.assertEqual(
            seitz_threshold(2, -1.0, -3.0, 2.0, 0.25, 0.5),
            12.0,
        )

    def test_v3_uncertainty_components(self):
        result = seitz_threshold_uncertainty(
            nc=2,
            p0=2,
            rho_c=0.25,
            rho_0=0.5,
            uc_uncertainty=0.2,
            u0_uncertainty=0.1,
            p0_uncertainty=0.3,
            rho_0_uncertainty=0.01,
        )
        self.assertAlmostEqual(result["uc"], 0.4)
        self.assertAlmostEqual(result["u0"], 0.2)
        self.assertAlmostEqual(result["P0"], 1.2)
        self.assertAlmostEqual(result["rho_0"], 0.16)
        self.assertAlmostEqual(
            result["total"], np.sqrt(0.4**2 + 0.2**2 + 1.2**2 + 0.16**2)
        )

    def test_calculation_interpolates_v4_eos_and_propagates_errors(self):
        cavitations = pd.DataFrame([{
            "Run_ID": "20260918190731",
            "N_Cells": 45,
            "Therm_kT": 0.8,
            "Nsteps": 500_000,
            "Initial_Density": 0.25,
            "BoxLength": 2.0,
            "rho_liquid": 0.5,
            "rho_liquid_unc": 0.01,
            "PE_Per_Particle_Mean": -1.0,
            "PE_Per_Particle_SEM": 0.2,
        }])
        eos = pd.DataFrame({
            "Therm_kT": [0.8, 0.8],
            "Density_End": [0.4, 0.6],
            "Pressure_Mean": [1.0, 3.0],
            "Pressure_SEM": [0.1, 0.2],
            "PE_Per_Particle_Mean": [-4.0, -2.0],
            "PE_Per_Particle_SEM": [0.05, 0.10],
        })
        result = calculate_cavitation_seitz(cavitations, eos).iloc[0]

        self.assertEqual(result["Nc"], 2)
        self.assertAlmostEqual(result["u_EOS"], -3.0)
        self.assertAlmostEqual(result["P_EOS"], 2.0)
        self.assertAlmostEqual(result["u_EOS_density_slope"], 10.0)
        self.assertAlmostEqual(result["P_EOS_density_slope"], 10.0)
        self.assertAlmostEqual(result["Q"], 12.0)
        expected_u_uncertainty = np.sqrt(0.075**2 + 0.1**2)
        expected_p_uncertainty = np.sqrt(0.15**2 + 0.1**2)
        self.assertAlmostEqual(
            result["u_EOS_uncertainty"], expected_u_uncertainty
        )
        self.assertAlmostEqual(
            result["P_EOS_uncertainty"], expected_p_uncertainty
        )


if __name__ == "__main__":
    unittest.main()
