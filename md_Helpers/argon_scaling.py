"""Fit and apply an argon-like physical interpretation of LJ reduced units."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BOLTZMANN_J_K = 1.380649e-23
ATOMIC_MASS_KG = 1.66053906660e-27
ELEMENTARY_CHARGE_C = 1.602176634e-19
AVOGADRO_MOL = 6.02214076e23
ARGON_MOLAR_MASS_KG_MOL = 39.948e-3
ARGON_ATOMIC_MASS_U = 39.948


def default_argon_saturation_path() -> Path:
    """Return the repository's cleaned Argon saturation table."""

    return (
        Path(__file__).resolve().parents[1]
        / "Tech Note"
        / "Argon Like"
        / "argon_saturation_clean.csv"
    )


@dataclass(frozen=True)
class ArgonLJScale:
    """The three LJ base units and their common physical conversions.

    The coexistence fit determines ``epsilon_over_kb_K`` and ``sigma_nm``.
    Equilibrium data do not determine mass, so the default assigns one argon
    atom to the reduced mass unit.
    """

    epsilon_over_kb_K: float
    sigma_nm: float
    mass_u: float = ARGON_ATOMIC_MASS_U
    epsilon_over_kb_uncertainty_K: float = 0.0
    sigma_uncertainty_nm: float = 0.0

    @property
    def epsilon_J(self) -> float:
        return self.epsilon_over_kb_K * BOLTZMANN_J_K

    @property
    def epsilon_uncertainty_J(self) -> float:
        return self.epsilon_over_kb_uncertainty_K * BOLTZMANN_J_K

    @property
    def sigma_m(self) -> float:
        return self.sigma_nm * 1e-9

    @property
    def mass_kg(self) -> float:
        return self.mass_u * ATOMIC_MASS_KG

    @property
    def time_s(self) -> float:
        return self.sigma_m * np.sqrt(self.mass_kg / self.epsilon_J)

    @property
    def time_ps(self) -> float:
        return self.time_s * 1e12

    @property
    def pressure_Pa(self) -> float:
        return self.epsilon_J / self.sigma_m**3

    @property
    def velocity_m_s(self) -> float:
        return self.sigma_m / self.time_s

    @property
    def force_N(self) -> float:
        return self.epsilon_J / self.sigma_m

    @property
    def density_mol_L(self) -> float:
        """Molar-density multiplier for one reduced number-density unit."""

        return 1.0 / (0.602214076 * self.sigma_nm**3)

    @property
    def density_scale_uncertainty_mol_L(self) -> float:
        return (
            3.0
            * self.density_mol_L
            * self.sigma_uncertainty_nm
            / self.sigma_nm
        )

    def temperature(self, reduced_temperature):
        return np.asarray(reduced_temperature) * self.epsilon_over_kb_K

    def length(self, reduced_length, unit: str = "nm"):
        factors = {
            "m": self.sigma_m,
            "nm": self.sigma_nm,
            "angstrom": 10.0 * self.sigma_nm,
        }
        if unit not in factors:
            raise ValueError("length unit must be 'm', 'nm', or 'angstrom'")
        return np.asarray(reduced_length) * factors[unit]

    def time(self, reduced_time, unit: str = "ps"):
        factors = {"s": self.time_s, "ps": self.time_ps}
        if unit not in factors:
            raise ValueError("time unit must be 's' or 'ps'")
        return np.asarray(reduced_time) * factors[unit]

    def energy_factor(self, unit: str = "eV") -> float:
        factors = {
            "J": self.epsilon_J,
            "eV": self.epsilon_J / ELEMENTARY_CHARGE_C,
            "keV": self.epsilon_J / (1e3 * ELEMENTARY_CHARGE_C),
            "kJ/mol": self.epsilon_J * AVOGADRO_MOL / 1e3,
        }
        if unit not in factors:
            raise ValueError("energy unit must be J, eV, keV, or kJ/mol")
        return factors[unit]

    def energy(self, reduced_energy, unit: str = "eV"):
        return np.asarray(reduced_energy) * self.energy_factor(unit)

    def energy_uncertainty(
        self,
        reduced_energy,
        reduced_uncertainty,
        unit: str = "eV",
    ):
        factor = self.energy_factor(unit)
        relative_scale_uncertainty = (
            self.epsilon_over_kb_uncertainty_K / self.epsilon_over_kb_K
        )
        return np.sqrt(
            (factor * np.asarray(reduced_uncertainty)) ** 2
            + (
                factor
                * np.asarray(reduced_energy)
                * relative_scale_uncertainty
            ) ** 2
        )

    def pressure(self, reduced_pressure, unit: str = "bar"):
        factors = {
            "Pa": self.pressure_Pa,
            "bar": self.pressure_Pa / 1e5,
            "MPa": self.pressure_Pa / 1e6,
        }
        if unit not in factors:
            raise ValueError("pressure unit must be Pa, bar, or MPa")
        return np.asarray(reduced_pressure) * factors[unit]

    def number_density(self, reduced_density, unit: str = "mol/L"):
        factors = {
            "mol/L": self.density_mol_L,
            "1/nm^3": 1.0 / self.sigma_nm**3,
            "kg/m^3": self.density_mol_L * 1000 * ARGON_MOLAR_MASS_KG_MOL,
        }
        if unit not in factors:
            raise ValueError(
                "density unit must be mol/L, 1/nm^3, or kg/m^3"
            )
        return np.asarray(reduced_density) * factors[unit]

    def number_density_uncertainty(
        self,
        reduced_density,
        reduced_uncertainty,
        unit: str = "mol/L",
    ):
        factor = float(self.number_density(1.0, unit=unit))
        relative_scale_uncertainty = (
            self.density_scale_uncertainty_mol_L / self.density_mol_L
        )
        return np.sqrt(
            (factor * np.asarray(reduced_uncertainty)) ** 2
            + (
                factor
                * np.asarray(reduced_density)
                * relative_scale_uncertainty
            ) ** 2
        )

    def summary(self) -> dict[str, float]:
        """Return the fitted base units and useful derived unit scales."""

        return {
            "epsilon_over_kB_K": self.epsilon_over_kb_K,
            "epsilon_over_kB_uncertainty_K": (
                self.epsilon_over_kb_uncertainty_K
            ),
            "epsilon_J": self.epsilon_J,
            "epsilon_eV": self.energy_factor("eV"),
            "sigma_nm": self.sigma_nm,
            "sigma_uncertainty_nm": self.sigma_uncertainty_nm,
            "mass_u": self.mass_u,
            "mass_kg": self.mass_kg,
            "time_ps": self.time_ps,
            "pressure_bar": self.pressure(1.0, "bar").item(),
            "density_mol_L": self.density_mol_L,
            "velocity_m_s": self.velocity_m_s,
            "force_N": self.force_N,
        }


@dataclass
class ArgonScaleFit:
    """Fitted scale plus the exact retained points and fit diagnostics."""

    scale: ArgonLJScale
    ratio_data: Any
    density_data: Any
    temperature_chi2: float
    temperature_dof: int
    density_chi2: float
    density_dof: int

    @property
    def temperature_reduced_chi2(self) -> float:
        return self.temperature_chi2 / self.temperature_dof

    @property
    def density_reduced_chi2(self) -> float:
        return self.density_chi2 / self.density_dof


def query_argon_scaling_states(database, **filters: Any):
    """Select the phase-separated states used by ``Argon_Like.ipynb``."""

    from .database import thermalization_dataframe

    query = {
        "N_Cells": 40,
        "Cumulative_LJ_Time": 200,
        "Phase_Separation_Status": "Separated",
        "rho_gas_unc": (None, 0.002),
        "Therm_kT": (0.85, None),
        "Clone_Run_ID": None,
        **filters,
    }
    database.initialize()
    return thermalization_dataframe(database, **query)


def _load_argon_table(argon_table):
    import pandas as pd

    source = default_argon_saturation_path() if argon_table is None else argon_table
    table = (
        pd.read_csv(source)
        if isinstance(source, (str, bytes, Path)) or hasattr(source, "__fspath__")
        else pd.DataFrame(source).copy()
    )
    required = [
        "temperature_K",
        "density_liquid_mol_L",
        "density_vapor_mol_L",
    ]
    missing = [column for column in required if column not in table]
    if missing:
        raise KeyError(f"Argon saturation table is missing columns: {missing}")
    table[required] = table[required].apply(pd.to_numeric, errors="coerce")
    table = (
        table.dropna(subset=required)
        .sort_values("temperature_K")
        .drop_duplicates("temperature_K")
        .reset_index(drop=True)
    )
    if len(table) < 2:
        raise ValueError("Argon saturation table needs at least two points")
    return table


def fit_argon_lj_scale(
    thermalizations,
    argon_table=None,
    *,
    initial_temperature_scale_K: float = 132.0,
    mass_u: float = ARGON_ATOMIC_MASS_U,
) -> ArgonScaleFit:
    """Fit ``epsilon/kB`` from density ratios and ``sigma`` from densities.

    This is the reusable form of the two fits in ``Argon_Like.ipynb``.  The
    input should be the selected phase-separated V4 thermalization table.
    """

    import pandas as pd
    from scipy.optimize import least_squares

    required = [
        "Run_ID",
        "Therm_kT",
        "rho_liquid",
        "rho_liquid_unc",
        "rho_gas",
        "rho_gas_unc",
    ]
    states = pd.DataFrame(thermalizations).copy()
    missing = [column for column in required if column not in states]
    if missing:
        raise KeyError(f"Thermalization table is missing columns: {missing}")
    numeric = required[1:]
    states[numeric] = states[numeric].apply(pd.to_numeric, errors="coerce")
    states = states.dropna(subset=required).copy()
    states = states.loc[
        (states[numeric] > 0).all(axis=1)
    ].copy()
    if states.empty:
        raise ValueError("No states have positive phase densities and uncertainties")

    states["rho_liquid_to_gas"] = states["rho_liquid"] / states["rho_gas"]
    states["ratio_unc"] = states["rho_liquid_to_gas"] * np.sqrt(
        (states["rho_liquid_unc"] / states["rho_liquid"]) ** 2
        + (states["rho_gas_unc"] / states["rho_gas"]) ** 2
    )

    argon = _load_argon_table(argon_table)
    argon_temperature = argon["temperature_K"].to_numpy(dtype=float)
    argon_liquid = argon["density_liquid_mol_L"].to_numpy(dtype=float)
    argon_vapor = argon["density_vapor_mol_L"].to_numpy(dtype=float)
    argon_ratio = argon_liquid / argon_vapor

    states = states.loc[
        (initial_temperature_scale_K * states["Therm_kT"]).between(
            argon_temperature.min(), argon_temperature.max()
        )
    ].copy()
    if len(states) < 3:
        raise ValueError(
            "Fewer than three simulation points overlap the Argon table "
            "at the initial temperature scale"
        )
    reduced_temperature = states["Therm_kT"].to_numpy(dtype=float)
    observed_ratio = states["rho_liquid_to_gas"].to_numpy(dtype=float)
    ratio_uncertainty = states["ratio_unc"].to_numpy(dtype=float)
    scale_lower = argon_temperature.min() / reduced_temperature.min()
    scale_upper = argon_temperature.max() / reduced_temperature.max()
    if scale_lower >= scale_upper:
        raise ValueError("Selected temperatures cannot share one Argon scale")

    def residual(parameters):
        prediction = np.interp(
            parameters[0] * reduced_temperature,
            argon_temperature,
            argon_ratio,
        )
        return (observed_ratio - prediction) / ratio_uncertainty

    fit = least_squares(
        residual,
        x0=[np.clip(initial_temperature_scale_K, scale_lower, scale_upper)],
        bounds=([scale_lower], [scale_upper]),
    )
    if not fit.success:
        raise RuntimeError(f"Argon temperature-scale fit failed: {fit.message}")
    epsilon_over_kb_K = float(fit.x[0])
    prediction = np.interp(
        epsilon_over_kb_K * reduced_temperature,
        argon_temperature,
        argon_ratio,
    )
    weighted_residual = (observed_ratio - prediction) / ratio_uncertainty
    temperature_chi2 = float(np.sum(weighted_residual**2))
    temperature_dof = len(states) - 1
    jacobian = fit.jac[:, 0]
    epsilon_uncertainty_K = (
        float(1.0 / np.sqrt(np.sum(jacobian**2)))
        if np.sum(jacobian**2) > 0
        else np.nan
    )
    states["temperature_K"] = epsilon_over_kb_K * states["Therm_kT"]
    states["argon_ratio"] = prediction
    states["ratio_weighted_residual"] = weighted_residual
    states["argon_liquid_mol_L"] = np.interp(
        states["temperature_K"], argon_temperature, argon_liquid
    )
    states["argon_vapor_mol_L"] = np.interp(
        states["temperature_K"], argon_temperature, argon_vapor
    )

    liquid = pd.DataFrame({
        "phase": "Liquid",
        "rho_sim": states["rho_liquid"],
        "rho_sim_unc": states["rho_liquid_unc"],
        "rho_argon": states["argon_liquid_mol_L"],
    })
    vapor = pd.DataFrame({
        "phase": "Vapor",
        "rho_sim": states["rho_gas"],
        "rho_sim_unc": states["rho_gas_unc"],
        "rho_argon": states["argon_vapor_mol_L"],
    })
    density_points = pd.concat([liquid, vapor], ignore_index=True)
    density_points["relative_unc"] = (
        density_points["rho_sim_unc"] / density_points["rho_sim"]
    )
    density_points["log_scale_needed"] = np.log(
        density_points["rho_argon"] / density_points["rho_sim"]
    )
    density_points["weight"] = 1.0 / density_points["relative_unc"] ** 2
    log_density_scale = np.average(
        density_points["log_scale_needed"],
        weights=density_points["weight"],
    )
    density_scale = float(np.exp(log_density_scale))
    log_density_scale_uncertainty = float(
        1.0 / np.sqrt(density_points["weight"].sum())
    )
    density_scale_uncertainty = (
        density_scale * log_density_scale_uncertainty
    )
    density_points["weighted_log_residual"] = (
        np.log(
            density_scale
            * density_points["rho_sim"]
            / density_points["rho_argon"]
        )
        / density_points["relative_unc"]
    )
    density_chi2 = float(
        np.sum(density_points["weighted_log_residual"] ** 2)
    )
    density_dof = len(density_points) - 1

    sigma_nm = (1.0 / (0.602214076 * density_scale)) ** (1.0 / 3.0)
    sigma_uncertainty_nm = (
        sigma_nm * density_scale_uncertainty / (3.0 * density_scale)
    )
    scale = ArgonLJScale(
        epsilon_over_kb_K=epsilon_over_kb_K,
        epsilon_over_kb_uncertainty_K=epsilon_uncertainty_K,
        sigma_nm=float(sigma_nm),
        sigma_uncertainty_nm=float(sigma_uncertainty_nm),
        mass_u=float(mass_u),
    )
    return ArgonScaleFit(
        scale=scale,
        ratio_data=states.reset_index(drop=True),
        density_data=density_points.reset_index(drop=True),
        temperature_chi2=temperature_chi2,
        temperature_dof=temperature_dof,
        density_chi2=density_chi2,
        density_dof=density_dof,
    )
