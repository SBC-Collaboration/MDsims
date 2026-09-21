"""Fit the Argon unit scale and make a physical-unit Seitz plot."""

import matplotlib.pyplot as plt

from md_Helpers import (
    ProjectPaths,
    SQLiteRunDatabase,
    display_cavitation_table,
    fit_argon_lj_scale,
    plot_cavitation_seitz,
    query_argon_scaling_states,
)


paths = ProjectPaths()
database = SQLiteRunDatabase(paths.database)

argon_states = query_argon_scaling_states(database)
argon_fit = fit_argon_lj_scale(argon_states)
print(argon_fit.scale.summary())

cavitations = display_cavitation_table(
    database,
    Nsteps=[500_000],
    Phase_Separation_Status="Separated",
)

figure, axis, physical_results = plot_cavitation_seitz(
    cavitations,
    database,
    eos_n_cells=45,
    eos_filters={
        "Nsteps": 200_000,
        "dt": 0.002,
        "Ensemble": "NVT",
    },
    physical_scale=argon_fit.scale,
    energy_unit="eV",
    density_unit="mol/L",
)
plt.show()
