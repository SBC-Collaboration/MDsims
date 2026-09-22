"""Notebook-style, read-only nbins sensitivity study for 30-cell runs."""

import matplotlib.pyplot as plt

from md_Helpers import (
    ProjectPaths,
    SQLiteRunDatabase,
    display_cavitation_table,
    plot_nbins_phase_fits,
    plot_nbins_seitz,
    refit_cavitation_nbins,
)


paths = ProjectPaths()
database = SQLiteRunDatabase(paths.database)

matches = display_cavitation_table(
    database,
    N_Cells=30,
    Nsteps=[500_000],
    Phase_Separation_Status="Separated",
)

# The normal 30-cell rule gives nbins=12. Try values around it.
trial_fits = refit_cavitation_nbins(
    matches,
    nbins_values=range(9, 16),
    project_paths=paths,
)

# Inspect every trial fit for one run. Pass another Run_ID to switch runs.
fit_figure, fit_axes = plot_nbins_phase_fits(
    trial_fits,
    run_id=matches.iloc[0]["Run_ID"],
)
plt.show()

# Recalculate Q from each trial fit against exactly the same EOS selection.
seitz_figure, seitz_axis, seitz_trials = plot_nbins_seitz(
    matches,
    trial_fits,
    database,
    eos_n_cells=45,
    eos_filters={
        "Nsteps": 200_000,
        "dt": 0.002,
        "Ensemble": "NVT",
    },
)
plt.show()

display_columns = [
    "Run_ID",
    "voxel_nbins",
    "Therm_kT",
    "rho_liquid",
    "rho_liquid_uncertainty",
    "Q",
    "Q_uncertainty",
]
display(seitz_trials[display_columns])
