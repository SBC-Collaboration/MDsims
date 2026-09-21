"""Notebook-style example for a V4 cavitation Seitz plot."""

from md_Helpers import (
    ProjectPaths,
    SQLiteRunDatabase,
    display_cavitation_table,
    plot_cavitation_seitz,
)


paths = ProjectPaths()
database = SQLiteRunDatabase(paths.database)

matches = display_cavitation_table(
    database,
    Nsteps=[500_000],
    Phase_Separation_Status="Separated",
)

figure, axis, seitz_results = plot_cavitation_seitz(
    matches,
    database,
    eos_n_cells=45,
    eos_filters={
        # Add protocol restrictions here when defining the EOS, for example:
        # "Nsteps": 200_000,
        # "dt": 0.002,
        # "Ensemble": "NVT",
        # "LJ_Mode": "xplor",
    },
)
