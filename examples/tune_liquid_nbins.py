"""Notebook-style Gaussian voxel study for one homogeneous liquid state."""

import matplotlib.pyplot as plt

from md_Helpers import (
    ProjectPaths,
    SQLiteRunDatabase,
    display_thermalization_table,
    fit_liquid_nbins,
    plot_liquid_gaussian_mean_vs_nbins,
    plot_liquid_nbins_gaussians,
)


paths = ProjectPaths()
database = SQLiteRunDatabase(paths.database)

# Adjust these filters to select exactly one state known to stay liquid.
state = display_thermalization_table(
    database,
    N_Cells=30,
    Phase_Separation_Status="Not_Separated",
).iloc[[0]]

# Set any positive integer voxel resolutions you want to compare.
trial_fits = fit_liquid_nbins(
    state,
    nbins_values=[8, 10, 12, 14, 16],
    project_paths=paths,
)

# Inspect the Gaussian shape at every resolution.
shape_figure, shape_axes = plot_liquid_nbins_gaussians(trial_fits)
plt.show()

# Requested summary: fitted Gaussian mean density on y, nbins on x.
mean_figure, mean_axis = plot_liquid_gaussian_mean_vs_nbins(trial_fits)
plt.show()

display(
    trial_fits[
        [
            "voxel_nbins",
            "gaussian_mean",
            "gaussian_mean_unc",
            "gaussian_sigma",
            "gaussian_mean_density",
            "success",
        ]
    ]
)
