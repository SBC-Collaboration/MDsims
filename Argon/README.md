# Argon Seitz workflow

Run the notebooks in `notebooks/` in numerical order with the `my_GPU` kernel.

1. `00_REFPROP_Validation.ipynb` validates the REFPROP installation and one Seitz point.
2. `01_Argon_LJ_Scale.ipynb` shows raw coexistence data, fits `epsilon/kB` and `sigma`, plots overlays and residuals, and saves the scale.
3. `02_MD_Seitz_Table.ipynb` shows raw cavitation/EOS data, calculates the MD threshold, converts it to physical units, and saves matched MD states.
4. `03_Argon_REFPROP_Surface.ipynb` generates a standalone real-Argon temperature-pressure Seitz grid with visual validity checks.
5. `04_MD_REFPROP_Comparison.ipynb` evaluates REFPROP at the exact MD states and produces overlay, parity, ratio, and radius checks.

Generated tables are written to `data/`. Figures are displayed inline; save selected publication figures to `plots/` only after the underlying validation cells pass.

The current LJ-to-Argon fit has a large reduced chi-squared, especially for the density scale. Treat it as an approximate Argon-like mapping until replicate scatter and model discrepancy are incorporated into the uncertainty model.
