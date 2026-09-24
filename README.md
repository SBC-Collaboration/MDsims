# MD Sims V4

V4 currently contains the smallest complete thermalization workflow. It:

1. builds the V3-style FCC lattice;
2. computes a deterministic signature from every dynamics/output input;
3. checks the local SQL Master table and skips an existing signature without
   loading any run file;
4. reserves a timestamp Run ID and incrementally updates its Master row;
5. runs NVT thermalization with the V3 Lennard-Jones defaults;
6. logs every requested thermodynamic sample to `run.hdf5`, while
   `trajectory.gsd` stores only the initial state and five terminal states;
7. records thermodynamic summaries and both voxel and PE-drop phase checks;
8. when the final-frame voxel classifier marks the state phase separated,
   averages the five terminal voxel histograms sampled every ten log points,
   then performs one mixture fit; all eight fit values remain SQL `NULL` for a
   homogeneous state;
9. inserts the Thermalization row and marks Master complete in one transaction.

## Output root

Change `TOP_DIRECTORY` in `md_Helpers/paths.py`, or set the
`MDSIMS_TOP_DIRECTORY` environment variable before importing the package.
The configured shared default is:

```text
/exp/e961/data/MDsims-data/pnichols/SQL/
├── mdsims.sqlite3
└── Thermalization/
    └── <Run_ID>/
        ├── trajectory.gsd
        └── run.hdf5
```

The live MySQL server storage will remain separate from this simulation-file
root.

## Notebook use

```python
from md_Helpers import ProjectPaths, ThermalizationConfig, run_thermalization

paths = ProjectPaths()

config = ThermalizationConfig(
    n_fcc_cells=45,
    target_rho=0.5,
    nsteps=100_000,
    kT=0.9,
    log_period=1_000,
    seed=1,
    dt=0.005,
    epsilon_LJ=1.0,
    sigma_LJ=1.0,
    r_cut_LJ=2.5,
    buffer_LJ=0.4,
    lj_mode="xplor",
    r_on_LJ=2.0,
)

result = run_thermalization(config, project_paths=paths)
result
```

## Reproducible visual examples

The retained visualization examples live in `examples/` instead of the legacy
V3 notebooks. Generate the HOOMD LJ cutoff-mode potential and force plots with:

```bash
python -m examples.plot_lj_modes
```

Render one FCC unit cell with its four basis particles colored separately with:

```bash
python -m examples.render_single_fcc_cell
```

Both commands write PNGs below `Plots/`. The LJ example requires HOOMD, and the
FCC renderer requires Fresnel and Matplotlib.

Thermalization requires at least 41 evolved HDF5 log points. With 100 log
points, the trajectory contains the initial state followed by states from logs
60, 70, 80, 90, and 100. These last five frames are the exact inputs to the
averaged voxel histogram.

Running the same cell again returns the existing SQL record with
`result["skipped"] == True`; it does not open or load its GSD/HDF5 files.

## Expanded FCC states

`Expanded_FCC` starts from the same central FCC lattice and NVT inputs as
thermalization, extends the box along x, and places identical lower-density
particle patterns on the left and right. `center_length_scale` controls the
liquid-region length in units of the original box length, independently of the
width added on each side. Defaults of `center_length_scale=1` and
`side_extension=1` produce a `3L x L x L` box. This workflow is indexed in
`MD_Master`; its dedicated SQL result table is intentionally deferred, while
complete protocol and state metadata are retained in `run.hdf5`.

```python
from md_Helpers import ExpandedFCCConfig, run_expanded_fcc

config = ExpandedFCCConfig(
    n_fcc_cells=45,
    target_rho=0.5,
    nsteps=100_000,
    kT=0.9,
    log_period=1_000,
    seed=1,
    dt=0.005,
    center_length_scale=1.0,
    side_extension=1.0,
    side_density_divisor=2.0,
    com_recenter_period=10_000,
)

result = run_expanded_fcc(config)
```

The mass-weighted COM is computed from unwrapped coordinates and translated to
the box center initially and every `com_recenter_period` steps. A trajectory
frame is saved immediately after each recenter; the initial state and a distinct
final step are also saved.

Existing inspection calls work for these runs. As in thermalization,
`plot_phase_fit()` uses exactly five scheduled terminal log frames separated by
ten log intervals; these frames are saved in addition to COM-recenter frames,
and their exact trajectory IDs are recorded in HDF5. The fit is computed on
demand. The final-frame density along the extended x direction is available as
both a table and a plot:

```python
run = open_run(result["run_id"])
run.plot_logs()
run.plot_phase_fit()

figure, density_slices = run.plot_density_profile(
    frame=-1,
    num_slices=60,
)
display(density_slices)
```

For `Expanded_FCC`, `plot_logs()` skips the first three pressure and
potential-energy-per-particle points by default to suppress the initial lattice
transient. Pass `skip_lattice_transient=False` to show them, or set
`expanded_lattice_skip_points` to choose a different cutoff.

## Cavitation states

Cavitation starts from the final frame of a completed, explicitly homogeneous
thermalization. It translates an optional reproducibly sampled location to the
box center, wraps all particles, removes particles inside a centered spherical
mask, and evolves the surviving particles in NVT without changing their
velocities. The source temperature, seed, integration timestep, Lennard-Jones
settings, device preference, log period, and progress period are inherited.

```python
from md_Helpers import CavitationConfig, run_cavitation

config = CavitationConfig(
    source_run_id="THERMALIZATION_RUN_ID",
    mask_radius=5.0,
    nsteps=100_000,
    ensemble="NVT",
    random_location=False,
    location_seed=None,
    notes=None,
)

result = run_cavitation(config)
```

For a random reproducible source location, set `random_location=True` and pass
a nonnegative `location_seed`. The mask diameter may not exceed 85% of the
smallest box length, and initialization fails if the mask removes zero or all
particles. Cavitation requires at least `41 * inherited_log_period` steps.

The trajectory saves frame 0 after translation and masking, every twentieth
evolved log, and the same five terminal logs used by thermalization. With 100
logs, the evolved frames are `20, 40, 60, 70, 80, 90, 100`; only the final five
are used for the averaged voxel histogram. Finished homogeneous states remain
completed Cavitation records, preventing accidental reruns. A phase-separated
source returns `skip_reason == "source_phase_separated"` without creating a
new Master row.

Use the same inspection tools as thermalization:

```python
from md_Helpers import display_cavitation_table, open_run

display_cavitation_table()
run = open_run(result["run_id"])
run.render(frame=0)
run.plot_logs()
run.plot_phase_fit()
```

`N_Cells` is stored in both `MD_Master` and `Thermalization`. New
thermalization writes require the values to match. To migrate and populate an
older database, run:

```python
from md_Helpers import ProjectPaths, SQLiteRunDatabase

database = SQLiteRunDatabase(ProjectPaths().database)
database.backfill_thermalization_n_cells()
```

Clone the last frame of a completed thermalization and change only its box so
that number density varies linearly through the new run:

```python
from md_Helpers import run_clone_rescale_thermalization

result = run_clone_rescale_thermalization(
    source_run_id="20260903214936",
    final_density=0.40,
    nsteps=200_000,
)
result
```

The clone preserves the source positions, velocities, particle properties,
HOOMD timestep, temperature, seed, integration timestep, LJ settings, device
preference, and output periods. The SQL duplicate check occurs before either
source file is opened. The Master note documents the source frame and density
schedule automatically; pass `notes="..."` to append a user note.

To resize a thermalized state without a thermostat, use the ensemble-aware
clone function:

```python
from md_Helpers import run_clone_rescale_ensemble

expansion = run_clone_rescale_ensemble(
    source_run_id="HIGH_DENSITY_RUN_ID",
    final_density=0.40,
    nsteps=200_000,
    ensemble="NVE",
    notes="decreasing-density branch",
)

compression = run_clone_rescale_ensemble(
    source_run_id="LOW_DENSITY_RUN_ID",
    final_density=0.60,
    nsteps=200_000,
    ensemble="NVE",
    notes="increasing-density branch",
)
```

The source velocities are retained and no thermostat is attached in NVE mode.
Because changing the box performs work on the system, total energy need not be
constant during the ramp. Pressure, volume, particle count, and kinetic
temperature are logged at every configured log point. For a pressure-density
curve, use `open_run(run_id).plot_logs(quantities=["pressure"], x="density")`.

Inspect any indexed run by its global ID:

```python
from md_Helpers import open_run

run = open_run("20260903182141")
run.info()
run.render(frame=-1)
run.xy_slice(frame=-1)
run.plot_phase_fit()
run.plot_logs()
```

For runs initialized directly from an FCC lattice, `plot_logs()` omits the
first 10 samples from the pressure and PE/N panels to hide the construction
transient. Clone runs retain every sample. Pass
`skip_lattice_transient=False` to show all points.

`open_run` itself performs only the SQL lookup and path resolution. GSD and
HDF5 data are loaded lazily by the individual inspection methods.

Query Master independently:

```python
from md_Helpers import SQLiteRunDatabase, display_master_table

database = SQLiteRunDatabase(paths.database)
database.query_runs(Sim_Type="Thermalization", Status="Complete")

# Render the latest 100 Master rows as a formatted pandas table.
master = display_master_table(database, limit=100)
```

Display the complete Thermalization table:

```python
from md_Helpers import display_thermalization_table

thermalizations = display_thermalization_table(database)
```

Filter it using equality, inclusive ranges, or lists of accepted values:

```python
thermalizations = display_thermalization_table(
    database,
    limit=100,                              # None returns every match
    Therm_kT=(0.85, 0.95),                 # inclusive range
    Density_End=(0.45, 0.55),              # inclusive range
    Nsteps=[10_000, 100_000, 1_000_000],   # any listed value
    Phase_Separation_Status="Separated",   # exact match
)
```

All supplied filters are combined with `AND`. A scalar means equality, a list
or set means SQL `IN`, a two-item tuple means an inclusive range, and `None`
selects rows where that column is SQL `NULL`.
