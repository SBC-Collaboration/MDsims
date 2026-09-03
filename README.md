# MD Sims V4

V4 currently contains the smallest complete thermalization workflow. It:

1. builds the V3-style FCC lattice;
2. computes a deterministic signature from every dynamics/output input;
3. checks the local SQL Master table and skips an existing signature without
   loading any run file;
4. reserves a timestamp Run ID and incrementally updates its Master row;
5. runs NVT thermalization with the V3 Lennard-Jones defaults;
6. writes exact initial, periodic, and final samples to `trajectory.gsd` and
   `run.hdf5`;
7. records thermodynamic summaries and both voxel and PE-drop phase checks;
8. when the final-frame voxel classifier marks the state phase separated,
   averages the voxel histograms from the final frame and every fifth frame
   backward for five sampled frames, then performs one mixture fit; all eight
   fit values remain SQL `NULL` for a homogeneous state;
9. inserts the Thermalization row and marks Master complete in one transaction.

## Output root

Change `TOP_DIRECTORY` in `md_Helpers/paths.py`, or set the
`MDSIMS_TOP_DIRECTORY` environment variable before importing the package.
The local default is:

```text
MDSims/
├── mdsims.sqlite3
└── Thermalization/
    └── <Run_ID>/
        ├── trajectory.gsd
        └── run.hdf5
```

The future shared root is
`/exp/e961/data/MDsims-data/pnichols/SQL`. The live MySQL server storage will
remain separate from this simulation-file root.

## Notebook use

```python
from md_Helpers import ProjectPaths, ThermalizationConfig, run_thermalization

paths = ProjectPaths()

config = ThermalizationConfig(
    n_fcc_cells=45,
    target_rho=0.5,
    nsteps=10_000,
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

Running the same cell again returns the existing SQL record with
`result["skipped"] == True`; it does not open or load its GSD/HDF5 files.

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

`open_run` itself performs only the SQL lookup and path resolution. GSD and
HDF5 data are loaded lazily by the individual inspection methods.

Query Master independently:

```python
from md_Helpers import SQLiteRunDatabase, display_master_table

database = SQLiteRunDatabase(paths.database)
database.query_runs(Sim_Type="Thermalization", Status="Complete")

# Render every Master row and column as a formatted pandas table.
master = display_master_table(database)
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
