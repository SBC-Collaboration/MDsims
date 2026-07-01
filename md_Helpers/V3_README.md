# V3 Saving Scheme

V3 keeps the existing readable folder style while making each saved object fit
one of three roles.

## Saved Object Types

```text
generated starting state -> one-frame GSD + creation metadata HDF5
evolved run              -> trajectory GSD + log HDF5 + optional final GSD
summary/search table     -> Parquet
```

## Folder Roots

These roots are defined in `md_Helpers.paths`. `Project_Paths.py` remains as
a compatibility wrapper for older notebooks.

```text
Simple_Lattices_v3/
Thermalized_States_v3/
Cavitation_States_v3/
Cavitation_Evolved_v3/
Excitation_States_v3/
Excitation_Evolved_v3/
Master_CSVs_v3/
```

## Phase Separation Policy

Run phase-separation classifiers on final states from real dynamics:

```text
Thermalized_States_v3/.../randomization.gsd
Cavitation_Evolved_v3/.../cavitation_final.gsd
Excitation_Evolved_v3/.../excitation_final.gsd
```

Do not classify artificial starting states by default:

```text
Cavitation_States_v3/.../cavitation_initial.gsd
Excitation_States_v3/.../excitation_initial.gsd
```

Those starting states should instead store creation metadata such as removed
particle count, selected particle count, altered velocities, injected energy,
and parent-state paths.

## Cavitation Files

Cavitation is split into two saved objects.

```text
Cavitation_States_v3/.../
    cavitation_initial.gsd
    cavitation_creation.hdf5

Cavitation_Evolved_v3/.../
    cavitation_trajectory.gsd
    cavitation_final.gsd
    cavitation_log.hdf5
```

`cavitation_creation.hdf5` stores the artificial bubble construction: bubble
center, radius, removed-particle count, density before and after, and parent
thermalized-state paths. `cavitation_log.hdf5` stores the actual dynamics:
thermodynamic time series, run metadata, trajectory/final-state paths, and
final-state classification.

## Metadata Layout

V3 keeps bare `metadata` as a container only. Metadata attributes live in
purpose-specific child groups:

```text
metadata/state
metadata/run
metadata/lj
metadata/source
metadata/paths
metadata/classification/phase_separation
metadata/classification/phase_separation/voxel
metadata/classification/phase_separation/PE_drop
```

`metadata/paths` stores current file locations. `metadata/source` stores
ancestry such as parent state paths and source data versions. Parent groups
such as `metadata` and `metadata/classification` are containers only and should
not store attributes directly.

## Helper Ownership

```text
paths.py                 V3 folder and file paths
spatial.py               periodic geometry and voxel calculations
lattices.py              FCC lattice creation
simulation.py            HOOMD setup and thermalization
runs.py                  HDF5/GSD writers and run execution
classification.py        current voxel and PE-drop classifiers
voxel_fit.py             gas/liquid/interface voxel histogram fitting
cavitation.py            cavitation creation and evolution
cavitation_analysis.py   trajectory bubble measurements
cavitation_sweep.py      FCC-size sweeps and bubble-survival summaries
visualization.py         plotting and animation
index.py                 searchable V3 Parquet index
metadata.py              structured HDF5 metadata I/O
```

Build the searchable index with:

```python
from md_Helpers import index

table = index.build_v3_index()
```

## Cavitation Size Sweeps

Use explicit `(density, temperature)` conditions to repeat the same FCC-size
sweep at nearby state points. The helper fixes `radius_fraction=0.15`, measures
the trajectory, classifies survival from the final 20% of frames, and writes a
CSV row after every completed run.

```python
from md_Helpers.cavitation_sweep import run_cavitation_size_sweep

summary = run_cavitation_size_sweep(
    n_fcc_cells_values=[10, 15, 20, 25, 30],
    conditions=[
        {"label": "baseline", "density": 0.71, "temperature": 0.80},
        {"label": "lower density", "density": 0.70, "temperature": 0.80},
        {"label": "higher temperature", "density": 0.71, "temperature": 0.82},
    ],
    source_nsteps=1_000_000,
    evolve_nsteps=100_000,
    evolve_seeds=[1, 2, 3],
)
```

The outcome is `stabilized` when the tail-median radius is at least 50% of the
constructed radius, `collapsed` when it is at most 10%, and `intermediate`
otherwise. These thresholds are configurable. Because the radius is a fixed
fraction of the box, its absolute value increases with `n_fcc_cells`; use the
reported `initial_bubble_radius` and `tail_radius_ratio` when comparing sizes.

Before creating a bubble, cavitation checks the voxel phase-separation result
of the thermalized source. A phase-separated source is not cavitated. Its sweep
row has `run_status="thermalization_failed_phase_separated"`,
`thermalization_passed=False`, and `outcome="not_cavitated"`, together with the
source state/log paths and source low-density fraction.
