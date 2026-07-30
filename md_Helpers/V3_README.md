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

The old single-timestep contents of `Excitation_Evolved_v3` should be archived
unchanged as `Excitation_Evolved_v3_legacy_single_dt/`.

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

## Two-Segment Excitation Evolution

New excitation evolutions always contain exactly two NVE segments. Segment 1
defaults to `dt1=0.0005` and `nsteps1=200_000`; callers provide `dt2` and
`nsteps2`. The final state of segment 1 is the starting state of segment 2.
During an uninterrupted call, the same live HOOMD simulation is retained and
the integrator `dt` is changed directly at the boundary. If a partial run is
resumed later, segment 2 restarts from segment 1's saved final GSD.

```text
Excitation_Evolved_v3/.../
    segment_1_dt_0.0005/
        nsteps_200000/
            segment_2_dt_<dt2>/
                nsteps_<nsteps2>/
                    seed_<seed>/
                        evolution_manifest.hdf5
                        segment_1/
                            excitation_trajectory.gsd
                            excitation_final.gsd
                            excitation_log.hdf5
                        segment_2/
                            excitation_trajectory.gsd
                            excitation_final.gsd
                            excitation_log.hdf5
```

`evolution_manifest.hdf5` is the authoritative overall record. It stores the
ordered timestep schedule, physical duration, boundary timesteps, paths, run
status, ancestry, creation details, and LJ settings. Segment 2's final GSD and
log are exposed as the overall final-state and final-classification files.

```python
from md_Helpers.hot_spike import get_or_create_hot_spike

result = get_or_create_hot_spike(
    n_fcc_cells=30,
    target_rho=0.71,
    kT=0.8,
    source_nsteps=1_000_000,
    radius=3.0,
    injected_energy=4_000,
    dt2=0.005,
    nsteps2=100_000,
)
```

Stitch the saved outputs only when needed:

```python
from md_Helpers import excitation_evolution

log = excitation_evolution.read_stitched_log(result)
frames = excitation_evolution.iter_stitched_trajectory(result)

combined_log_path = excitation_evolution.write_stitched_log(result)
combined_gsd_path = excitation_evolution.write_stitched_trajectory(result)
```

The stitched log adds an `elapsed_time` array computed piecewise from each
segment's own `dt`. Both stitched readers remove the duplicated segment
boundary. The optional materialized files are derivatives; the two raw
segments and manifest remain authoritative.

The hot-spike animation accepts the result directly. It applies `stride` to
the complete ordered sequence and then takes the first `max_frames`, crossing
from segment 1 into segment 2 only when the selected sequence reaches that
boundary. It shows segment number, raw timestep, and continuous physical time:

```python
from md_Helpers.visualization import (
    animate_hot_spike_xy_slice_trajectory,
)

animate_hot_spike_xy_slice_trajectory(result)
```

Create or refresh the lightweight excitation-evolution inventory from one
notebook cell:

```python
from md_Helpers import master_csv

excitation_runs = master_csv.build_excitation_evolved_master_csv()
excitation_runs
```

The CSV is written to
the dataset's `Master_CSVs_v3/excitation_evolved_master.csv` and mirrored as
`excitation_evolved_master.csv` in the repository root. It contains one row
per current two-segment evolution (plus archived single-timestep evolutions),
keeps the main physical inputs and voxel outcome first, and preserves the
`checked`, `notes`, and any hand-added columns when refreshed.

## NPH Comparison Runs

NVE remains the default. To run the same two-segment hot-spike evolution with
an isotropic NPH barostat, keep the physical and numerical inputs unchanged
and add the ensemble and pressure set point:

```python
nph_result = get_or_create_hot_spike(
    n_fcc_cells=30,
    target_rho=0.71,
    kT=0.8,
    source_nsteps=1_000_000,
    radius=3.0,
    injected_energy=4_000,
    dt2=0.005,
    nsteps2=100_000,
    ensemble="NPH",
    pressure=P0,
)
```

Use the equilibrated homogeneous source pressure (for example, the tail mean
from the matching EOS or source log) for `P0`, not the instantaneous pressure
immediately after the hot spike. NPH uses no thermostat. The default
sets `tauS = 1000 * dt2`, giving `tauS=5.0` when `dt2=0.005`. This physical
barostat time is held constant across both segments even though their
integration timesteps differ. Override `tauS` only as part of a
barostat-sensitivity check.

NPH results are saved below an `ensemble_NPH/pressure_.../tauS_...` branch, so
they cannot collide with existing NVE results. HDF5 logs include the changing
box volume and barostat energy in addition to pressure, temperature, and
particle energies. Each segment also saves the barostat degrees of freedom so
a resumed second segment keeps the same extended NPH state.

Before producing new results, preview and then archive the old root:

```python
from md_Helpers.excitation_evolution import (
    archive_legacy_excitation_evolved,
)

archive_legacy_excitation_evolved(dry_run=True)
archive_legacy_excitation_evolved(dry_run=False)
```

The archive helper refuses to overwrite or merge with an existing archive. It
moves the old root intact and creates a new empty `Excitation_Evolved_v3`.

## Metadata Layout

V3 keeps bare `metadata` as a container only. Metadata attributes live in
purpose-specific child groups:

```text
metadata/state
metadata/run
metadata/lj
metadata/source
metadata/paths
metadata/segments/segment_1
metadata/segments/segment_2
metadata/continuity
metadata/classification/phase_separation
metadata/classification/phase_separation/voxel
metadata/classification/phase_separation/PE_drop
```

`metadata/paths` stores current file locations. `metadata/source` stores
ancestry such as parent state paths and source data versions. Parent groups
such as `metadata` and `metadata/classification` are containers only and should
not store attributes directly.

Thermalized-state logs use a leaner schema. Their standard folder and filename
layout replaces `metadata/paths`; `metadata/state` omits `data_version` and the
derived `fcc_cell_size`; and `metadata/run` omits `final_timestep`, which is
identical to `nsteps` for these runs. The thermalized voxel classifier stores
only its method, binning and threshold configuration, decision, and
`low_density_fraction`.

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
hot_spike.py             localized excitation creation and public runner
excitation_evolution.py  two-segment dt changes, stitching, legacy archiving
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
sweep at nearby state points. The helper holds an absolute starting radius
fixed in simulation length units, measures the trajectory, classifies survival
from the final 20% of frames, and writes a CSV row after every completed run.

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
    radius=3.0,
    evolve_seeds=[1, 2, 3],
)
```

The primary `outcome` uses the voxel classification of the final cavitation
state: `phase_separated=True` is `stabilized`, while `False` is
`rethermalized`. Radial tracking remains available as a separate diagnostic in
`radius_outcome`: `persisted` when the tail-median radius is at least 50% of the
constructed radius, `collapsed` when it is at most 10%, and `intermediate`
otherwise. Because the absolute radius is fixed, changing `n_fcc_cells`
isolates finite-size effects rather than changing the constructed bubble.

Before creating a bubble, cavitation checks the voxel phase-separation result
of the thermalized source. A phase-separated source is not cavitated. Its sweep
row has `run_status="thermalization_failed_phase_separated"`,
`thermalization_passed=False`, and `outcome="not_cavitated"`, together with the
source state/log paths and source low-density fraction.
