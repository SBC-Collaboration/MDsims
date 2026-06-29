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
cavitation.py            cavitation creation and evolution
cavitation_analysis.py   trajectory bubble measurements
visualization.py         plotting and animation
index.py                 searchable V3 Parquet index
metadata.py              structured HDF5 metadata I/O
```

Build the searchable index with:

```python
from md_Helpers import index

table = index.build_v3_index()
```
