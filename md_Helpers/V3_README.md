# V3 Saving Scheme

V3 keeps the existing readable folder style while making each saved object fit
one of three roles.

## Saved Object Types

```text
generated starting state -> one-frame GSD + creation metadata HDF5
evolved run              -> trajectory GSD + log HDF5 + optional final GSD
summary/search table     -> CSV or Parquet
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

## Migration

Use `V3_Migrate_Saved_Files.ipynb` from the repository root. Start with
`DRY_RUN = True`, review the migration plan, then rerun with `DRY_RUN = False`.

The migration upgrades metadata for lattices and thermalized states:

```text
Simple_Lattices_v3/.../lattice.gsd
Simple_Lattices_v3/.../lattice_metadata.hdf5

Thermalized_States_v3/.../randomization.gsd
Thermalized_States_v3/.../randomization_log.hdf5
```

For migrated thermalized logs, the original flat `metadata.attrs` are left in
place. V3 also writes grouped metadata so migrated logs and future V3 logs can
be read the same way:

```text
metadata/state
metadata/run
metadata/lj
metadata/source
metadata/paths
metadata/classification
```

The migration is copy-only. Original V2 files stay unchanged; the copied V3
HDF5 logs preserve the original flat `metadata.attrs` and add the grouped V3
metadata beside it.
