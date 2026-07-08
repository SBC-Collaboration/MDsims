"""
Helper package for the Lennard-Jones MD simulation workflow.

Primary modules:
- paths: data roots and V3 path builders
- spatial: shared periodic geometry and voxel calculations
- lattices: FCC lattice construction and loading
- simulation: HOOMD simulation setup and thermalization
- runs: logging, final-state saving, and trajectory runs
- classification: phase-separation and PE-drop classifiers
- voxel_fit: gas, liquid, and interface voxel-mixture fitting
- cavitation: cavitated starting-state creation and evolution
- cavitation_analysis: trajectory and bubble measurements
- cavitation_sweep: fixed-radius cavitation parameter sweeps
- index: searchable V3 Parquet index creation
- master_csv: rebuildable CSV summaries for completed runs
- metadata: structured HDF5 metadata helpers
- seitz: Seitz threshold calculation helpers
- visualization: rendering and plotting
"""

__all__ = [
    "paths",
    "spatial",
    "lattices",
    "simulation",
    "runs",
    "classification",
    "voxel_fit",
    "cavitation",
    "cavitation_analysis",
    "cavitation_sweep",
    "index",
    "master_csv",
    "metadata",
    "seitz",
    "visualization",
]
