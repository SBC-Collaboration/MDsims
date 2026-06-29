"""
Helper package for the Lennard-Jones MD simulation workflow.

Primary modules:
- paths: data roots and V3 path builders
- spatial: shared periodic geometry and voxel calculations
- lattices: FCC lattice construction and loading
- simulation: HOOMD simulation setup and thermalization
- runs: logging, final-state saving, and trajectory runs
- classification: phase-separation and PE-drop classifiers
- cavitation: cavitated starting-state creation and evolution
- cavitation_analysis: trajectory and bubble measurements
- index: searchable V3 Parquet index creation
- metadata: structured HDF5 metadata helpers
- visualization: rendering and plotting
"""

__all__ = [
    "paths",
    "spatial",
    "lattices",
    "simulation",
    "runs",
    "classification",
    "cavitation",
    "cavitation_analysis",
    "index",
    "metadata",
    "visualization",
]
