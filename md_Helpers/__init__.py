"""
Helper package for the Lennard-Jones MD simulation workflow.

Primary modules:
- paths: data roots and V3 path builders
- lattices: FCC lattice construction and loading
- simulation: HOOMD simulation setup and thermalization
- runs: logging, final-state saving, and trajectory runs
- classification: phase-separation and PE-drop classifiers
- cavitation: cavitated starting-state creation
- database: sweep summaries and master CSV builders
- metadata: structured HDF5 metadata helpers
- visualization: rendering and plotting
- migration: temporary V2-to-V3 data migration helpers
"""

__all__ = [
    "paths",
    "lattices",
    "simulation",
    "runs",
    "classification",
    "cavitation",
    "database",
    "metadata",
    "visualization",
    "migration",
]
