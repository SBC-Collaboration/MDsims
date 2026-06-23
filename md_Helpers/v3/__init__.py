"""
V3 helpers for modular MD state storage.

V3 keeps the existing project style of readable parameter folders, but makes
thermalized, cavitation, and excitation outputs follow the same state/run
rules:

- generated starting states: one-frame GSD + creation metadata HDF5
- evolved runs: trajectory GSD + log HDF5 + optional final-frame GSD
- searchable tables: CSV or Parquet summaries built from metadata
"""

__all__ = [
    "paths",
    "metadata",
    "migration",
    "classification",
    "runs",
]
