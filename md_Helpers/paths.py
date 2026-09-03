"""One place to configure every V4 output path."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Change this one value for a different local output root. On the shared system,
# this will eventually become Path("/exp/e961/data/MDsims-data/pnichols/SQL").
TOP_DIRECTORY = Path("/exp/e961/data/MDsims-data/pnichols/SQL")


def _default_top_directory() -> Path:
    configured = os.environ.get("MDSIMS_TOP_DIRECTORY")
    return Path(configured).expanduser() if configured else TOP_DIRECTORY


@dataclass(frozen=True)
class RunPaths:
    """Canonical files for one run."""

    run_id: str
    sim_type: str
    top_directory: Path

    @property
    def relative_directory(self) -> Path:
        return Path(self.sim_type) / self.run_id

    @property
    def directory(self) -> Path:
        return self.top_directory / self.relative_directory

    @property
    def trajectory(self) -> Path:
        return self.directory / "trajectory.gsd"

    @property
    def hdf5(self) -> Path:
        return self.directory / "run.hdf5"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve database and simulation paths from one configurable root."""

    top_directory: Path = field(default_factory=_default_top_directory)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "top_directory",
            Path(self.top_directory).expanduser().resolve(),
        )

    @property
    def database(self) -> Path:
        return self.top_directory / "mdsims.sqlite3"

    def for_run(self, sim_type: str, run_id: str) -> RunPaths:
        return RunPaths(
            run_id=str(run_id),
            sim_type=str(sim_type),
            top_directory=self.top_directory,
        )
