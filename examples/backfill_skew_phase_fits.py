"""Resumable backfill for the default skew-liquid phase-fit model."""

from __future__ import annotations

import argparse
from pathlib import Path

from md_Helpers import (
    ProjectPaths,
    SQLiteRunDatabase,
    backfill_skew_phase_fits,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--apply", action="store_true", help="write validated fits")
    result.add_argument(
        "--backup",
        type=Path,
        help="required SQLite backup destination when --apply is used",
    )
    result.add_argument("--run-id", action="append", dest="run_ids")
    result.add_argument("--limit", type=int)
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--max-attempts", type=int, default=3)
    return result


def main() -> None:
    arguments = parser().parse_args()
    paths = ProjectPaths()
    database = SQLiteRunDatabase(paths.database)
    database.initialize()
    if arguments.apply:
        if arguments.backup is None:
            raise SystemExit("--backup is required with --apply")
        backup = database.backup(arguments.backup)
        print(f"Database backup: {backup}")
    results = backfill_skew_phase_fits(
        database,
        project_paths=paths,
        run_ids=arguments.run_ids,
        limit=arguments.limit,
        dry_run=not arguments.apply,
        retry_failed=arguments.retry_failed,
        max_attempts=arguments.max_attempts,
    )
    print(results.to_string(index=False))
    if not results.empty:
        print("\nStatus totals:")
        print(results["Status"].value_counts().to_string())


if __name__ == "__main__":
    main()
