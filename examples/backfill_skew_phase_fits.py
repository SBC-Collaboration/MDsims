"""Resumable backfill for the default skew-liquid phase-fit model."""

from __future__ import annotations

import argparse
from pathlib import Path

from md_Helpers import (
    ProjectPaths,
    SQLiteRunDatabase,
    backfill_skew_phase_fits,
    create_skew_staging_database,
    promote_skew_phase_fits,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    action = result.add_mutually_exclusive_group()
    action.add_argument(
        "--apply",
        action="store_true",
        help="write validated fits to the shadow database only",
    )
    action.add_argument(
        "--promote",
        action="store_true",
        help="promote completed shadow fits into production SQL and HDF5",
    )
    result.add_argument(
        "--staging-database",
        type=Path,
        help="shadow SQLite path (default: production name plus -skew-staging)",
    )
    result.add_argument(
        "--backup",
        type=Path,
        help="required production SQLite backup destination with --promote",
    )
    result.add_argument("--run-id", action="append", dest="run_ids")
    result.add_argument("--limit", type=int)
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--max-attempts", type=int, default=3)
    return result


def main() -> None:
    arguments = parser().parse_args()
    paths = ProjectPaths()
    production = SQLiteRunDatabase(paths.database)
    production.initialize()
    staging_path = arguments.staging_database
    if staging_path is None:
        staging_path = production.path.with_name(
            f"{production.path.stem}-skew-staging{production.path.suffix}"
        )
    staging_path = staging_path.expanduser().resolve()
    if staging_path == production.path.expanduser().resolve():
        raise SystemExit("The staging database must differ from production.")
    if arguments.promote and not staging_path.exists():
        raise SystemExit(
            f"Cannot promote because the shadow database does not exist: "
            f"{staging_path}"
        )
    if staging_path.exists():
        staging = SQLiteRunDatabase(staging_path, timeout=production.timeout)
        staging.initialize()
    else:
        staging = create_skew_staging_database(production, staging_path)
        print(f"Created shadow database: {staging.path}")

    if arguments.promote:
        if arguments.backup is None:
            raise SystemExit("--backup is required with --promote")
        results = promote_skew_phase_fits(
            staging,
            production,
            backup_path=arguments.backup,
            project_paths=paths,
            run_ids=arguments.run_ids,
            limit=arguments.limit,
        )
        print(f"Production backup: {results.attrs['production_backup']}")
    else:
        results = backfill_skew_phase_fits(
            staging,
            project_paths=paths,
            run_ids=arguments.run_ids,
            limit=arguments.limit,
            dry_run=not arguments.apply,
            retry_failed=arguments.retry_failed,
            max_attempts=arguments.max_attempts,
            write_hdf5_metadata=False,
        )
    print(results.to_string(index=False))
    if not results.empty:
        print("\nStatus totals:")
        print(results["Status"].value_counts().to_string())


if __name__ == "__main__":
    main()
