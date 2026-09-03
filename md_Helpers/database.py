"""Run-database interface with a local SQLite implementation.

The simulation workflow depends on this class, not on sqlite3 directly. A
MySQL implementation can therefore provide the same methods later without
changing thermalization, cavitation, or excitation code.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


MASTER_TABLE = "MD_Master"
THERMALIZATION_TABLE = "Thermalization"

MASTER_COLUMN_ORDER = (
    "Run_ID",
    "Run_Signature",
    "N_Cells",
    "Nsteps",
    "Current_Nstep",
    "ElapsedTime",
    "StartTime",
    "EndTime",
    "Last_Update_Time",
    "Sim_Type",
    "Status",
    "Stop_Reason",
    "Status_Message",
    "Notes",
)
MASTER_COLUMNS = set(MASTER_COLUMN_ORDER)

THERMALIZATION_COLUMNS = {
    "Run_ID",
    "File_Location",
    "Clone_Run_ID",
    "Clone_Frame_ID",
    "Therm_kT",
    "Therm_Seed",
    "Density_Start",
    "Density_End",
    "BoxLength_Start",
    "BoxLength_End",
    "dt",
    "Nsteps",
    "This_LJ_Time",
    "Cumulative_LJ_Time",
    "Ensemble",
    "T_Set",
    "P_Set",
    "LJ_r_cut",
    "LJ_r_on",
    "LJ_Mode",
    "Phase_Separation_Status",
    "Phase_Separation_Method",
    "Phase_Separation_Method_Version",
    "rho_liquid",
    "rho_liquid_unc",
    "rho_gas",
    "rho_gas_unc",
    "V_liquid",
    "V_liquid_unc",
    "V_gas",
    "V_gas_unc",
    "Phase_Fit_Status",
    "Phase_Fit_Method",
    "Phase_Fit_Method_Version",
    "Summary_Start_Step",
    "Summary_End_Step",
    "Summary_Num_Samples",
    "Pressure_Mean",
    "Pressure_Std",
    "Pressure_SEM",
    "PE_Per_Particle_Mean",
    "PE_Per_Particle_Std",
    "PE_Per_Particle_SEM",
    "Num_Frames",
}

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS MD_Master (
    Run_ID TEXT PRIMARY KEY,
    Run_Signature TEXT,
    N_Cells INTEGER,
    Nsteps INTEGER,
    Current_Nstep INTEGER,
    ElapsedTime REAL,
    StartTime TEXT,
    EndTime TEXT,
    Last_Update_Time TEXT,
    Sim_Type TEXT,
    Status TEXT,
    Stop_Reason TEXT,
    Status_Message TEXT,
    Notes TEXT,
    CHECK (Run_ID GLOB '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'),
    CHECK (N_Cells IS NULL OR N_Cells > 0),
    CHECK (Nsteps IS NULL OR Nsteps >= 0),
    CHECK (Current_Nstep IS NULL OR Current_Nstep >= 0),
    CHECK (ElapsedTime IS NULL OR ElapsedTime >= 0),
    CHECK (
        Sim_Type IS NULL OR Sim_Type IN (
            'Thermalization', 'Cavitation',
            'Excitation_NVE', 'Excitation_NPH'
        )
    ),
    CHECK (
        Status IS NULL OR Status IN (
            'Initializing', 'Running', 'Complete',
            'Safety_Stopped', 'Failed', 'Cancelled'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_MD_Master_Run_Signature
    ON MD_Master (Run_Signature);

CREATE INDEX IF NOT EXISTS idx_MD_Master_Type_Status
    ON MD_Master (Sim_Type, Status);

CREATE TABLE IF NOT EXISTS Thermalization (
    Run_ID TEXT PRIMARY KEY,
    File_Location TEXT NOT NULL,
    Clone_Run_ID TEXT,
    Clone_Frame_ID INTEGER,
    Therm_kT REAL NOT NULL,
    Therm_Seed INTEGER NOT NULL,
    Density_Start REAL NOT NULL,
    Density_End REAL NOT NULL,
    BoxLength_Start REAL NOT NULL,
    BoxLength_End REAL NOT NULL,
    dt REAL NOT NULL,
    Nsteps INTEGER NOT NULL,
    This_LJ_Time REAL NOT NULL,
    Cumulative_LJ_Time REAL NOT NULL,
    Ensemble TEXT NOT NULL,
    T_Set REAL,
    P_Set REAL,
    LJ_r_cut REAL NOT NULL,
    LJ_r_on REAL,
    LJ_Mode TEXT NOT NULL,
    Phase_Separation_Status TEXT NOT NULL,
    Phase_Separation_Method TEXT NOT NULL,
    Phase_Separation_Method_Version TEXT NOT NULL,
    rho_liquid REAL,
    rho_liquid_unc REAL,
    rho_gas REAL,
    rho_gas_unc REAL,
    V_liquid REAL,
    V_liquid_unc REAL,
    V_gas REAL,
    V_gas_unc REAL,
    Phase_Fit_Status TEXT NOT NULL,
    Phase_Fit_Method TEXT,
    Phase_Fit_Method_Version TEXT,
    Summary_Start_Step INTEGER NOT NULL,
    Summary_End_Step INTEGER NOT NULL,
    Summary_Num_Samples INTEGER NOT NULL,
    Pressure_Mean REAL,
    Pressure_Std REAL,
    Pressure_SEM REAL,
    PE_Per_Particle_Mean REAL,
    PE_Per_Particle_Std REAL,
    PE_Per_Particle_SEM REAL,
    Num_Frames INTEGER NOT NULL,
    FOREIGN KEY (Run_ID) REFERENCES MD_Master (Run_ID),
    FOREIGN KEY (Clone_Run_ID) REFERENCES MD_Master (Run_ID),
    CHECK (Clone_Frame_ID IS NULL OR Clone_Frame_ID >= 0),
    CHECK (Therm_kT > 0),
    CHECK (Density_Start > 0 AND Density_End > 0),
    CHECK (BoxLength_Start > 0 AND BoxLength_End > 0),
    CHECK (dt > 0),
    CHECK (Nsteps >= 0),
    CHECK (Summary_Num_Samples > 0),
    CHECK (Num_Frames >= 2)
);
"""


def utc_now() -> str:
    """Return a MySQL-compatible, microsecond-resolution UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def run_id_now() -> str:
    """Return a Run ID candidate formatted as YYYYMMDDHHMMSS in UTC."""

    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class SQLiteRunDatabase:
    """Short-lived SQLite connections for local workflow development."""

    def __init__(self, path: str | Path, timeout: float = 30.0):
        self.path = Path(path).expanduser().resolve()
        self.timeout = float(timeout)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open, transact on, and always close one database connection."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SQLITE_SCHEMA)
            connection.execute("PRAGMA user_version = 1")

    def check_run_exists(self, run_signature: str) -> dict[str, Any] | None:
        """Return an existing Master row without opening any run files."""

        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM MD_Master
                WHERE Run_Signature = ?
                ORDER BY Run_ID
                LIMIT 1
                """,
                (str(run_signature),),
            ).fetchone()
        return _row_dict(row)

    def reserve_run_id(self, max_attempts: int = 3) -> str:
        """Insert a row containing only Run_ID, retrying timestamp collisions."""

        max_attempts = int(max_attempts)
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        for attempt in range(max_attempts):
            run_id = run_id_now()
            try:
                with self.connection() as connection:
                    connection.execute(
                        "INSERT INTO MD_Master (Run_ID) VALUES (?)",
                        (run_id,),
                    )
                return run_id
            except sqlite3.IntegrityError as error:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        "Could not reserve a unique Run_ID after "
                        f"{max_attempts} attempts"
                    ) from error
                time.sleep(1.0)

        raise RuntimeError("Run_ID reservation ended unexpectedly")

    def update_master(self, run_id: str, **values: Any) -> None:
        if not values:
            return
        unknown = set(values) - (MASTER_COLUMNS - {"Run_ID"})
        if unknown:
            raise ValueError(f"Unknown Master columns: {sorted(unknown)}")

        assignments = ", ".join(f"{column} = ?" for column in values)
        parameters = [*values.values(), str(run_id)]
        with self.connection() as connection:
            cursor = connection.execute(
                f"UPDATE MD_Master SET {assignments} WHERE Run_ID = ?",
                parameters,
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Run_ID was not found: {run_id}")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM MD_Master WHERE Run_ID = ?",
                (str(run_id),),
            ).fetchone()
        return _row_dict(row)

    def query_runs(self, **filters: Any) -> list[dict[str, Any]]:
        """Query Master rows using exact-match filters."""

        unknown = set(filters) - MASTER_COLUMNS
        if unknown:
            raise ValueError(f"Unknown Master filters: {sorted(unknown)}")

        sql = "SELECT * FROM MD_Master"
        parameters: list[Any] = []
        if filters:
            clauses = []
            for column, value in filters.items():
                if value is None:
                    clauses.append(f"{column} IS NULL")
                else:
                    clauses.append(f"{column} = ?")
                    parameters.append(value)
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY Run_ID"

        with self.connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def complete_thermalization(
        self,
        run_id: str,
        thermalization: dict[str, Any],
        master: dict[str, Any],
    ) -> None:
        """Insert results and mark Master complete in one transaction."""

        thermalization = {"Run_ID": str(run_id), **thermalization}
        unknown_thermal = set(thermalization) - THERMALIZATION_COLUMNS
        unknown_master = set(master) - (MASTER_COLUMNS - {"Run_ID"})
        if unknown_thermal:
            raise ValueError(
                f"Unknown Thermalization columns: {sorted(unknown_thermal)}"
            )
        if unknown_master:
            raise ValueError(f"Unknown Master columns: {sorted(unknown_master)}")

        thermal_columns = list(thermalization)
        thermal_placeholders = ", ".join("?" for _ in thermal_columns)
        master_assignments = ", ".join(
            f"{column} = ?" for column in master
        )

        with self.connection() as connection:
            connection.execute(
                f"""
                INSERT INTO Thermalization ({', '.join(thermal_columns)})
                VALUES ({thermal_placeholders})
                """,
                list(thermalization.values()),
            )
            cursor = connection.execute(
                f"""
                UPDATE MD_Master
                SET {master_assignments}
                WHERE Run_ID = ?
                """,
                [*master.values(), str(run_id)],
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Run_ID was not found: {run_id}")


def master_dataframe(database: SQLiteRunDatabase):
    """Return every Master row as a pandas table in canonical column order."""

    import pandas as pd

    rows = database.query_runs()
    return pd.DataFrame.from_records(
        rows,
        columns=MASTER_COLUMN_ORDER,
    ).convert_dtypes()


def display_master_table(
    database: SQLiteRunDatabase | None = None,
    project_paths=None,
):
    """Display the complete Master table cleanly in a Jupyter notebook.

    The returned DataFrame can also be filtered or reused by the caller.
    """

    import pandas as pd
    from IPython.display import display

    if database is None:
        from .paths import ProjectPaths

        project_paths = project_paths or ProjectPaths()
        database = SQLiteRunDatabase(project_paths.database)
    database.initialize()
    table = master_dataframe(database)

    integer_columns = ["N_Cells", "Nsteps", "Current_Nstep"]
    float_columns = ["ElapsedTime"]
    numeric_columns = integer_columns + float_columns
    text_columns = [
        column for column in MASTER_COLUMN_ORDER if column not in numeric_columns
    ]

    formatters = {
        **{column: "{:,.0f}" for column in integer_columns},
        **{column: "{:,.3f}" for column in float_columns},
    }
    styled = (
        table.style.hide(axis="index")
        .format(formatters, na_rep="NULL")
        .set_properties(
            subset=numeric_columns,
            **{"text-align": "right", "white-space": "nowrap"},
        )
        .set_properties(
            subset=text_columns,
            **{"text-align": "left", "white-space": "nowrap"},
        )
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                    ("white-space", "nowrap"),
                ],
            }
        ])
    )

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
    ):
        display(styled)
    return table
