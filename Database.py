import hashlib
import json
from datetime import datetime, timezone

def create_run_signature(ncell, kT, rho):
    """Create a deterministic SHA-256 signature for a thermalization setup."""
    
    parameters = {
        "n_cells": int(ncell),
        "therm_kT": float(kT),
        "density": float(rho),
    }

    canonical_parameters = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        canonical_parameters.encode("utf-8")
    ).hexdigest()







def create_run_id():
    """Return a UTC Run ID formatted as YYYYMMDDHHMMSS."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
