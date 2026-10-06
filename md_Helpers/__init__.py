"""Small, reusable helpers for the V4 molecular-dynamics workflows."""

from .argon_scaling import (
    ArgonLJScale,
    ArgonScaleFit,
    fit_argon_lj_scale,
    query_argon_scaling_states,
)

from .database import (
    SQLiteRunDatabase,
    cavitation_dataframe,
    display_cavitation_table,
    display_master_table,
    display_thermalization_table,
    master_dataframe,
    thermalization_dataframe,
)
from .cavitation import CavitationConfig, run_cavitation
from .expanded_fcc import (
    ExpandedFCCConfig,
    build_expanded_fcc_lattice,
    expanded_fcc_frame_schedule,
    recenter_snapshot_arrays,
    run_expanded_fcc,
)
from .expanded_clone import (
    ExpandedCloneConfig,
    build_expanded_clone_state,
    run_expanded_clone,
)
from .paths import ProjectPaths
from .run_analysis import RunAnalysis, open_run
from .run_management import delete_run
from .seitz import (
    calculate_cavitation_seitz,
    plot_cavitation_seitz,
    query_seitz_eos_states,
    seitz_threshold,
    seitz_threshold_uncertainty,
)
from .nbins_tuning import (
    cavitations_with_nbins_fit,
    fit_liquid_nbins,
    plot_liquid_gaussian_mean_vs_nbins,
    plot_liquid_nbins_gaussians,
    plot_phase_liquid_density_vs_nbins,
    plot_phase_nbins_mixtures,
    plot_nbins_phase_fits,
    plot_nbins_seitz,
    refit_cavitation_nbins,
    refit_skewed_phase_nbins,
)
from .thermalization import (
    CloneRescaleThermalizationConfig,
    ThermalizationConfig,
    run_clone_rescale_constant_volume_rate,
    run_clone_rescale_ensemble,
    run_clone_rescale_thermalization,
    run_thermalization,
)

__all__ = [
    "ArgonLJScale",
    "ArgonScaleFit",
    "ProjectPaths",
    "RunAnalysis",
    "SQLiteRunDatabase",
    "CloneRescaleThermalizationConfig",
    "CavitationConfig",
    "ExpandedFCCConfig",
    "ExpandedCloneConfig",
    "ThermalizationConfig",
    "cavitation_dataframe",
    "calculate_cavitation_seitz",
    "build_expanded_fcc_lattice",
    "backfill_skew_phase_fits",
    "create_skew_staging_database",
    "build_expanded_clone_state",
    "expanded_fcc_frame_schedule",
    "cavitations_with_nbins_fit",
    "display_cavitation_table",
    "display_master_table",
    "display_thermalization_table",
    "delete_run",
    "master_dataframe",
    "fit_argon_lj_scale",
    "fit_liquid_nbins",
    "open_run",
    "plot_cavitation_seitz",
    "plot_liquid_gaussian_mean_vs_nbins",
    "plot_liquid_nbins_gaussians",
    "plot_phase_liquid_density_vs_nbins",
    "plot_phase_nbins_mixtures",
    "promote_skew_phase_fits",
    "plot_nbins_phase_fits",
    "plot_nbins_seitz",
    "query_seitz_eos_states",
    "query_argon_scaling_states",
    "refit_cavitation_nbins",
    "refit_skewed_phase_nbins",
    "recenter_snapshot_arrays",
    "seitz_threshold",
    "seitz_threshold_uncertainty",
    "thermalization_dataframe",
    "run_clone_rescale_constant_volume_rate",
    "run_clone_rescale_ensemble",
    "run_clone_rescale_thermalization",
    "run_thermalization",
    "run_cavitation",
    "run_expanded_fcc",
    "run_expanded_clone",
]


def backfill_skew_phase_fits(*args, **kwargs):
    """Load the optional migration machinery only when it is requested."""

    from .phase_fit_backfill import backfill_skew_phase_fits as implementation

    return implementation(*args, **kwargs)


def create_skew_staging_database(*args, **kwargs):
    """Lazily create a shadow database for phase-fit migration."""

    from .phase_fit_backfill import create_skew_staging_database as implementation

    return implementation(*args, **kwargs)


def promote_skew_phase_fits(*args, **kwargs):
    """Lazily promote validated shadow-database fits into production."""

    from .phase_fit_backfill import promote_skew_phase_fits as implementation

    return implementation(*args, **kwargs)
