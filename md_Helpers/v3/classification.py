from pathlib import Path

from .. import Phase_Separation as ps


def classify_final_state(
    state_path,
    log_path,
    nbins=ps.DEFAULT_PHASE_SEP_NBINS,
    density_threshold=ps.DEFAULT_PHASE_SEP_DENSITY_THRESHOLD,
    voxel_fraction_threshold=ps.DEFAULT_PHASE_SEP_VOXEL_FRACTION_THRESHOLD,
    dry_run=False,
):
    """
    Run the current voxel phase-separation code on any final state.

    Intended inputs:
    - thermalized/randomized final states
    - cavitation evolved final states
    - excitation evolved final states

    This is intentionally not meant for artificial starting states such as
    cavitation_initial.gsd or excitation_initial.gsd.
    """

    return ps.write_voxel_phase_separation_metadata(
        log_path=Path(log_path),
        state_path=Path(state_path),
        nbins=nbins,
        density_threshold=density_threshold,
        voxel_fraction_threshold=voxel_fraction_threshold,
        updated_from_saved_gsd=True,
        dry_run=dry_run,
    )


def classify_PE_drop(
    log_path,
    dry_run=False,
    **kwargs,
):
    """
    Run the current PE-drop classifier on any evolved run log.
    """

    result = ps.compute_PE_drop_phase_separation_from_log(
        log_path=Path(log_path),
        **kwargs,
    )

    if dry_run:
        return result

    ps.write_PE_drop_phase_separation_metadata(
        log_path=Path(log_path),
        **kwargs,
    )

    return result
