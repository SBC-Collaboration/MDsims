from pathlib import Path

import hoomd

from .. import Logging_Helpers as lh
from . import metadata as v3_metadata
from .classification import classify_final_state


def start_gsd_trajectory_writer(
    simulation,
    trajectory_path,
    trajectory_period=1_000,
    mode="wb",
):
    """
    Start a many-frame GSD trajectory writer for evolved V3 runs.
    """

    trajectory_path = Path(trajectory_path)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    writer = hoomd.write.GSD(
        filename=str(trajectory_path),
        trigger=hoomd.trigger.Periodic(int(trajectory_period)),
        mode=mode,
        filter=hoomd.filter.All(),
        dynamic=[
            "property",
            "momentum",
        ],
    )

    simulation.operations.writers.append(writer)

    return {
        "writer": writer,
        "trajectory_path": trajectory_path,
        "trajectory_period": int(trajectory_period),
    }


def stop_gsd_trajectory_writer(simulation, writer_handle):
    writer = writer_handle["writer"]

    if writer in simulation.operations.writers:
        simulation.operations.writers.remove(writer)


def run_logged_trajectory_phase(
    simulation,
    nsteps,
    log_path,
    trajectory_path,
    final_state_path=None,
    log_period=1_000,
    trajectory_period=1_000,
    metadata_groups=None,
    classify_final=True,
    classification_kwargs=None,
):
    """
    Run any evolved V3 phase with one shared pattern:

    - HDF5 thermodynamic log
    - many-frame GSD trajectory
    - optional one-frame final GSD
    - optional phase classification on the final state

    This is the common runner for future cavitation_evolved and
    excitation_evolved workflows.
    """

    log_path = Path(log_path)
    trajectory_path = Path(trajectory_path)

    if final_state_path is not None:
        final_state_path = Path(final_state_path)

    logger_handle = lh.start_hdf5_logger(
        simulation=simulation,
        log_path=log_path,
        log_period=log_period,
    )

    trajectory_handle = start_gsd_trajectory_writer(
        simulation=simulation,
        trajectory_path=trajectory_path,
        trajectory_period=trajectory_period,
    )

    simulation.run(0)
    simulation.run(int(nsteps))

    stop_gsd_trajectory_writer(
        simulation=simulation,
        writer_handle=trajectory_handle,
    )

    lh.stop_hdf5_logger(
        simulation=simulation,
        logger_objects=logger_handle,
    )

    if final_state_path is not None:
        lh.save_final_state(
            simulation=simulation,
            gsd_path=final_state_path,
        )

    if metadata_groups:
        v3_metadata.write_metadata_groups(
            hdf5_path=log_path,
            groups=metadata_groups,
            mode="a",
            overwrite=True,
        )

    classification_result = None

    if classify_final and final_state_path is not None:
        classification_kwargs = classification_kwargs or {}
        classification_result = classify_final_state(
            state_path=final_state_path,
            log_path=log_path,
            **classification_kwargs,
        )

    return {
        "log_path": log_path,
        "trajectory_path": trajectory_path,
        "final_state_path": final_state_path,
        "classification_result": classification_result,
    }
