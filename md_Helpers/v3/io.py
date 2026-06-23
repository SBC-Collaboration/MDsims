import os
import shutil
from pathlib import Path


def copy_or_link_file(
    source_path,
    destination_path,
    link_mode="copy",
    overwrite=False,
    fallback_to_copy=True,
):
    """
    Move existing saved data into the V3 layout without rerunning simulations.

    link_mode options:
    - "copy": duplicate the file with metadata preserved
    - "hardlink": no duplicate storage on the same filesystem
    - "symlink": destination points to the old file
    """

    source_path = Path(source_path)
    destination_path = Path(destination_path)

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        if not overwrite:
            return {
                "action": "skipped_exists",
                "source_path": source_path,
                "destination_path": destination_path,
            }

        destination_path.unlink()

    try:
        if link_mode == "copy":
            shutil.copy2(source_path, destination_path)
            action = "copied"

        elif link_mode == "hardlink":
            os.link(source_path, destination_path)
            action = "hardlinked"

        elif link_mode == "symlink":
            destination_path.symlink_to(source_path)
            action = "symlinked"

        else:
            raise ValueError(
                "link_mode must be one of: copy, hardlink, symlink"
            )

    except OSError:
        if not fallback_to_copy or link_mode == "copy":
            raise

        shutil.copy2(source_path, destination_path)
        action = "copied_fallback"

    return {
        "action": action,
        "source_path": source_path,
        "destination_path": destination_path,
    }


def save_last_frame_as_gsd(trajectory_path, final_state_path, overwrite=False):
    import gsd.hoomd

    trajectory_path = Path(trajectory_path)
    final_state_path = Path(final_state_path)

    if final_state_path.exists() and not overwrite:
        return final_state_path

    final_state_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(trajectory_path), mode="r") as trajectory:
        frame = trajectory[-1]

    with gsd.hoomd.open(name=str(final_state_path), mode="w") as final_file:
        final_file.append(frame)

    return final_state_path
