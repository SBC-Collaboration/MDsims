# lattices.py

import numpy as np
import gsd.hoomd
from pathlib import Path

from .paths import SIMPLE_LATTICES_V3_ROOT


# ============================================================
# Get FCC lattice information
# ============================================================

def get_fcc_lattice_info(
    n_fcc_cells,
    target_rho,
):
    """
    Compute the derived FCC lattice quantities.

    V3 convention:
    - n_fcc_cells is the number of conventional FCC unit cells per side
    - each FCC cell contains 4 particles
    - N is fixed by n_fcc_cells
    - BoxLength is chosen to give the requested target_rho exactly

    Parameters
    ----------
    n_fcc_cells : int
        Number of FCC unit cells along each side of the cubic box.

    target_rho : float
        Requested number density.

    Returns
    -------
    info : dict
        Dictionary containing N, BoxLength, fcc_cell_size, volume,
        target_rho, and actual_rho.
    """

    # ============================================================
    # Validate inputs
    # ============================================================
    n_fcc_cells = int(n_fcc_cells)

    if n_fcc_cells <= 0:
        raise ValueError("n_fcc_cells must be a positive integer")

    if target_rho <= 0:
        raise ValueError("target_rho must be positive")

    # ============================================================
    # FCC particle count and box size
    # ============================================================

    N = 4 * n_fcc_cells**3

    volume = N / target_rho

    BoxLength = volume ** (1.0 / 3.0)

    fcc_cell_size = BoxLength / n_fcc_cells

    actual_rho = N / BoxLength**3

    # ============================================================
    # Return useful quantities
    # ============================================================
    info = {
        "lattice_type": "fcc",
        "density_mode": "fixed_N_variable_L",
        "n_fcc_cells": n_fcc_cells,
        "N": N,
        "target_rho": target_rho,
        "actual_rho": actual_rho,
        "BoxLength": BoxLength,
        "volume": volume,
        "fcc_cell_size": fcc_cell_size,
    }

    return info




# ============================================================
# Fill FCC lattice by number of FCC cells
# ============================================================

def FillFccLattice(
    n_fcc_cells,
    target_rho,
):
    """
    Build an FCC lattice using the same parity/checkerboard style
    as the original source code.

    V3 convention:
    - n_fcc_cells is the number of conventional FCC cells per side
    - N = 4 * n_fcc_cells**3
    - BoxLength is chosen from target_rho
    - the fine grid has n_grid = 2 * n_fcc_cells points per side
    - keep only sites where i + j + k is even

    Returns
    -------
    positions : np.ndarray
        Array of shape (N, 3) containing particle positions.

    info : dict
        Lattice information from get_fcc_lattice_info().
    """

    # ============================================================
    # Get lattice information
    # ============================================================
    info = get_fcc_lattice_info(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
    )

    n_fcc_cells = info["n_fcc_cells"]
    BoxLength = info["BoxLength"]

    # ============================================================
    # Old-style FCC construction
    #
    # The fine grid has twice as many points per side as the number
    # of conventional FCC cells.
    #
    # Keeping sites with even i + j + k produces the FCC lattice.
    # ============================================================
    n_grid = 2 * n_fcc_cells

    grid_spacing = BoxLength / n_grid

    # This is the conventional FCC unit-cell side length.
    fcc_cell_size = 2.0 * grid_spacing

    # ============================================================
    # Build integer grid indices
    # ============================================================
    vec_i = np.arange(n_grid)

    pos_i = np.zeros(
        (n_grid * n_grid * n_grid, 3),
        dtype=np.intp,
    )

    pos_i_reshaped = pos_i.reshape(
        (n_grid, n_grid, n_grid, 3)
    )

    pos_i_reshaped[:, :, :, 0] = vec_i[:, None, None]
    pos_i_reshaped[:, :, :, 1] = vec_i[None, :, None]
    pos_i_reshaped[:, :, :, 2] = vec_i[None, None, :]

    # ============================================================
    # FCC parity cut
    # ============================================================
    fcc_cut = np.mod(
        np.sum(pos_i, axis=1),
        2,
    ) == 0

    # ============================================================
    # Convert grid indices to positions
    # ============================================================
    offset = -BoxLength / 2.0

    positions = (
        pos_i[fcc_cut].astype(np.float64) * grid_spacing
        + np.array([offset, offset, offset])
    )

    # ============================================================
    # Sanity check
    # ============================================================
    expected_N = info["N"]

    if len(positions) != expected_N:
        raise RuntimeError(
            f"FCC lattice has wrong number of particles: "
            f"got {len(positions)}, expected {expected_N}"
        )

    # ============================================================
    # Make sure info stores the same FCC cell size used here
    # ============================================================
    info["fcc_cell_size"] = fcc_cell_size
    info["grid_spacing"] = grid_spacing
    info["n_grid"] = n_grid

    return positions, info


# ============================================================
# Get lattice path
# ============================================================

def get_lattice_path(
    n_fcc_cells,
    target_rho,
    base_folder=SIMPLE_LATTICES_V3_ROOT,
):
    """
    Build the V3 lattice path.

    Folder structure:

        Simple_Lattices_v3/
            FCC/
                n_cells_30/
                    rho_0.500/
                        lattice.gsd
    """

    n_cells_str = f"{int(n_fcc_cells)}"
    rho_str = f"{target_rho:.3f}"

    return (
        Path(base_folder)
        / "FCC"
        / f"n_cells_{n_cells_str}"
        / f"rho_{rho_str}"
        / "lattice.gsd"
    )


# ============================================================
# Make or load lattice frame
# ============================================================

def make_lattice_frame(
    n_fcc_cells,
    target_rho,
    particle_type="A",
    end_print=True,
    overwrite=False,
    base_folder=SIMPLE_LATTICES_V3_ROOT,
):
    """
    Create or load an FCC lattice frame using the V3 convention.

    V3 convention:
    - User chooses n_fcc_cells
    - User chooses target_rho
    - N is fixed by n_fcc_cells
    - BoxLength is derived from N / target_rho

    Parameters
    ----------
    n_fcc_cells : int
        Number of FCC cells per side.

    target_rho : float
        Requested density.

    particle_type : str
        HOOMD particle type name.

    end_print : bool
        Whether to print summary information.

    overwrite : bool
        If True, rebuild and overwrite an existing lattice file.

    base_folder : str or pathlib.Path
        Root folder for lattice database.

    Returns
    -------
    frame : gsd.hoomd.Frame
        Single-frame GSD object containing the lattice.
    """

    # ============================================================
    # Build lattice path
    # ============================================================
    filepath = get_lattice_path(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
        base_folder=base_folder,
    )

    # ============================================================
    # Load existing file if present
    # ============================================================
    if filepath.exists() and not overwrite:
        if end_print:
            print("Loaded existing FCC lattice:")
            print(filepath)

        with gsd.hoomd.open(name=str(filepath), mode="r") as f:
            frame = f[0]

        return frame

    # ============================================================
    # Create new FCC lattice
    # ============================================================
    positions, info = FillFccLattice(
        n_fcc_cells=n_fcc_cells,
        target_rho=target_rho,
    )

    BoxLength = info["BoxLength"]
    N = info["N"]
    actual_rho = info["actual_rho"]
    fcc_cell_size = info["fcc_cell_size"]

    # ============================================================
    # Build GSD frame
    # ============================================================
    frame = gsd.hoomd.Frame()

    frame.configuration.step = 0
    frame.configuration.box = [
        BoxLength,
        BoxLength,
        BoxLength,
        0,
        0,
        0,
    ]

    frame.particles.N = N
    frame.particles.types = [particle_type]
    frame.particles.position = positions
    frame.particles.typeid = np.zeros(N, dtype=np.uint32)

    # ============================================================
    # Print info
    # ============================================================
    if end_print:
        print("Created new FCC lattice")
        print("----------------------------")
        print("n_fcc_cells =", info["n_fcc_cells"])
        print("N =", N)
        print("Target rho =", target_rho)
        print("Actual rho =", actual_rho)
        print("Density error =", actual_rho - target_rho)
        print("BoxLength =", BoxLength)
        print("FCC cell size =", fcc_cell_size)

    # ============================================================
    # Save GSD
    # ============================================================
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(filepath), mode="w") as f:
        f.append(frame)

    if end_print:
        print("Saved lattice to:")
        print(filepath)

    return frame
