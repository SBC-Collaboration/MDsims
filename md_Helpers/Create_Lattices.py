#Create_Lattices.py

import numpy as np
import gsd.hoomd
from pathlib import Path

from .Project_Paths import SIMPLE_LATTICES_ROOT



def FillBoxCubicLattice(BoxLength, rho):
    cellsize = rho**(-1.0 / 3.0)

    n = int(np.floor(BoxLength / cellsize))

    offset = -BoxLength / 2 + 0.5 * (BoxLength - n * cellsize)

    vec = cellsize * np.arange(n, dtype=np.float64) + offset

    positions = np.zeros((n * n * n, 3), dtype=np.float64)
    pos = positions.reshape((n, n, n, 3))

    pos[:, :, :, 0] = vec[:, None, None]
    pos[:, :, :, 1] = vec[None, :, None]
    pos[:, :, :, 2] = vec[None, None, :]

    return positions



def FillBoxFccLattice(BoxLength, rho):
    cellsize = (2.0 * rho) ** (-1.0 / 3.0)

    n = int(np.floor(BoxLength / 2 / cellsize)*2)

    cellsize = BoxLength/n

    offset = -BoxLength / 2 + 0.5 * (BoxLength - n * cellsize)

    numparticles = n * n * n
    vec_i = np.arange(n)

    pos_i = np.zeros((numparticles, 3), dtype=np.intp)
    pos_i_reshaped = pos_i.reshape((n, n, n, 3))

    pos_i_reshaped[:, :, :, 0] = vec_i[:, None, None]
    pos_i_reshaped[:, :, :, 1] = vec_i[None, :, None]
    pos_i_reshaped[:, :, :, 2] = vec_i[None, None, :]

    fcc_cut = np.mod(np.sum(pos_i, axis=1), 2) == 0

    positions = (
        pos_i[fcc_cut].astype(np.float64) * cellsize
        + np.array([offset, offset, offset])
    )

    return positions



def get_lattice_path(BoxLength, rho, lattice_type="fcc", base_folder=SIMPLE_LATTICES_ROOT):
    BoxLength_str = f"{BoxLength:.1f}"
    rho_str = f"{rho:.2f}"

    if lattice_type == "cubic":
        lattice_folder_name = "Cubic"
    elif lattice_type == "fcc":
        lattice_folder_name = "FCC"
    else:
        raise ValueError("lattice_type must be 'cubic' or 'fcc'")

    return (
        Path(base_folder)
        / lattice_folder_name
        / f"BoxLength_{BoxLength_str}"
        / f"rho_{rho_str}"
        / "lattice.gsd"
    )



def make_lattice_frame(
    BoxLength,
    rho,
    lattice_type="fcc",
    particle_type="A",
    end_print=True,
    base_folder=SIMPLE_LATTICES_ROOT,
    return_path=False,
):
    filepath = get_lattice_path(
        BoxLength=BoxLength,
        rho=rho,
        lattice_type=lattice_type,
        base_folder=base_folder,
    )

    # Load existing file if it exists
    if filepath.exists():
        if end_print:
            print("Loaded existing lattice:")
            print(filepath)

        with gsd.hoomd.open(name=str(filepath), mode="r") as f:
            frame = f[0]

        if return_path:
            return frame, filepath
        
        return frame


    #Otherwise create a new lattice and save it into proper folder
    if lattice_type == "cubic":
        positions = FillBoxCubicLattice(BoxLength, rho)

    elif lattice_type == "fcc":
        positions = FillBoxFccLattice(BoxLength, rho)

    else:
        raise ValueError("lattice_type must be 'cubic' or 'fcc'")

    frame = gsd.hoomd.Frame()
    frame.particles.N = len(positions)
    frame.particles.types = [particle_type]
    frame.particles.position = positions
    frame.particles.typeid = [0] * len(positions)
    frame.configuration.box = [BoxLength, BoxLength, BoxLength, 0, 0, 0]

    actual_rho = len(positions) / BoxLength**3

    # ============================================================
    # PRINT INFO
    # ============================================================
    if end_print:
        print(f"Created new {lattice_type} lattice")
        print("----------------------------")
        print("Target BoxLength =", BoxLength)
        print("Target rho =", rho)
        print("Actual rho =", actual_rho)
        print("Density error =", actual_rho - rho)
        print("Number of particles =", len(positions))

    # ============================================================
    # SAVE FILE
    # ============================================================
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(filepath), mode="w") as f:
        f.append(frame)

    if end_print:
        print(f"Saved lattice to {filepath}")

    if return_path:
        return frame, filepath
    
    return frame