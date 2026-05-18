import numpy as np
import gsd.hoomd


def FillBoxCubicLattice(xlim, ylim, zlim, rho):
    cellsize = rho**(-1.0 / 3.0)

    nx = int(np.floor((xlim[1] - xlim[0]) / cellsize))
    ny = int(np.floor((ylim[1] - ylim[0]) / cellsize))
    nz = int(np.floor((zlim[1] - zlim[0]) / cellsize))

    xoffset = xlim[0] + 0.5 * ((xlim[1] - xlim[0]) - cellsize * nx)
    yoffset = ylim[0] + 0.5 * ((ylim[1] - ylim[0]) - cellsize * ny)
    zoffset = zlim[0] + 0.5 * ((zlim[1] - zlim[0]) - cellsize * nz)

    xvec = cellsize * np.arange(nx, dtype=np.float64) + xoffset
    yvec = cellsize * np.arange(ny, dtype=np.float64) + yoffset
    zvec = cellsize * np.arange(nz, dtype=np.float64) + zoffset

    positions = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    pos = positions.reshape((nx, ny, nz, 3))

    pos[:, :, :, 0] = xvec[:, None, None]
    pos[:, :, :, 1] = yvec[None, :, None]
    pos[:, :, :, 2] = zvec[None, None, :]

    return positions


def FillBoxFccLattice(xlim, ylim, zlim, rho):
    cellsize = (2.0 * rho) ** (-1.0 / 3.0)

    nx = int(np.floor((xlim[1] - xlim[0]) / cellsize))
    ny = int(np.floor((ylim[1] - ylim[0]) / cellsize))
    nz = int(np.floor((zlim[1] - zlim[0]) / cellsize))

    xoffset = xlim[0] + 0.5 * ((xlim[1] - xlim[0]) - cellsize * nx)
    yoffset = ylim[0] + 0.5 * ((ylim[1] - ylim[0]) - cellsize * ny)
    zoffset = zlim[0] + 0.5 * ((zlim[1] - zlim[0]) - cellsize * nz)

    numparticles = nx * ny * nz

    xvec_i = np.arange(nx)
    yvec_i = np.arange(ny)
    zvec_i = np.arange(nz)

    pos_i = np.zeros((numparticles, 3), dtype=np.intp)
    pos_i_reshaped = pos_i.reshape((nx, ny, nz, 3))

    pos_i_reshaped[:, :, :, 0] = xvec_i[:, None, None]
    pos_i_reshaped[:, :, :, 1] = yvec_i[None, :, None]
    pos_i_reshaped[:, :, :, 2] = zvec_i[None, None, :]

    fcc_cut = np.mod(np.sum(pos_i, axis=1), 2) == 0

    positions = (
        np.float64(pos_i[fcc_cut, :]) * cellsize
        + np.float64([xoffset, yoffset, zoffset])
    )

    return positions


def make_lattice_frame(BoxLength, rho, lattice_type="cubic", particle_type="A", verbose=True):
    boxlimit = [-BoxLength / 2, BoxLength / 2]

    if lattice_type == "cubic":
        positions = FillBoxCubicLattice(
            xlim=boxlimit,
            ylim=boxlimit,
            zlim=boxlimit,
            rho=rho,
        )
        cellsize = rho**(-1.0 / 3.0)

    elif lattice_type == "fcc":
        positions = FillBoxFccLattice(
            xlim=boxlimit,
            ylim=boxlimit,
            zlim=boxlimit,
            rho=rho,
        )
        cellsize = (2.0 * rho) ** (-1.0 / 3.0)

    else:
        raise ValueError("lattice_type must be 'cubic' or 'fcc'")

    frame = gsd.hoomd.Frame()
    frame.particles.N = len(positions)
    frame.particles.types = [particle_type]
    frame.particles.position = positions
    frame.particles.typeid = [0] * len(positions)
    frame.configuration.box = [BoxLength, BoxLength, BoxLength, 0, 0, 0]

    actual_rho = len(positions) / BoxLength**3

    if verbose:
        print(f"{lattice_type} lattice")
        print("----------------------------")
        print("Target BoxLength =", BoxLength)
        print("Target rho =", rho)
        print("Derived cell spacing =", cellsize)
        print("Number of particles =", len(positions))
        print("Actual rho =", actual_rho)
        print("Density error =", actual_rho - rho)

    return frame