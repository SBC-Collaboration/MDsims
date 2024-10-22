import numpy as np
import hoomd
import gsd.hoomd


def FillBoxCubicLattice(
                        xlim=np.float64([-5, 5]), # number of cells (1 - # of particles) in x direction
                        ylim=np.float64([-5, 5]), # number of cells in y direction
                        zlim=np.float64([-5, 5]), # number of cells in z direction
                        cellsize=np.float64(1), # dimension of cubic cell
                        noise_sigma = np.float64(0), # Gaussian noise term, radial sigma
                        noise_cap = np.float64(0), # noise cutoff
                        ):
    ''' outputs an ndarray with shape (N,3) giving the x,y,z positions of
        N particles arranged in a cubic lattice.'''

    nx = np.intp(np.floor(np.diff(xlim)/cellsize))
    ny = np.intp(np.floor(np.diff(xlim)/cellsize))
    nz = np.intp(np.floor(np.diff(xlim)/cellsize))

    xoffset = xlim[0] + 0.5*(np.diff(xlim) - cellsize*nx)
    yoffset = ylim[0] + 0.5*(np.diff(ylim) - cellsize*ny)
    zoffset = zlim[0] + 0.5*(np.diff(zlim) - cellsize*nz)

    numparticles = nx*ny*nz
    xvec = cellsize * np.float64(range(nx)) + xoffset
    yvec = cellsize * np.float64(range(ny)) + yoffset
    zvec = cellsize * np.float64(range(nz)) + zoffset

    positions = np.zeros((numparticles, 3), dtype=np.float64)
    positions_reshaped = positions.reshape((nx,ny,nz,3))
    positions_reshaped[:,:,:,0] = xvec[:,None,None]
    positions_reshaped[:,:,:,1] = yvec[None,:,None]
    positions_reshaped[:,:,:,2] = zvec[None,None,:]

    if noise_sigma > 0:
        noise_r = np.random.randn(numparticles) * noise_sigma
        noise_r[noise_r>noise_cap] = noise_cap
        noise_r[noise_r<-noise_cap] = -noise_cap
        noise_phi = np.random.rand(numparticles) * 2 * np.pi
        noise_costheta = np.random.rand(numparticles) * 2 - 1
        noise_x = noise_r * np.cos(noise_phi) * np.sqrt(1 - noise_costheta*noise_costheta)
        noise_y = noise_r * np.sin(noise_phi) * np.sqrt(1 - noise_costheta*noise_costheta)
        noise_z = noise_r * noise_costheta
        positions[:,0] = positions[:,0] + noise_x
        positions[:,1] = positions[:,1] + noise_y
        positions[:,2] = positions[:,2] + noise_z

    return positions


def BuildFccLatice():
    pass


def FillBoxRandom(
                  xlim=np.float64([-5, 5]), # number of cells (1 - # of particles) in x direction
                  ylim=np.float64([-5, 5]), # number of cells in y direction
                  zlim=np.float64([-5, 5]), # number of cells in z direction
                  rho=np.float64(1) # average particle density
                  ):
    ''' outputs an ndarray with shape (N,3) giving the x,y,z positions of
        N particles randomly filling the box, with average density rho '''
    Lx = np.diff(xlim)
    Ly = np.diff(ylim)
    Lz = np.diff(zlim)
    numparticles = np.int(np.floor(Lx*Ly*Lz*rho))
    positions = np.random.rand(numparticles,3) * np.float64([Lx,Ly,Lz]) + np.float64([xlim[0],ylim[0],zlim[0]])
    return positions



def SelectLJModel(rcut=3.0, forceshift=False, tail_correction=False, mode="none", r_on=2.9):
    nl = hoomd.md.nlist.Cell(buffer=0.4)
    if forceshift:
        lj = hoomd.md.pair.ForceShiftedLJ(nlist=nl, default_r_cut=rcut)
    else:
        if mode == "xplor":
            lj = hoomd.md.pair.LJ(nlist=nl, default_r_cut=rcut,
                                  default_r_on = r_on,
                                  mode=mode, tail_correction=tail_correction)
        else:
            lj = hoomd.md.pair.LJ(nlist=nl, default_r_cut=rcut,
                                  mode=mode, tail_correction=tail_correction)

    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    return lj


def RunVaporPressureCalc(logfile, lj, n=40, kT=1.0, nsteps=1e5):
    ''' Creates and runs a hoomd md simulation object, writing a logfile'''
    return True


def CalcVaporPresure(logfile):
    ''' Opens logfile created by RunVaporPressureCalc, and analyzes it '''
    vpdict = dict(vapor_pressure=np.float64(0), vapor_pressure_rms=np.float64(0), vapor_pressure_slope=np.float64(0))
    return vpdict


def RunSurfaceTensionCalc():
    return True


def RunHeatOfVaporizationCalc():
    return True

