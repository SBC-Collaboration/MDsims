import numpy as np
import hoomd
import gsd.hoomd

def BuildCubicLattice(
                      cellsize=np.float64(1), # dimension of cubic cell
                      nx=np.intp(10), # number of cells (1 - # of particles) in x direction
                      ny=np.intp(-1), # number of cells in y direction
                      nz=np.intp(-1), # number of cells in z direction
                      xoffset=np.float64(0), # x coord of center of lattice
                      yoffset=np.float64(0), # y coord of center of lattice
                      zoffset=np.float64(0), # z coord of center of lattice
                      noise_sigma = np.float64(0), # Gaussian noise term, radial sigma
                      noise_cap = np.float64(0), # noise cutoff
                      ):
    ''' outputs an ndarray with shape (N,3) giving the x,y,z positions of
        N particles arranged in a cubic lattice.'''

    if ny<0:
        ny = nx
    if nz<0:
        nz = nx

    numparticles = (nx+1)*(ny+1)*(nz+1)
    xvec = cellsize * (np.linspace(0,nx,nx+1,dtype=np.float64) - nx*0.5)
    yvec = cellsize * (np.linspace(0,ny,ny+1,dtype=np.float64) - ny*0.5)
    zvec = cellsize * (np.linspace(0,nz,nz+1,dtype=np.float64) - nz*0.5)

    positions = np.zeros((numparticles, 3), dtype=np.float64)
    positions_reshaped = positions.reshape((nx+1,ny+1,nz+1,3))
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

def RunVaporPressureCalc(lj, n=40, kT=1.0):
    pass

def RunSurfaceTensionCalc():
    pass

def RunHeatOfVaporizationCalc():
    pass

