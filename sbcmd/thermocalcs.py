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
                      zoffset=np.float64(0) # z coord of center of lattice
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

    return positions

def BuildFccLatice():
    pass

def CalculateVaporPressure():
    pass

def CalculateSurfaceTension():
    pass

def CalculateHeatOfVaporization():
    pass

