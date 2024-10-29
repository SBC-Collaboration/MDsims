import numpy as np
import hoomd
import gsd.hoomd


def FillBoxCubicLattice(xlim=np.float64([-5, 5]), # x-limits of box to fill
                        ylim=np.float64([-5, 5]), # y-limits of box to fill
                        zlim=np.float64([-5, 5]), # z-limits of box to fill
                        rho=np.float64(1), # density = 1 / dimension of cubic cell
                        ):
    ''' outputs an ndarray with shape (N,3) giving the x,y,z positions of
        N particles arranged in a cubic lattice.'''

    cellsize = 1/rho

    nx = np.intp(np.floor(np.diff(xlim)/cellsize))
    ny = np.intp(np.floor(np.diff(ylim)/cellsize))
    nz = np.intp(np.floor(np.diff(zlim)/cellsize))

    xoffset = xlim[0] + 0.5*(np.diff(xlim) - cellsize*nx)
    yoffset = ylim[0] + 0.5*(np.diff(ylim) - cellsize*ny)
    zoffset = zlim[0] + 0.5*(np.diff(zlim) - cellsize*nz)

    numparticles = nx*ny*nz
    xvec = cellsize * np.arange(nx,dtype=np.float64) + xoffset
    yvec = cellsize * np.arange(ny,dtype=np.float64) + yoffset
    zvec = cellsize * np.arange(nz,dtype=np.float64) + zoffset

    positions = np.zeros((numparticles, 3), dtype=np.float64)
    positions_reshaped = positions.reshape((nx,ny,nz,3))
    positions_reshaped[:,:,:,0] = xvec[:,None,None]
    positions_reshaped[:,:,:,1] = yvec[None,:,None]
    positions_reshaped[:,:,:,2] = zvec[None,None,:]

    return positions


def FillBoxFccLattice(xlim=np.float64([-5, 5]), # x-limits of box to fill
                      ylim=np.float64([-5, 5]), # y-limits of box to fill
                      zlim=np.float64([-5, 5]), # z-limits of box to fill
                      rho=np.float64(1), # density = 1 / dimension of cubic cell
                      ):
    ''' outputs an ndarray with shape (N,3) giving the x,y,z positions of
        N particles arranged in a cubic lattice.'''

    cellsize = 2 / rho # start with a half-size cubic lattice, then apply cut

    nx = np.intp(np.floor(np.diff(xlim)/cellsize))
    ny = np.intp(np.floor(np.diff(ylim)/cellsize))
    nz = np.intp(np.floor(np.diff(zlim)/cellsize))

    xoffset = xlim[0] + 0.5*(np.diff(xlim) - cellsize*nx)
    yoffset = ylim[0] + 0.5*(np.diff(ylim) - cellsize*ny)
    zoffset = zlim[0] + 0.5*(np.diff(zlim) - cellsize*nz)

    numparticles = nx*ny*nz
    xvec_i = np.arange(nx)
    yvec_i = np.arange(ny)
    zvec_i = np.arange(nz)

    pos_i = np.zeros((numparticles,3),dtype=np.intp)
    pos_i_reshaped = pos_i.reshape((nx,ny,nz,3))
    pos_i_reshaped[:,:,:,0] = xvec_i[:,None,None]
    pos_i_reshaped[:,:,:,1] = yvec_i[None,:,None]
    pos_i_reshaped[:,:,:,2] = zvec_i[None,None,:]

    fcc_cut = np.mod(np.sum(pos_i, axis=1),2) == 0

    positions = np.float64(pos_i[fcc_cut,:]) * cellsize + np.float64([xoffset, yoffset, zoffset])
    return positions


def FillBoxRandom(xlim=np.float64([-5, 5]), # x-limits of box to fill
                  ylim=np.float64([-5, 5]), # y-limits of box to fill
                  zlim=np.float64([-5, 5]), # z-limits of box to fill
                  rho=np.float64(1), # density = 1 / dimension of cubic cell
                  ):
    ''' outputs an ndarray with shape (N,3) giving the x,y,z positions of
        N particles randomly filling the box, with average density rho '''
    Lx = np.diff(xlim)
    Ly = np.diff(ylim)
    Lz = np.diff(zlim)
    numparticles = np.int(np.floor(Lx*Ly*Lz*rho))
    positions = np.random.rand(numparticles,3) * np.float64([Lx,Ly,Lz]) + np.float64([xlim[0],ylim[0],zlim[0]])
    return positions


def AddNoise2Positions(positions, # (N,3) np.float64 ndarray
                       noise_sigma, # Gaussian noise term, radial sigma
                       noise_cap # noise cutoff (max radial shift)
                       ):
    ''' randomly shifts the input position array by independently adding 
        an isotropic 3D Gaussian with cutoff to each position '''

    numparticles = positions.shape[0]
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


def SelectLJModel(rcut=3.0,
                  forceshift=False,
                  tail_correction=False,
                  mode="none",
                  r_on=2.9
                  ):
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


def RunSurfaceTensionCalc(filename, logfile, L=50.0, kT=0.0, nsteps=1000):
# filename (str): The HDF5 file to log data to.
# logfile (str): The file to log simulation properties.
# L (float): The size of the simulation box.
# kT (float): The thermal energy of the particles.
# nsteps (int): Number of simulation steps to run.
   
    # Setup initial positions in a cubic lattice
    boxlimit = np.float64([-0.5 * L, 0.5 * L])
    positions = FillBoxCubicLattice(xlim=boxlimit, ylim=boxlimit, zlim=boxlimit, rho=0.125)
    N_particles = positions.shape[0]

    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=1)
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = positions
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['A']
    
    simulation.create_state_from_snapshot(frame)
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    # Define interactions and integrate
    lj = SelectLJModel(rcut=3.0)
    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),
                                          thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT))
    integrator.methods.append(nvt)
    simulation.operations.integrator = integrator

    logger = hoomd.logging.Logger(categories=['scalar'])
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermodynamic_properties)
    
    logger.add(simulation, quantities=['timestep'])
    logger.add(thermodynamic_properties, quantities=['pressure', 'volume', 'num_particles'])
    
    file = open(logfile, mode='w', newline='\n')
    table_file = hoomd.write.Table(output=file,
                                   trigger=hoomd.trigger.Periodic(period=50),
                                   logger=logger
                                  )
    simulation.operations.writers.append(table_file)

    surface_tension_log = []
    for step in range(nsteps):
        simulation.run(1)
        
        with h5py.File(filename, 'r') as hdf5_file:
            pressure_tens = np.float64(hdf5_file['hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor'][:])
          
        # do calculation on sim
        P_xx = pressure_tens[:, 0]
        P_yy = pressure_tens[:, 4]
        surface_tension = (P_yy - P_xx) / 2
        surface_tension_log.append([step, surface_tension])

    simulation.operations.writers.remove(table_file)
    file.close()


def CalcVaporPresure(logfile):
    ''' Opens logfile created by RunVaporPressureCalc, and analyzes it '''
    with open(logfile, mode='r') as file:
        pass

    vpdict = dict(vapor_pressure=np.float64(0), vapor_pressure_rms=np.float64(0), vapor_pressure_slope=np.float64(0))
    return vpdict



def RunSurfaceTensionCalc(filename, logfile, L=50.0, kT=0.0, nsteps=1000):
  
        # filename (str): The HDF5 file to log data to.
        # logfile (str): The file to log simulation properties.
        # L (float): The size of the simulation box.
        # kT (float): The thermal energy of the particles.
        # nsteps (int): Number of simulation steps to run.
   
    # cubic lattice
    boxlimit = np.float64([-0.5 * L, 0.5 * L])
    positions = FillBoxCubicLattice(xlim=boxlimit, ylim=boxlimit, zlim=boxlimit, rho=0.125)
    N_particles = positions.shape[0]

    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=1)
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = positions
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['A']
    
    simulation.create_state_from_snapshot(frame)
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    lj = SelectLJModel(rcut=3.0)
    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),
                                          thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT))
    integrator.methods.append(nvt)
    simulation.operations.integrator = integrator

    logger = hoomd.logging.Logger(categories=['scalar'])
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermodynamic_properties)
    
    logger.add(simulation, quantities=['timestep'])
    logger.add(thermodynamic_properties, quantities=['pressure', 'volume', 'num_particles'])
    
    file = open(logfile, mode='w', newline='\n')
    table_file = hoomd.write.Table(output=file,
                                   trigger=hoomd.trigger.Periodic(period=50),
                                   logger=logger
                                  )
    simulation.operations.writers.append(table_file)

    surface_tension_log = []
    for step in range(nsteps):
        simulation.run(1)
        
        with h5py.File(filename, 'r') as hdf5_file:
            pressure_tens = np.float64(hdf5_file['hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor'][:])
          
        # do calculation on sim
        P_xx = pressure_tens[:, 0]
        P_yy = pressure_tens[:, 4]
        surface_tension = (P_yy - P_xx) / 2
        surface_tension_log.append([step, surface_tension])

    simulation.operations.writers.remove(table_file)
    file.close()

def RunHeatOfVaporizationCalc():
    return True

