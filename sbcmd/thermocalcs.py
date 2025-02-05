import numpy as np
import hoomd
import gsd.hoomd
import pandas as pd
import h5py
import matplotlib.pyplot as plt


def FillBoxCubicLattice(xlim=np.float64([-5, 5]), # x-limits of box to fill
                        ylim=np.float64([-5, 5]), # y-limits of box to fill
                        zlim=np.float64([-5, 5]), # z-limits of box to fill
                        rho=np.float64(1), # density = 1 / dimension of cubic cell
                        ):
    ''' outputs an ndarray with shape (N,3) giving the x,y,z positions of
        N particles arranged in a cubic lattice.'''

    cellsize = np.float_power(rho, -1.0/3.0)

    nx = np.intp(np.floor(np.diff(xlim)/cellsize))[0]
    ny = np.intp(np.floor(np.diff(ylim)/cellsize))[0]
    nz = np.intp(np.floor(np.diff(zlim)/cellsize))[0]

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

    cellsize = np.float_power(2.0*rho, -1.0/3.0) # start with a half-size cubic lattice, then apply cut

    nx = np.intp(np.floor(np.diff(xlim)/cellsize))[0]
    ny = np.intp(np.floor(np.diff(ylim)/cellsize))[0]
    nz = np.intp(np.floor(np.diff(zlim)/cellsize))[0]

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
    Lx = np.diff(xlim)[0]
    Ly = np.diff(ylim)[0]
    Lz = np.diff(zlim)[0]
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

def RunPressureTime(logfile, lj, L, kT, rho, nsteps):
    '''Plot pressure versus time'''

    boxlimit = np.float64([-0.5*L, 0.5*L])
    positions = FillBoxCubicLattice(xlim=boxlimit, ylim=boxlimit, zlim=boxlimit, rho=rho) # start with unstable density
    N_particles = positions.shape[0]


    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = positions
    frame.particles.typeid = [0] * N_particles
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['A']

    cpu=hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=1)
    simulation.create_state_from_snapshot(frame)
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),
                                          thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT)
                                          )
    integrator.methods.append(nvt)
    simulation.operations.integrator = integrator

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermodynamic_properties)
    simulation.run(0)

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(simulation, quantities=['timestep'])
    logger.add(thermodynamic_properties, quantities=['pressure'])
    file = open(logfile, mode='w', newline='\n')
    table_file = hoomd.write.Table(output=file,
                                  trigger=hoomd.trigger.Periodic(period=50),
                                  logger=logger
                                  )
    simulation.operations.writers.append(table_file)

    simulation.run(nsteps)

    simulation.operations.writers.remove(table_file)

    return True

def PlotPressureTime(logfile):
    data = pd.read_csv(logfile,sep='\s+',header=None)
    data = pd.DataFrame(data)

    x = data[0]
    y = data[1]
    plt.plot(x, y,'r--')
    plt.show()

    return True


def RunVaporPressureCalc(logfile, lj, L=10., kT=1.0, nsteps=1e5):
    ''' Creates and runs a hoomd md simulation object, writing a logfile'''

    # get initial positions
    boxlimit = np.float64([-0.5*L, 0.5*L])
    positions = FillBoxCubicLattice(xlim=boxlimit, ylim=boxlimit, zlim=boxlimit, rho=0.125) # start with unstable density
    N_particles = positions.shape[0]

    # create initial frame
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = positions
    frame.particles.typeid = [0] * N_particles
    frame.configuration.box = [L, L, L, 0, 0, 0]
    frame.particles.types = ['A']


    # create simulation object and initialize state
    gpu=hoomd.device.GPU()
    simulation = hoomd.Simulation(device=gpu, seed=1)
    simulation.create_state_from_snapshot(frame)
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    # add integrator to simulation object
    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),
                                          thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT)
                                          )
    integrator.methods.append(nvt)
    simulation.operations.integrator = integrator

    # add thermo computation to simulation object
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermodynamic_properties)
    simulation.run(0)

    # add logfile
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(simulation, quantities=['timestep'])
    logger.add(thermodynamic_properties, quantities=['pressure'])
    logger.add(thermodynamic_properties, quantities=['volume'])
    logger.add(thermodynamic_properties, quantities=['num_particles'])
    file = open(logfile, mode='w', newline='\n')
    table_file = hoomd.write.Table(output=file,
                                  trigger=hoomd.trigger.Periodic(period=50),
                                  logger=logger
                                  )
    simulation.operations.writers.append(table_file)

    # now run the simulation
    simulation.run(nsteps)

    simulation.operations.writers.remove(table_file)

    return True



def CalcVaporPresure(logfile):
    ''' Opens logfile created by RunVaporPressureCalc, and analyzes it '''
    content = []

    with open(logfile, mode='r') as file:
        for line in file:
          data = line.strip()
          content.append(data)

    content.pop(0)
    timesteps = len(content)

    start = int((timesteps / 50) - 100)

    content = content[start:]

    sum = 0

    for d in content:
        sum = sum + float(d)

    vp = sum / 100
    return vp

def RunSurfaceTensionCalc(nsteps=1000):
    # plan: run this code and compare in *excel*
    # import thermocalcs.py
    # thermocalcs.RunSurfaceTensionCalc()
    #python -c "import h5py, pandas as pd; f = h5py.File('surface_tension_data.h5', 'r'); pd.DataFrame(f['surface_tension'][:], columns=['Surface_Tension']).to_csv('surface_tension_data.csv', index=False)"
    ''' convert to csv ^'''
    filename = "surface_tension_data.h5"  # auto make output HDF5
    L = 50.0
    kT = 0.0
    
    # Cubic lattice setup
    boxlimit = np.float64([-0.5 * L, 0.5 * L])
    positions = FillBoxCubicLattice(xlim=boxlimit, ylim=boxlimit, zlim=boxlimit, rho=0.9) 
    # should rho be dynamic?
    N_particles = positions.shape[0]

    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=1)
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = positions
    frame.configuration.box = [L, L, 2 * L, 0, 0, 0] # make rectangular
    frame.particles.types = ['A']
    
    simulation.create_state_from_snapshot(frame)
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

    lj = SelectLJModel(rcut=3.0) #nist uses 2.5
    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),
                                          thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT))
    integrator.methods.append(nvt)
    simulation.operations.integrator = integrator

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermodynamic_properties)
    
    # logger.add(simulation, quantities=['timestep', 'sequence'])
    # logger.add(thermodynamic_properties, quantities=['pressure', 'volume', 'num_particles', 'pressure_tensor'])
    
    simulation.run(nsteps)

        # Access pressure tensor directly from the compute object after the simulation
    pressure_tens = thermodynamic_properties.pressure_tensor[:]
    P_xx = pressure_tens[:, 0]
    P_yy = pressure_tens[:, 4]
    surface_tension = (P_yy - P_xx) / 2


    with h5py.File(filename, 'w') as hdf5_file:
        hdf5_file.create_dataset('surface_tension', data=surface_tension)
    
    return {"final_surface_tension": surface_tension}


def RunHeatOfVaporizationCalc():
    return True

def colormap_histogram(gsd_file, in_bins = 50):
    # silly question, should we force a gsd file name convention
    # perhaps have z num as input
    # assumming continous file -> no need for my method for connecting timesteps(?)
    with gsd.hoomd.open(file_name, 'r') as f:
        num_frames = len(f)
        last_frame = f[num_frames -1]

    # filter particles with |z| < 3 and get x, y positions
    positions = last_frame.particles.position[np.abs(last_frame.particles.position[:, 2]) < 3]
    positions = np.array(positions)
    
    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist2d(positions[:, 0], positions[:, 1], bins=in_bins, cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Histogram of Particle Positions (Last Timestep Frame) z < 3')
    plt.show()
    

