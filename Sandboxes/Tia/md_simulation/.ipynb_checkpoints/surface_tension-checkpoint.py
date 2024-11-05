import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

def calculate_surface_tension(filename, radius, tempSpike):
    with h5py.File(filename, 'r') as hdf5_file:
        pressure_tens = np.float64(hdf5_file['hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor'][:])

        # [pxx,pxy,pxz,pyy,pyz,pzz]
        P_xx = pressure_tens[:, 0]  # pressure in x-direction
        P_yy = pressure_tens[:, 4]  # pressure in y-direction
        surface_tension = (P_yy - P_xx) / 2
        
        timesteps = np.arange(len(surface_tension))
        
        # output surface tension data to a file
        output_filename = f"surface_tension_r{radius}_kt{tempSpike}.txt"
        np.savetxt(output_filename, np.column_stack((timesteps, surface_tension)), 
                   header="Timestep Surface_Tension", fmt="%d %.6f")
        
        print(f"File saved as: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate surface tension from HDF5 file.")
    parser.add_argument('filename', type=str, help='The HDF5 file to process')
    parser.add_argument('radius', type=int, help='radius value')
    parser.add_argument('tempSpike', type=int, help='tempSpike (kt) value')

    args = parser.parse_args()

    calculate_surface_tension(args.filename, args.radius, args.tempSpike)

# cd MDsims/Sandboxes/Tia
# python surface_tension.py longerRun8energy0.9r2.h5 2 8

