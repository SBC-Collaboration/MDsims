import numpy as np
import hoomd
import gsd.hoomd
import pandas as pd
import h5py
import matplotlib.pyplot as plt

def colormap_histogram(gsd_file, in_bins = 50, z = 3, x1_name="X Position", z2_name="Y Position"):
    with gsd.hoomd.open(file_name, 'r') as f:
        num_frames = len(f)
        last_frame = f[num_frames -1]

    # filter particles with |z| < 3 and get x, y positions
    positions = last_frame.particles.position[np.abs(last_frame.particles.position[:, 2]) < z]
    positions = np.array(positions)
    
    title = f"2D Histogram of Particle Positions (Last Timestep Frame) z < {z}"

    # histogram
    plt.figure(figsize=(8, 6))
    plt.hist2d(positions[:, 0], positions[:, 1], bins=in_bins, cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.show()

def line_plot(x1_data = np.arange(0, 100, 1), y1_data = np.arange(0, 100, 1), x_name = "Default_x", y_name = "Default_y"):
    plt.plot(x1_data, y1_data)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()
