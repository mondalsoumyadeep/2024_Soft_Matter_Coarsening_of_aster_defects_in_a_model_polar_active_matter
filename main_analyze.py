# Import necessary libraries
from cmath import pi  
from re import A  
import numpy as np
from matplotlib import pyplot as plt
import argparse
import matplotlib.colors

######################### Parameters #########################
# Grid and simulation parameters
N = 64             # Grid size (NxN)
dt = 1e-3          # Time step size
T = 10000001       # Total number of steps in the simulation
tstart = 0         # Starting time for analysis
skip = 100000      # Number of steps between saved frames
dx = 1e0           # Grid spacing
dpi = 75           # DPI for saved figures
prefix1 = 'actin'  # Prefix for saved images

# Initial analysis frame setup
startStep = int(tstart / skip)
start_frame = startStep

######################### Analysis Setup ######################
# Calculate the number of analysis steps
nSteps = int(T / skip)

# Set up grid for plotting
x = dx * np.linspace(0, N - 1, N)
y = dx * np.linspace(0, N - 1, N)
X, Y = np.meshgrid(x, y)
xs = X.flatten()  # Flattened X-coordinates for vector plotting
ys = Y.flatten()  # Flattened Y-coordinates for vector plotting
id = np.arange(0, x.size, 1)  # Array of indices (can be used for analysis if needed)

##################### Analysis Loop #####################
# Loop over each saved time step to generate plots
for i in range(startStep, nSteps + 1, 1):
    
    # Load data files for concentration and vector fields at current step
    cfile = f"data/c_{skip * i}.txt"
    nxfile = f"data/nx_{skip * i}.txt"
    nyfile = f"data/ny_{skip * i}.txt"
    C = np.loadtxt(cfile)  # Concentration field
    nx = np.loadtxt(nxfile)  # X-component of polar vector field
    ny = np.loadtxt(nyfile)  # Y-component of polar vector field
    
    ################### Plotting ###################
    
    # Plot concentration field
    fig1, ax1 = plt.subplots(figsize=(16.5, 15))
    im = ax1.pcolormesh(x, y, C.T, vmin=0, vmax=20, cmap="Reds", shading='nearest')
    cbar = plt.colorbar(im, ax=ax1)
    
    # Remove axis ticks and labels for a cleaner plot
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Overlay quiver plot for vector field (nx, ny) with reduced opacity
    ax1.quiver(x, y, nx.T, ny.T, alpha=0.6)
    
    # Add time title as a text annotation
    title = "Time=%0.2e" % (skip * i * dt)
    ax1.text(0.98, 0.0, title, ha='right', va='bottom', transform=ax1.transAxes,
             color='black', fontsize=25, weight="bold")
    
    # Save figure with specified naming convention
    figname1 = ("images/%s_%04d.png" % (prefix1, i))
    fig1.savefig(figname1, bbox_inches="tight", dpi=dpi)
    plt.close(fig1)  # Close figure to free memory


