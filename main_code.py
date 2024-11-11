# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numba import njit  # For Just-In-Time compilation to speed up specific functions

# Simulation parameters
N = 64             # Grid size (NxN)
Df = 2.5             # Diffusion coefficient for concentration field
k1 = 2.5             # Elastic constant (term 1)
k2 = 0.0             # Elastic constant (term 2)
z = 10               # Active parameter, Contractility
Ta = 0               # Active temperature, Noise strength
seed = 10            # Random seed 
C = 1                # Initial concentration level
Csig = 0.001         # Fluctuation level for concentration
l = 0                # Self-propulsion term (lambda)
v0 = 1               # Advection speed 
a = 100              # phenomenological parameter 
b = 100              # beta
dx = 1               # Spatial step size along x
dy = 1               # Spatial step size along y 
dt = 1e-3            # Time step
T = 10000001           # Total time steps for simulation
tstart = 0           # Starting time step
skip = 100000        # Output frequency
disorderedIC = False # Initial condition (True for disordered, False for ordered)
var = 0.1            # Variance in initial noise

#########################################
# Initialize simulation
#########################################

np.random.seed(seed)

# Core simulation function to evolve fields over time
@njit
def evolve(nx, ny, c, v0):
    """Computes the time evolution of fields nx, ny, and concentration c."""
    dnx = np.zeros((N, N))
    dny = np.zeros((N, N))
    dc = np.zeros((N, N))

    # Loop over all grid points
    for i in range(N):
        for j in range(N):
            # Define neighboring indices with periodic boundary conditions
            In = (i + 1) % N
            Ip = (i - 1) % N
            Jn = (j + 1) % N
            Jp = (j - 1) % N
            
            # Calculate terms for nx evolution (Implementing Central in space method)
            t1x = (-l / (2 * dx)) * (nx[i, j] * (nx[In, j] - nx[Ip, j]) + ny[i, j] * (nx[i, Jn] - nx[i, Jp]))
            t2x = (k1 / (dx * dx)) * (nx[In, j] + nx[Ip, j] + nx[i, Jp] + nx[i, Jn] - 4 * nx[i, j])
            t3x = (k2 / (dx * dx)) * (nx[In, j] - 2 * nx[i, j] + nx[Ip, j]) + \
                  (k2 / (4 * dx * dx)) * (ny[In, Jn] - ny[In, Jp] - ny[Ip, Jn] + ny[Ip, Jp])
            t4x = (z / dx) * (c[In, j] - c[Ip, j]) * 0.5
            t5x = a * nx[i, j] - b * (nx[i, j]**2 + ny[i, j]**2) * nx[i, j]
            t6x = np.sqrt(Ta * dt / (dx * dx)) * np.random.randn()

            # Calculate terms for ny evolution (Implementing Central in Space method)
            t1y = (-l / (2 * dx)) * (nx[i, j] * (ny[In, j] - ny[Ip, j]) + ny[i, j] * (ny[i, Jn] - ny[i, Jp]))
            t2y = (k1 / (dx * dx)) * (ny[In, j] + ny[Ip, j] + ny[i, Jp] + ny[i, Jn] - 4 * ny[i, j])
            t3y = (k2 / (dx * dx)) * (ny[i, Jn] - 2 * ny[i, j] + ny[i, Jp]) + \
                  (k2 / (4 * dx * dx)) * (nx[In, Jn] - nx[In, Jp] - nx[Ip, Jn] + nx[Ip, Jp])
            t4y = (z / dx) * (c[i, Jn] - c[i, Jp]) * 0.5
            t5y = a * ny[i, j] - b * (nx[i, j]**2 + ny[i, j]**2) * ny[i, j]
            t6y = np.sqrt(Ta * dt / (dx * dx)) * np.random.randn()  # Noise term

            # Calculate terms for concentration c evolution
            t1c = (-v0 / dx) * (c[In, j] * nx[In, j] - c[Ip, j] * nx[Ip, j] + c[i, Jn] * ny[i, Jn] - c[i, Jp] * ny[i, Jp]) * 0.5
            t2c = (Df / (dx * dx)) * (c[In, j] + c[Ip, j] + c[i, Jn] + c[i, Jp] - 4 * c[i, j])

            # Update increments
            dnx[i, j] = dt * (t1x + t2x + t3x + t4x + t5x) + t6x
            dny[i, j] = dt * (t1y + t2y + t3y + t4y + t5y) + t6y
            dc[i, j] = dt * (t1c + t2c)

    return dnx, dny, dc

# Set up initial conditions
if disorderedIC:
    # Disordered initial condition
    theta = np.random.uniform(0, 2 * np.pi, (N, N))
    nx = np.cos(theta)
    ny = np.sin(theta)
else:
    # Ordered initial condition
    theta = np.zeros((N, N))
    nx = np.cos(theta)
    ny = np.sin(theta)

# Initialize concentration field
c = C * (np.ones((N, N)) + Csig * np.random.rand(N, N))

#########################################
# Main simulation loop
#########################################
for t in range(tstart, T):
    # Check for NaNs in concentration field to avoid numerical instability
    if np.isnan(np.sum(c)):
        break

    # Time loop for evolution
    dnt = evolve(nx, ny, c, v0) 
    nx += dnt[0]  # Update nx
    ny += dnt[1]  # Update ny
    c += dnt[2]   # Update c 
    
    # Output results periodically
    if t % skip == 0:
        nxFile = f"data/nx_{t}.txt"  # Save in 'data' folder
        nyFile = f"data/ny_{t}.txt"  # Save in 'data' folder
        cFile = f"data/c_{t}.txt"    # Save in 'data' folder
        np.savetxt(nxFile, nx)  # Save nx field
        np.savetxt(nyFile, ny)  # Save ny field
        np.savetxt(cFile, c)    # Save concentration field
