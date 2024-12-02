import numpy as np
from scipy.spatial.distance import cdist
import os

# Function to get the next index with periodic boundary
def get_next(i, N):
    return 0 if i == N - 1 else i + 1

# Function to calculate the area (phase difference) between two angles, adjusted for periodicity
def area(tn, t0):
    dt = tn - t0
    if dt > np.pi:
        dt -= 2 * np.pi  # Adjust to [-pi, pi]
    if dt < -np.pi:
        dt += 2 * np.pi  # Adjust to [-pi, pi]
    return dt

# Function to find topological defects 
def find_defects(phi, phi0=1.6 * np.pi, mx=0, my=0):
    sx, sy = np.shape(phi)  # Get the shape of the field
    qx, qy, qi, q_direction = [], [], [], []  # Lists to store defect data
    
    # Loop through the field to calculate the defects
    for j in range(sy - my):
        jnext = get_next(j, sy)
        for i in range(sx - mx):
            inext = get_next(i, sx)
            t1, t2, t3, t4 = phi[i, j], phi[inext, j], phi[inext, jnext], phi[i, jnext]  # Get the 4 surrounding points
            
            # Calculate the phase difference for the edges of the cell
            dphi = area(t2, t1) + area(t3, t2) + area(t4, t3) + area(t1, t4)
            flux1 = -nx[i, j] - ny[i, j]
            flux2 = nx[inext, j] - ny[inext, j]
            flux3 = nx[inext, jnext] + ny[inext, jnext]
            flux4 = -nx[i, jnext] + ny[i, jnext]
            total_flux = flux1 + flux2 + flux3 + flux4
            # If the absolute phase difference is greater than the threshold, it's a defect
            if np.abs(dphi) > phi0:
                qx.append((i + 1) % sx)
                qy.append((j + 1) % sy)
                qi.append(int(np.round(dphi / (2 * np.pi))))  # Winding number
                q_direction.append(np.sign(total_flux))  # Record defect direction
                
    return np.array(qx), np.array(qy), np.array(qi), np.array(q_direction)

# Function to calculate spot size and intensity distribution based on the positions of the defects
def spot_size_distro(nx, ny, c):
    L = c.shape[0]  # L is the size of the concentration matrix
    x, y = np.arange(c.shape[0]), np.arange(c.shape[1])  
    X, Y = np.meshgrid(x, y)
    
    # Calculate phi (angle) from nx and ny components
    phi = np.arctan2(ny, nx)
    
   
    qx, qy, qi, q_direction = find_defects(phi)
    
    # Identify inward-pointing defects of topological charge +1 similarly one can find outward-pointing defect by changing q_direction to +1
    inward_pid = (qi == 1) & (q_direction == -1)
    
    # Calculate pairwise distances between defects, with periodic boundary conditions
    xi, xj = qx[:, np.newaxis], qx[np.newaxis, :]
    xij = xi - xj - np.round((xi - xj) / L) * L
    
    yi, yj = qy[:, np.newaxis], qy[np.newaxis, :]
    yij = yi - yj - np.round((yi - yj) / L) * L
    
    rij = np.sqrt(xij**2 + yij**2)
    rij[rij == 0] = np.nan  # Handle zero distance (avoid division by zero)
    rSpot = np.nanmin(rij, axis=0)  # Minimum distance to the nearest defect
    
    inAsters = np.where(inward_pid)[0]  # Indices of inward-pointing defects
    C = c.T  
    
    # Initialize lists for spot size and intensity
    spotSize, spotIntensity = [], []
    
    # Loop over inward-pointing defects to calculate spot size and intensity
    for i in inAsters:
        r = rSpot[i]  # Get radius of defect spot
        Xi = X - qx[i] + 0.5 - np.round((X - qx[i] + 0.5) / L) * L  # Adjust for periodic boundary
        Yi = Y - qy[i] + 0.5 - np.round((Y - qy[i] + 0.5) / L) * L
        Rij = Xi**2 + Yi**2  # Squared distance from defect
        iid = Rij < r**2  # Points within the defect's spot
        spotIntensity.append(np.sum(C[iid]))  # Calculate intensity in the spot
        spotSize.append(r**2)  # Store the area of the spot
        
    return np.array(spotSize), np.array(spotIntensity)

# Function to process files and count defects over time
def process_files():
    t_values = np.arange(0, 10000001, 100000)  # Define the time steps to analyze
    results = []

    for t in t_values:
        
        nxfile = f"data/nx_{t}.txt"  # File for nx component
        nyfile = f"data/ny_{t}.txt" # File for ny component
        
        # Load the nx, ny components from the files
        nx, ny = np.loadtxt(nxfile), np.loadtxt(nyfile)
        
        # Calculate phi (angle) and find defects
        phi = np.arctan2(ny, nx)
        qx, qy, qi, q_direction = find_defects(phi)
        
        # Classify defects into positive, negative, and total counts
        pid, nid = qi == 1, qi == -1
        out_pid, in_pid = (qi == 1) & (q_direction == +1), (qi == 1) & (q_direction == -1)
        
        # Store the defect counts for each time step
        results.append([len(qi[pid]), len(qi[nid]), len(qi[pid]) + len(qi[nid])])

    return results

# Main function to process files and save results
if __name__ == "__main__":
    results = process_files()
    #print(results)

    # Save defect counts to a text file
    with open("analysis/defect_counts.txt", "w") as f:
        f.write("Positive Defects\tNegative Defects\tTotal Defects\n")
        for res in results:
            f.write("\t".join(map(str, res)) + "\n")

    # Load the field data for the last time step
    nx = np.loadtxt("data/nx_10000000.txt")
    ny = np.loadtxt("data/ny_10000000.txt")
    c = np.loadtxt("data/c_1000000.txt")

    # Calculate the angle and find defects
    phi = np.arctan2(ny, nx)
    qx, qy, qi, q_direction = find_defects(phi)
    q = np.vstack((qx, qy)).T
    pid, nid = qi == 1, qi == -1  
    out_pid, in_pid = (qi == 1) & (q_direction == +1), (qi == 1) & (q_direction == -1)
    
    # Calculate spot size and intensity distribution
    spot_size, intensity = spot_size_distro(nx, ny, c)
    
    # Save defect positions and spot size data
    np.savez("analysis/positions.npz", posid=q[pid], negid=q[nid], out_pid=q[out_pid], in_pid=q[in_pid])
    np.savez("analysis/spot_size.npz", spot_size=spot_size, intensity=intensity)

