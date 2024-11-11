import numpy as np
import os
from numba import njit

filepath = os.getcwd()
dr = 0.1
L = 32

@njit
def give_n(xid, yid, r, dr, plistx, plisty, L):
    count = 0
    for i in range(len(plistx)):
        dist_abs_x = abs(plistx[i] - xid)
        dist_abs_y = abs(plisty[i] - yid)
        dist_pbc_x = min(dist_abs_x, (L - dist_abs_x))
        dist_pbc_y = min(dist_abs_y, (L - dist_abs_y))

        dist = np.sqrt(dist_pbc_x**2 + dist_pbc_y**2)
        if r <= dist <= (r + dr) and dist != 0:
            count += 1
    return count

# Load positions from npz file
positions_data = np.load(os.path.join(filepath, 'positions.npz'))
posid = positions_data['posid']
negid = positions_data['negid']

plistx = posid[:, 0]  # X-coordinates for positive charges
plisty = posid[:, 1]  # Y-coordinates for positive charges

nlistx = negid[:, 0]  # X-coordinates for negative charges
nlisty = negid[:, 1]  # Y-coordinates for negative charges

g1, g2 = np.array([]), np.array([])
R = np.arange(1, int(L / 2), dr)

for r in R:
    N1, N2 = 0, 0
    for i in range(len(plistx)):
        xid, yid = plistx[i], plisty[i]
        xid2, yid2 = nlistx[i], nlisty[i]

        # Calculate number of neighbors within distance
        n1 = give_n(xid, yid, r, dr, plistx, plisty, L)
        n2 = give_n(xid2, yid2, r, dr, nlistx, nlisty, L)

        N1 += n1
        N2 += n2

    # Normalize counts
    N1 = N1 * L * L / ((len(plistx) ** 2) * (2 * np.pi * r * dr))
    N2 = N2 * L * L / ((len(nlistx) ** 2) * (2 * np.pi * r * dr))
    
    g1 = np.append(g1, N1)
    g2 = np.append(g2, N2)

# Save results to files
g1_file = os.path.join(filepath, "positive_positive_defects.txt")
g2_file = os.path.join(filepath, "negetive_negetive_defects.txt")

np.savetxt(g1_file, g1)
np.savetxt(g2_file, g2)

