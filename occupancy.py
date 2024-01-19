import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def occupancy( unitSize, x ):
   y = x-(x%unitSize) + np.where( (x%unitSize)>0, 1, 0 )*unitSize
   return x/y

fig, ax = plt.subplots(2, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.15, hspace=0.02)

xmin = 1
xmax = 1024

x = np.linspace(xmin, xmax, num=int(xmax-xmin)+1)
ax[1][0].plot(x, occupancy(32, x), 'o', markersize=0.5, color='black')
ax[1][1].plot(x, occupancy(72, x), 'o', markersize=0.5, color='black')


File = "throughput_vs_block_grid_size.dat"

gridSize  = np.loadtxt( File,  usecols=(0), unpack=True, delimiter=' ' )
blockSize = np.loadtxt( File,  usecols=(1), unpack=True, delimiter=' ' )
Time      = np.loadtxt( File,  usecols=(2), unpack=True, delimiter=' ' )

ax[0][0].plot(blockSize, Time, 'o', markersize=0.5, color='black')
ax[0][1].plot(gridSize,  Time, 'o', markersize=0.5, color='black')

ax[0][1].set_yscale('log')
ax[1][0].set_yscale('log')
ax[1][1].set_yscale('log')
ax[0][0].set_yscale('log')

ax00max = ax[0][0].get_ylim()[1]
ax00min = ax[0][0].get_ylim()[0]
ax10max = ax[1][0].get_ylim()[1]
ax10min = ax[1][0].get_ylim()[0]

ax01max = ax[0][1].get_ylim()[1]
ax01min = ax[0][1].get_ylim()[0]
ax11max = ax[1][1].get_ylim()[1]
ax11min = ax[1][1].get_ylim()[0]

for threadIdx in range(32,1025,32):
   ax[0][0].vlines( threadIdx, ax00min, ax00max, color='red', linestyles='dashed' )
   ax[1][0].vlines( threadIdx, ax10min, ax10max, color='red', linestyles='dashed' )

for blockIdx in range(72,1025,72):
   ax[0][1].vlines( blockIdx, ax01min, ax01max, color='red', linestyles='dashed' )
   ax[1][1].vlines( blockIdx, ax11min, ax11max, color='red', linestyles='dashed' )

ax[0][0].set_ylim(ax00min,ax00max)
ax[0][1].set_ylim(ax01min,ax01max)
ax[1][0].set_ylim(ax10min,ax10max)
ax[1][1].set_ylim(ax11min,ax11max)

ax[0][0].set_ylabel("Time (sec)", fontsize=15)
ax[1][0].set_ylabel("CUDA core occupancy", fontsize=15)
ax[1][0].set_xlabel("Number of threads per block", fontsize=15)
ax[0][1].set_ylabel("Time (sec)", fontsize=15)
ax[1][1].set_ylabel("SM occupancy", fontsize=15)
ax[1][1].set_xlabel("Number of blocks per grid", fontsize=15)

ax[0][0].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[0][1].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[1][0].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[1][1].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)

ax[0][0].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[0][1].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[1][0].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[1][1].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)

ax[1][0].set_xticks(np.arange(0, max(x)+1, 32, dtype=np.short), dtype=np.short)
ax[1][1].set_xticks(np.arange(0, max(x)+1, 72, dtype=np.short), dtype=np.short)

ax[1][0].set_xticklabels(ax[1][0].get_xticks(), rotation = 90)
ax[1][1].set_xticklabels(ax[1][1].get_xticks(), rotation = 90)

ax[0][0].set_xticks([])
ax[0][1].set_xticks([])

ax[0][0].set_xlim(1,1024)
ax[0][1].set_xlim(1,1024)
ax[1][0].set_xlim(1,1024)
ax[1][1].set_xlim(1,1024)

#plt.show()
plt.savefig("throughput_vs_block_grid_size.png", dpi=96, format='png', bbox_inches='tight')
