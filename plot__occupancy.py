import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



def Ceiling(x_values, unitSize):
   return x_values-(x_values%unitSize) + np.where( (x_values%unitSize)>0, 1, 0 )*unitSize


def Occupancy( x_values, unitSize ):
   return x_values/Ceiling(x_values, unitSize)

fig, ax = plt.subplots(3, 2, figsize=(20, 20))
plt.subplots_adjust(wspace=0.15, hspace=0.02)

warpSize = 32
numSMs = 72

xmin = 1
xmax = 1024
x_values = np.linspace(xmin, xmax, num=int(xmax-xmin)+1)

# Scatter plot for y values excluding the start and end of each segment
x = x_values[x_values%warpSize != 0]
ax[2][0].scatter(x, Ceiling(x, warpSize), facecolors='blue', edgecolors='blue', label='y = adjusted x', s=1)
x = x_values[x_values%numSMs != 0]
ax[2][1].scatter(x, Ceiling(x,   numSMs), facecolors='blue', edgecolors='blue', label='y = adjusted x', s=1)

# Add closed circles at the end of each segment
x = x_values[x_values%warpSize == 0]
ax[2][0].scatter(x, Ceiling(x, warpSize), facecolors='blue', edgecolors='blue', s=30)
x = x_values[x_values%numSMs == 0]
ax[2][1].scatter(x, Ceiling(x,   numSMs), facecolors='blue', edgecolors='blue', s=30)


# Add open circles at the start of each segment
x = x_values[x_values%warpSize == 0]
x = np.insert( x, 0, 0 )
ax[2][0].scatter(x, Ceiling(x, warpSize)+warpSize, facecolors='white', edgecolors='blue', s=30)
x = x_values[x_values%numSMs == 0]
x = np.insert( x, 0, 0 )
ax[2][1].scatter(x, Ceiling(x,   numSMs)+numSMs  , facecolors='white', edgecolors='blue', s=30)



ax[1][0].plot(x_values, Occupancy(x_values, warpSize), 'o', markersize=0.5, color='blue')
ax[1][1].plot(x_values, Occupancy(x_values,   numSMs), 'o', markersize=0.5, color='blue')


File = "throughput_vs_block_grid_size.dat"

gridSize  = np.loadtxt( File,  usecols=(0), unpack=True, delimiter=' ' )
blockSize = np.loadtxt( File,  usecols=(1), unpack=True, delimiter=' ' )
Time      = np.loadtxt( File,  usecols=(2), unpack=True, delimiter=' ' )

ax[0][0].plot(blockSize, Time, 'o', markersize=0.5, color='blue')
ax[0][1].plot(gridSize,  Time, 'o', markersize=0.5, color='blue')

ax[0][1].set_yscale('log')
ax[1][0].set_yscale('log')
ax[1][1].set_yscale('log')
ax[0][0].set_yscale('log')

ax00max = ax[0][0].get_ylim()[1]
ax00min = ax[0][0].get_ylim()[0]
ax10max = ax[1][0].get_ylim()[1]
ax10min = ax[1][0].get_ylim()[0]
ax20max = ax[2][0].get_ylim()[1]
ax20min = ax[2][0].get_ylim()[0]

ax01max = ax[0][1].get_ylim()[1]
ax01min = ax[0][1].get_ylim()[0]
ax11max = ax[1][1].get_ylim()[1]
ax11min = ax[1][1].get_ylim()[0]
ax21max = ax[2][1].get_ylim()[1]
ax21min = ax[2][1].get_ylim()[0]

for threadIdx in range(warpSize,1025,warpSize):
   ax[0][0].vlines( threadIdx, ax00min, ax00max, color='red', linestyles='dashed' )
   ax[1][0].vlines( threadIdx, ax10min, ax10max, color='red', linestyles='dashed' )
   ax[2][0].vlines( threadIdx, ax20min, ax20max, color='red', linestyles='dashed' )

for blockIdx in range(numSMs,1025,numSMs):
   ax[0][1].vlines( blockIdx, ax01min, ax01max, color='red', linestyles='dashed' )
   ax[1][1].vlines( blockIdx, ax11min, ax11max, color='red', linestyles='dashed' )
   ax[2][1].vlines( blockIdx, ax21min, ax21max, color='red', linestyles='dashed' )

ax[0][0].set_ylim(ax00min,ax00max)
ax[0][1].set_ylim(ax01min,ax01max)
ax[1][0].set_ylim(ax10min,ax10max)
ax[1][1].set_ylim(ax11min,ax11max)
ax[2][0].set_ylim(ax20min,ax20max)
ax[2][1].set_ylim(ax21min,ax21max)

ax[0][0].set_ylabel("Time (sec)", fontsize=15)
ax[1][0].set_ylabel("Warp occupancy", fontsize=15)
ax[0][1].set_ylabel("Time (sec)", fontsize=15)
ax[1][1].set_ylabel("SM occupancy", fontsize=15)
ax[2][0].set_ylabel("ceiling(blockSize, warpSize)", fontsize=13)
ax[2][1].set_ylabel("ceiling(gridSize, # SMs)", fontsize=13)
ax[2][0].set_xlabel("Number of threads per block", fontsize=15)
ax[2][1].set_xlabel("Number of blocks per grid", fontsize=15)

ax[0][0].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[0][1].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[1][0].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[1][1].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[2][0].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)
ax[2][1].tick_params(axis='both', direction='in', which='major', labelsize=12, length=5)

ax[0][0].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[0][1].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[1][0].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[1][1].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[2][0].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)
ax[2][1].tick_params(axis='both', direction='in', which='minor', labelsize=12, length=3)


ax[0][0].set_xticks([])
ax[0][1].set_xticks([])
ax[1][0].set_xticks([])
ax[1][1].set_xticks([])
ax[2][0].set_xticks(np.arange(0, max(x_values)+1, warpSize, dtype=np.short))#, dtype=np.short)
ax[2][1].set_xticks(np.arange(0, max(x_values)+1, numSMs, dtype=np.short))#, dtype=np.short)

ax[2][0].set_xticklabels(ax[2][0].get_xticks(), rotation = 90)
ax[2][1].set_xticklabels(ax[2][1].get_xticks(), rotation = 90)

ax[0][0].set_xlim(1,1024)
ax[0][1].set_xlim(1,1024)
ax[1][0].set_xlim(1,1024)
ax[1][1].set_xlim(1,1024)
ax[2][0].set_xlim(1,1024)
ax[2][1].set_xlim(1,1024)

#plt.show()
plt.savefig("fig__occupancy.png", dpi=256, format='png', bbox_inches='tight')
