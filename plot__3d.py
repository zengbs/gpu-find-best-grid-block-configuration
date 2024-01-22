import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator

File = "log"


gridSize  = np.loadtxt( File,  usecols=(0), unpack=True, delimiter=',' )
blockSize = np.loadtxt( File,  usecols=(1), unpack=True, delimiter=',' )
Time      = np.loadtxt( File,  usecols=(2), unpack=True, delimiter=',' )


fig = plt.figure(figsize=(12, 12), dpi=160)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(gridSize, blockSize, np.log10(Time), c=np.log10(Time), cmap=cm.jet, edgecolors=None)


ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.set_xlabel("Number of blocks")
ax.set_ylabel("Number of threads per block")
ax.set_zlabel("Time (sec)")

plt.show()
#plt.savefig("test.png", dpi=96, format='png', bbox_inches='tight')
