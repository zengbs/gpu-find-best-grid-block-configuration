import numpy as np
import matplotlib.pyplot as plt

# Define the function based on the given expression
def Ceiling(x, unitSize):
    return x - (x % unitSize) + np.where((x % unitSize) > 0, 1, 0) * unitSize

# Generate a range of x values
numPoints = 129*2
unitSize = 32  # Example unit size

x_values = np.linspace(0, numPoints-2, numPoints-1, endpoint=True, dtype=np.int16)

# Calculate y values using the function
y_values = Ceiling(x_values, unitSize)

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot for y values excluding the start and end of each segment
x = x_values[x_values%unitSize != 0]
plt.scatter(x, Ceiling(x, unitSize), facecolors='blue', edgecolors='blue', label='y = adjusted x', s=20)


# Adding closed circles
# Closed circles at the end of each segment
x = x_values[x_values%unitSize == 0]
plt.scatter(x, Ceiling(x, unitSize), facecolors='blue', edgecolors='blue', s=80)

# Adding open circles
# Open circles at the start of each segment
x = x_values[x_values%unitSize == 0]
plt.scatter(x, Ceiling(x, unitSize)+unitSize, facecolors='none', edgecolors='blue', s=80)

plt.xlabel('x', fontsize=20)
plt.ylabel('Ceiling(x, unitSize=32)', fontsize=20)
plt.ylim(0, max(y_values))
plt.xlim(0, max(x_values))
plt.tick_params(axis='both', direction='in', labelsize=15, length=5 )
plt.grid(True)
plt.xticks(np.arange(0, numPoints, unitSize, dtype=np.short))
plt.savefig("fig__ceiling.png", dpi=96, format='png', bbox_inches='tight')
plt.show()
