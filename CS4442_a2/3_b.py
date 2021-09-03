import numpy as np
import matplotlib.pyplot as plt

# Extracting the data
vals = np.loadtxt('faces.dat.txt')

# Finding the mean of each column
mean_cols = vals.mean(axis=0)

# Subtracting the mean
centred = vals - mean_cols

# Getting the 100th image
vals_extracted = centred[99, :]

# Reshaping from 4096 to 64x64
final_matrix = vals_extracted.reshape(64, 64)

# Plotting
plot1 = plt.figure(1)
plt.title("100th Image With Removed Mean", fontweight='bold')
# Had to rotate the image to view normally
plt.imshow(np.swapaxes(final_matrix, 0, 1), cmap='bone')
plt.show()



