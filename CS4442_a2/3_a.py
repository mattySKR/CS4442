import numpy as np
import matplotlib.pyplot as plt

# Extracting the data
vals = np.loadtxt('faces.dat.txt')

# Getting the 100th image
vals_extracted = vals[99, :]

# Reshaping from 4096 to 64x64
final = vals_extracted.reshape(64, 64)

# Plotting
plot1 = plt.figure(1)
plt.title("100th Image", color="black", fontweight='bold')
# Had to rotate the image to view normally
plt.imshow(np.swapaxes(final, 0, 1), cmap='bone')
plt.show()








