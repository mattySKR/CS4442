import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Extracting the data
vals = np.loadtxt('faces.dat.txt')

# Creating 4 pca instances with different
# amount of components for each
pca1 = PCA(10)
pca2 = PCA(100)
pca3 = PCA(200)
pca4 = PCA(399)

# Fitting each pca over the given data
pca1.fit(vals)
pca2.fit(vals)
pca3.fit(vals)
pca4.fit(vals)

# Projecting our data into n-dimensional space,
# where n = 10, 100, 200, 399. PCA toolbox
# performs the same projection approach as
# discussed in Lecture 10, i.e., we project
# our data into n-dimensional space using
# n principal components:
comp1 = pca1.transform(vals)
comp2 = pca2.transform(vals)
comp3 = pca3.transform(vals)
comp4 = pca4.transform(vals)

proj_1 = pca1.inverse_transform(comp1)
proj_2 = pca2.inverse_transform(comp2)
proj_3 = pca3.inverse_transform(comp3)
proj_4 = pca4.inverse_transform(comp4)

# Plotting four 100th reconstructed images
fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0, 0].imshow(np.swapaxes(proj_1[99].reshape(64, 64), 0, 1), cmap='binary_r')
axs[0, 0].set_title('PCs = 10', fontweight='bold')

axs[0, 1].imshow(np.swapaxes(proj_2[99].reshape(64, 64), 0, 1), cmap='binary_r')
axs[0, 1].set_title('PCs = 100', fontweight='bold')

axs[1, 0].imshow(np.swapaxes(proj_3[99].reshape(64, 64), 0, 1), cmap='binary_r')
axs[1, 0].set_title('PCs = 200', fontweight='bold')

axs[1, 1].imshow(np.swapaxes(proj_4[99].reshape(64, 64), 0, 1), cmap='binary_r')
axs[1, 1].set_title('PCs = 399', fontweight='bold')

plt.show()
