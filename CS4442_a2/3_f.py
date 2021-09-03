import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Extracting the data
vals = np.loadtxt('faces.dat.txt')

# Creating our PCA object instance
pca = PCA(400)

# Fitting pca over the given data
pca.fit(vals)

# Computing our Principal components or Eigenvectors
comp = pca.components_

# Computing our Explained Variance or Eigenvalues
var = pca.explained_variance_

# Sorting the eigenvalues in descending order
# It looks like pca tool has taken care of it, but
# will sort just in case.
descend_index = np.argsort(var)[::-1]
descend_eigenvalues = var[descend_index]

# Can also sort eigenvectos based on the descending order
# of eigenvalues above. This will arrange the principal
# components in descending order of their variability
sort_eigenvectors = comp[descend_index,:]

# Printing put the eigenvalues iin descending order to the screen
eigen_tuples = [(np.abs(descend_eigenvalues[i]), sort_eigenvectors[i,:]) for i in range(len(descend_eigenvalues))]
print("Eigenvalues in descending order:")
for i in eigen_tuples:
    print(i[0])


# ----------------------- The above is just a copy of 3_c.py ------------------------


# Displaying out top 5 eigenfaces
eigenface_1 = plt.figure(1)
plt.title("First Best")
plt.imshow(np.swapaxes(sort_eigenvectors[0].reshape(64, 64), 0, 1), cmap='bone')

eigenface_2 = plt.figure(2)
plt.title("Second Best")
plt.imshow(np.swapaxes(sort_eigenvectors[1].reshape(64, 64), 0, 1), cmap='bone')

eigenface_3 = plt.figure(3)
plt.title("Third Best")
plt.imshow(np.swapaxes(sort_eigenvectors[2].reshape(64, 64), 0, 1), cmap='bone')

eigenface_4 = plt.figure(4)
plt.title("Fourth Best")
plt.imshow(np.swapaxes(sort_eigenvectors[3].reshape(64, 64), 0, 1), cmap='bone')

eigenface_5 = plt.figure(5)
plt.title("Fifth Best")
plt.imshow(np.swapaxes(sort_eigenvectors[4].reshape(64, 64), 0, 1), cmap='bone')

plt.show()



