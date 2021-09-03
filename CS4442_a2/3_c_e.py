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
sort_eigenvectors = comp[:,descend_index]

# Printing put the eigenvalues in descending order to the screen
eigen_tuples = [(np.abs(descend_eigenvalues[i]), sort_eigenvectors[:,i]) for i in range(len(descend_eigenvalues))]
print("Eigenvalues in descending order:")
for i in eigen_tuples:
    print(i[0])

# Can print the smallest (i.e. 400th) to the screen if you wish
'''
min_eigenval = np.amin(descend_eigenvalues)
print("\nSmallest Eigenvalue:")
print(min_eigenval)
'''


# Plotting the eigenvalues in descending order
plt.figure(figsize=(24,10))
plt.title("Eigenvalues In Descending Order", color='black', fontweight='bold')
plt.xlabel("i-th Eigenvalue (400 Total Eigenvalues)", color='black', fontweight='bold')
plt.ylabel("Values of Each Eigenvalue", color='black', fontweight='bold')
plt.plot(descend_eigenvalues, 'xr', label='Eigenvalue')
plt.legend()

# --------------------------- part d) -------------------------------------

# Here we are computing the percentage of variance that
# is accounted for by each component. Thus, we need to
# divide the eigenvalue of each component by the sum
# of all eigenvalues
eigenval_sum = descend_eigenvalues.sum()

div = descend_eigenvalues / eigenval_sum

percent_variance = div * 100

s = percent_variance[:123].sum()
print(s)


pca_comps = []
for i in range(1, 11):

    pca_comps.append('PC' + str(i))


variance = []
for j in range(0,10):
    variance.append(percent_variance[j])


fig1 = plt.figure(figsize=(40,8))
plt.title("Principle Components and Respected Variance Accounted", color='black', fontweight='bold')
plt.ylabel("Variance(%)", color='black', fontweight='bold')
plt.bar(pca_comps, variance, align='center', ec='black')

for index, value in enumerate(variance):
    plt.text(index, value, str("%.2f" % value + '%'), fontweight='bold')

plt.axhline(y=5, color='r', linestyle='-')


plt.show()


