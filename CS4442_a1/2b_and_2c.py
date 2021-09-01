import numpy as np
from matplotlib import pyplot as plt


# ------------------------------ Part b) ------------------------------------

# Extracting the training values for x and y from the given files
xtr = np.loadtxt('hw1xtr.dat.txt')
ytr = np.loadtxt('hw1ytr.dat.txt')

# Assigning values for easier distinction between variables
x = xtr
y = ytr

# Creating our matrix for x with column of 1's
temp = np.ones((len(x), 2))
temp[:,0] = np.asarray(x)
x = temp
y = np.asarray(y)

# Obtaining our weights using the closed form from lectures
w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.transpose(x)), y)
print("The weight vector is: ", w)

# Here we obtain our model with the known weights
x1 = xtr
w_0 = w[0]
w_1 = w[1]

# Here is the function/matrix. We could have also
# obtained the matrix h = np.dot(w,x.T) from the dot product instead
h = w_0 * x1 + w_1

# Finally plotting our regression model and training data
plot1 = plt.figure(1)
plt.title("Linear Regression and Training Data", color="black")
plt.xlabel("XTR", color="black")
plt.ylabel("YTR", color="black")

plt.scatter(xtr, y, color='red')
plt.plot(xtr, h, color='black', linewidth=2)

# Finally, we are calculating our training error using the given formula
err_train = (1 / 40) * np.sum((h - y) ** 2)
print("The train error is: ", err_train)


# -------------------------------- Part c) -------------------------------------

# Extracting the test values for x and y from the given files
xte = np.loadtxt('hw1xte.dat.txt')
yte = np.loadtxt('hw1yte.dat.txt')

# Assigning values for easier distinction between variables
x_test = xte
y_test = yte

# Using the previously obtained weights and applying them on
# the test data for x, so we can later compute the test error
x1_test = xte

# Function to use for test error
h_test = w_0 * x1_test + w_1

# Finally plotting our linear regression model and test data
plot2 = plt.figure(2)
plt.title("Linear Regression and Testing Data", color="black")
plt.xlabel("XTE", color="black")
plt.ylabel("YTE", color="black")

plt.scatter(x_test, y_test, color='green')
plt.plot(xte, h_test, color='black', linewidth=2)

# Finally, we are calculating our test error using the given formula
err_test = (1 / 20) * np.sum((h_test - y_test) ** 2)
print("The test error is: ", err_test)

# Make the figures show
plt.show()
