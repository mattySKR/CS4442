import numpy as np
from matplotlib import pyplot as plt


# -------------------- Part e) where we are repeating b) -------------------------

# Extracting the training values for x and y from the given files
xtr = np.loadtxt('hw1xtr.dat.txt')
ytr = np.loadtxt('hw1ytr.dat.txt')

# Assigning values for easier distinction between variables
x = xtr
y = ytr

# Creating our matrix for x with column of 1's, column of x^2's and column of x^3's
temp = np.ones((len(x), 4))
temp[:,0] = np.power(np.asarray(x), 3)
temp[:,1] = np.power(np.asarray(x), 2)
temp[:,2] = np.asarray(x)
x = temp
y = np.asarray(y)

# Obtaining our weights using the closed form from lectures
w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.transpose(x)), y)
print("The weight vector is: ", w)

# Here we obtain our model with the known weights
x1 = xtr
x2 = np.power(x1, 2)
x3 = np.power(x1, 3)

w_0 = w[0]
w_1 = w[1]
w_2 = w[2]
w_3 = w[3]

# Here is the function. We could have also
# obtained the matrix h = np.dot(w,x.T) from the dot product instead
h = w_0 * x3 + w_1 * x2 + w_2 * x1 + w_3

# We will use poly1d() function for plotting instead of using our h above,
# where we do the same thing manually, essentially. This
# will make the graph look more appealing
pol_reg = np.poly1d(w)
x_pol_reg = np.linspace(0,2.6,40)

# Finally plotting our regression model and training data
plot1 = plt.figure(1)
plt.title("3rd-Order Polynomial Regression and Training Data", color="black")
plt.xlabel("XTR", color="black")
plt.ylabel("YTR", color="black")
plt.scatter(xtr, y, color='red')

# You may uncomment this and see what the plot looks like
# plt.plot(xtr, h, color='black', linewidth=2)

# So, again, to make graph look prettier we are plotting
# (xp, p(xp)) instead of the commented out above (xtr, h)
plt.plot(x_pol_reg, pol_reg(x_pol_reg), color='black', linewidth=2)

# Finally, we are calculating our training error using the given formula
err_train = (1 / 40) * np.sum((h - y) ** 2)
print("The train error is: ", err_train)


# -------------------- Part e) where we are repeating c) -------------------------

# Extracting the test values for x and y from the given files
xte = np.loadtxt('hw1xte.dat.txt')
yte = np.loadtxt('hw1yte.dat.txt')

# Assigning values for easier distinction between variables
x_test = xte
y_test = yte

# Using the previously obtained weights and applying them on
# the test data for x, so we can later compute the test error
x1_test = xte
x2_test = np.power(x1_test, 2)
x3_test = np.power(x1_test, 3)

# Function to use for test error
h_test = w_0 * x3_test + w_1 * x2_test + w_2 * x1_test + w_3

# Finally plotting our regression model and test data
plot2 = plt.figure(2)
plt.title("3rd-Order Polynomial Regression and Testing Data", color="black")
plt.xlabel("XTE", color="black")
plt.ylabel("YTE", color="black")
plt.scatter(x_test, y_test, color='green')

# plt.plot(xte, h_test, color='black', linewidth=2)

# So, again, to make graph look prettier we are plotting
# (xp, p(xp)) instead of the commented out above (xte, h_test)
plt.plot(x_pol_reg, pol_reg(x_pol_reg), color='black', linewidth=2)


# Finally, we are calculating our test error using the given formula
err_test = (1 / 20) * np.sum((h_test - y_test) ** 2)
print("The test error is: ", err_test)

# Make the figures show
plt.show()


