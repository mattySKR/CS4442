import numpy as np
from matplotlib import pyplot as plt

# I have made my own regularization class
from regularization_class import L2_Regularization


# ------------------------------ Part a) Train data-------------------------------------

# Extracting the training values for x and y from the given files
xtr = np.loadtxt('hw1xtr.dat.txt')
ytr = np.loadtxt('hw1ytr.dat.txt')

# Assigning values for easier distinction between variables
x = xtr
y = ytr

# Generating L2_Regularization objects with different lambda values
reg1 = L2_Regularization(0.01)
reg2 = L2_Regularization(0.1)
reg3 = L2_Regularization(1)
reg4 = L2_Regularization(10)
reg5 = L2_Regularization(100)
reg6 = L2_Regularization(1000)
reg7 = L2_Regularization(10000)

# Training data with different lambda values
reg1.fit(x, y)
reg2.fit(x, y)
reg3.fit(x, y)
reg4.fit(x, y)
reg5.fit(x, y)
reg6.fit(x, y)
reg7.fit(x, y)

# Generating our functions
x1 = xtr
x2 = np.power(x1, 2)
x3 = np.power(x1, 3)
x4 = np.power(x1, 4)

l1 = reg1._coeff_weight[0] * x4 + reg1._coeff_weight[1] * x3 + reg1._coeff_weight[2] * x2 + reg1._coeff_weight[3] * x1 + [1.6375279]
l2 = reg2._coeff_weight[0] * x4 + reg2._coeff_weight[1] * x3 + reg2._coeff_weight[2] * x2 + reg2._coeff_weight[3] * x1 + [1.6375279]
l3 = reg3._coeff_weight[0] * x4 + reg3._coeff_weight[1] * x3 + reg3._coeff_weight[2] * x2 + reg3._coeff_weight[3] * x1 + [1.6375279]
l4 = reg4._coeff_weight[0] * x4 + reg4._coeff_weight[1] * x3 + reg4._coeff_weight[2] * x2 + reg4._coeff_weight[3] * x1 + [1.6375279]
l5 = reg5._coeff_weight[0] * x4 + reg5._coeff_weight[1] * x3 + reg5._coeff_weight[2] * x2 + reg5._coeff_weight[3] * x1 + [1.6375279]
l6 = reg6._coeff_weight[0] * x4 + reg6._coeff_weight[1] * x3 + reg6._coeff_weight[2] * x2 + reg6._coeff_weight[3] * x1 + [1.6375279]
l7 = reg7._coeff_weight[0] * x4 + reg7._coeff_weight[1] * x3 + reg7._coeff_weight[2] * x2 + reg7._coeff_weight[3] * x1 + [1.6375279]


# Calculating the train errors for each function
err_train1 = (1 / 40) * np.sum((l1 - y) ** 2)
err_train2 = (1 / 40) * np.sum((l2 - y) ** 2)
err_train3 = (1 / 40) * np.sum((l3 - y) ** 2)
err_train4 = (1 / 40) * np.sum((l4 - y) ** 2)
err_train5 = (1 / 40) * np.sum((l5 - y) ** 2)
err_train6 = (1 / 40) * np.sum((l6 - y) ** 2)
err_train7 = (1 / 40) * np.sum((l7 - y) ** 2)

# Printing out each train error to the screen
print("The train error 1 is: ", err_train1)
print("The train error 2 is: ", err_train2)
print("The train error 3 is: ", err_train3)
print("The train error 4 is: ", err_train4)
print("The train error 5 is: ", err_train5)
print("The train error 6 is: ", err_train6)
print("The train error 7 is: ", err_train7)

# Preparing the values for plotting
lambdas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
errors_train = [err_train1, err_train2, err_train3, err_train4, err_train5, err_train6, err_train7]


# Plotting Training Error as a Function of Lambda
plot1 = plt.figure(1)
plt.title("Training Error as a Function of Lambda ", color="black")
plt.xlabel("Lambda", color="black")
plt.ylabel("Training error", color="black")
plt.plot(lambdas, errors_train, color='red', linewidth=2, marker="o")
plt.xscale("log")
plt.grid(alpha=0.6)


# ------------------------------ Part a) Test data ----------------------------------------

# Extracting the test values for x and y from the given files
xte = np.loadtxt('hw1xte.dat.txt')
yte = np.loadtxt('hw1yte.dat.txt')

# Assigning values for easier distinction between variables
x_test = xte
y_test = yte

# Generating our functions for test data
x1_test = xte
x2_test = np.power(x1_test, 2)
x3_test = np.power(x1_test, 3)
x4_test = np.power(x1_test, 4)

l1_test = reg1._coeff_weight[0] * x4_test + reg1._coeff_weight[1] * x3_test + reg1._coeff_weight[2] * x2_test + reg1._coeff_weight[3] * x1_test + [1.6375279]
l2_test = reg2._coeff_weight[0] * x4_test + reg2._coeff_weight[1] * x3_test + reg2._coeff_weight[2] * x2_test + reg2._coeff_weight[3] * x1_test + [1.6375279]
l3_test = reg3._coeff_weight[0] * x4_test + reg3._coeff_weight[1] * x3_test + reg3._coeff_weight[2] * x2_test + reg3._coeff_weight[3] * x1_test + [1.6375279]
l4_test = reg4._coeff_weight[0] * x4_test + reg4._coeff_weight[1] * x3_test + reg4._coeff_weight[2] * x2_test + reg4._coeff_weight[3] * x1_test + [1.6375279]
l5_test = reg5._coeff_weight[0] * x4_test + reg5._coeff_weight[1] * x3_test + reg5._coeff_weight[2] * x2_test + reg5._coeff_weight[3] * x1_test + [1.6375279]
l6_test = reg6._coeff_weight[0] * x4_test + reg6._coeff_weight[1] * x3_test + reg6._coeff_weight[2] * x2_test + reg6._coeff_weight[3] * x1_test + [1.6375279]
l7_test = reg7._coeff_weight[0] * x4_test + reg7._coeff_weight[1] * x3_test + reg7._coeff_weight[2] * x2_test + reg7._coeff_weight[3] * x1_test + [1.6375279]

# Calculating the test errors for each function
err_test1 = (1 / 20) * np.sum((l1_test - y_test) ** 2)
err_test2 = (1 / 20) * np.sum((l2_test - y_test) ** 2)
err_test3 = (1 / 20) * np.sum((l3_test - y_test) ** 2)
err_test4 = (1 / 20) * np.sum((l4_test - y_test) ** 2)
err_test5 = (1 / 20) * np.sum((l5_test - y_test) ** 2)
err_test6 = (1 / 20) * np.sum((l6_test - y_test) ** 2)
err_test7 = (1 / 20) * np.sum((l7_test - y_test) ** 2)

# Printing out each test error to the screen
print("The test error 1 is: ", err_test1)
print("The test error 2 is: ", err_test2)
print("The test error 3 is: ", err_test3)
print("The test error 4 is: ", err_test4)
print("The test error 5 is: ", err_test5)
print("The test error 6 is: ", err_test6)
print("The test error 7 is: ", err_test7)

# Preparing the test error values for plotting
errors_test = [err_test1, err_test2, err_test3, err_test4, err_test5, err_test6, err_test7]

# Plotting Test Error as a Function of Lambda
plot2 = plt.figure(2)
plt.title("Testing Error as a Function of Lambda", color="black")
plt.xlabel("Lambda", color="black")
plt.ylabel("Testing error", color="black")
plt.plot(lambdas, errors_test, color='green', linewidth=2, marker='o')
plt.xscale("log")
plt.grid(alpha=0.6)


# ---------------------------------- Part b) ---------------------------------------------

all_w_4 = [reg1._coeff_weight[0], reg2._coeff_weight[0], reg3._coeff_weight[0], reg4._coeff_weight[0], reg5._coeff_weight[0], reg6._coeff_weight[0], reg7._coeff_weight[0]]
all_w_3 = [reg1._coeff_weight[1], reg2._coeff_weight[1], reg3._coeff_weight[1], reg4._coeff_weight[1], reg5._coeff_weight[1], reg6._coeff_weight[1], reg7._coeff_weight[1]]
all_w_2 = [reg1._coeff_weight[2], reg2._coeff_weight[2], reg3._coeff_weight[2], reg4._coeff_weight[2], reg5._coeff_weight[2], reg6._coeff_weight[2], reg7._coeff_weight[2]]
all_w_1 = [reg1._coeff_weight[3], reg2._coeff_weight[3], reg3._coeff_weight[3], reg4._coeff_weight[3], reg5._coeff_weight[3], reg6._coeff_weight[3], reg7._coeff_weight[3]]
# all_w_0 = [reg1._coeff_weight[4], reg2._coeff_weight[4], reg3._coeff_weight[4], reg4._coeff_weight[4], reg5._coeff_weight[4], reg6._coeff_weight[4], reg7._coeff_weight[4]]
all_w_0_unpenalized = [1.6375279, 1.6375279, 1.6375279, 1.6375279, 1.6375279, 1.6375279, 1.6375279]

plot3 = plt.figure(3)
plt.title("Value of Each Weight Parameter as a Function of Lambda", color="black")
plt.xlabel("Lambda", color="black")
plt.ylabel("Weights", color="black")
plt.xscale("log")


plt.plot(lambdas, all_w_0_unpenalized, color="blue", label='w_0', marker="o")
plt.plot(lambdas, all_w_1, color="green", label='w_1', marker="o")
plt.plot(lambdas, all_w_2, color="red", label='w_2', marker="o")
plt.plot(lambdas, all_w_3, color="magenta", label='w_3', marker="o")
plt.plot(lambdas, all_w_4, color="yellow", label='w_4', marker="o")
plt.legend()


# ---------------------------------- Part c) Just the test data plot ---------------------------------------------

plot4 = plt.figure(4)
plt.title("Best fit l2-regularized 4th-order polynomial regression line", color="black")
plt.xlabel("X", color="black")
plt.ylabel("Y", color="black")
plt.scatter(x_test, y_test, color='green')


p = np.poly1d(reg2._coeff_weight.flatten())
xp = np.linspace(0,2.8,40)
plt.plot(xp, p(xp), color='black', linewidth=2)


plt.show()



