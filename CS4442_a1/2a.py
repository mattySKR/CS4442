import numpy as np
import matplotlib.pyplot as plt

# Extracting training data
xtr = np.loadtxt('hw1xtr.dat.txt')
ytr = np.loadtxt('hw1ytr.dat.txt')

x = xtr
y = ytr

# Plotting first graph with training data
plot1 = plt.figure(1)
plt.title("XTR vs. YTR graph")
plt.xlabel("XTR")
plt.ylabel("YTR")
plt.plot(x, y, 'or')

# -----------------------------------------------------------

# Extracting testing data
xte = np.loadtxt('hw1xte.dat.txt')
yte = np.loadtxt('hw1yte.dat.txt')

x_test = xte
y_test = yte

# Plotting second graph with testing data
plot2 = plt.figure(2)
plt.title("XTE vs. YTE graph")
plt.xlabel("XTE")
plt.ylabel("YTE")
plt.plot(x_test, y_test, 'og')

plt.show()

