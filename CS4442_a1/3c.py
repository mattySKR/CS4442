import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

# For the issue with the very last graph, please scroll to very bottom

# Extracting the training values for x and y from the given files
xtr = np.loadtxt('hw1xtr.dat.txt')
ytr = np.loadtxt('hw1ytr.dat.txt')

# Need to reshape so we can use the cross_val_score
x = xtr.reshape(-1, 1)
y = ytr.reshape(-1, 1)

mean_scores = []
# Was not able to call cross-validation library using
# my obtained l2_regularization, so have to use
# the Ridge model library instead
for cur_alpha in [0.01, 0.1, 1, 10, 100, 1000, 10000]:

    lr_obj = Ridge(alpha=cur_alpha)
    scores = cross_val_score(lr_obj, x, y, cv=5)
    mean_scores += [abs(np.mean(scores))]
    print("lambda {0}\t R^2 scores: {1}\t Mean R^2: {2}".format(cur_alpha,scores, abs(np.mean(scores))))


# It is very very tough to understand the assignment requirements due
# to the lack of information.

# So I assume we have to plot the errors we got as a function of lambda
lambdas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
plot1 = plt.figure(1)
plt.title("Average error on the validation set", color="black")
plt.xlabel("Lambda", color="black")
plt.ylabel("Average Error", color="black")
plt.plot(lambdas, mean_scores, color='magenta', linewidth=2, marker="o")
plt.xscale("log")
plt.grid(alpha=0.6)

# Now refitting on the full training set
ridgeCV_object = RidgeCV(alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0), cv=5)
ridgeCV_object.fit(x, y)
print("\n")
print("Best model searched:\nalpha = {}\nintercept = {}\nweights = {} ".format(ridgeCV_object.alpha_, ridgeCV_object.intercept_, ridgeCV_object.coef_))

# !!!!!! For some reason, the above ridgeCV is only giving me one weight coefficient instead of the wanted 5 weights !!!!!
# I ran out of time and was not able to sort this issue out

#plt.show()
