import numpy as np


class L2_Regularization:

    def __init__(self, lam):

        self._lambda = lam

    @staticmethod
    def _x_bar(x):

        # Our x matrix with all proper columns
        return np.hstack((np.power(np.asarray(x), 4), np.power(np.asarray(x), 3), np.power(np.asarray(x), 2), x, [1.0]))

    def fit(self, x_tr, y_tr):

        # Fitting the train data for x and y
        X = np.vstack(([self._x_bar(x) for x in x_tr]))
        Y = np.vstack(([y for y in y_tr]))

        # Now we need to compute the model coefficients
        # w = inv(xTx + lambda * I) * xTy
        XT = np.transpose(X)
        XTX = np.matmul(XT, X) + self._lambda * np.identity(X.shape[1])
        self._coeff_weight = np.matmul(np.matmul(np.linalg.inv(XTX), XT), Y)


