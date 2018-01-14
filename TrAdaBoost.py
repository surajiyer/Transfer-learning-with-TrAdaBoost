# coding: utf-8

"""
    description:
        Scikit-learn compatible Implementation of the 
        TrAdaBoost algorithm from the ICML'07 paper
        "Boosting for Transfer Learning"
    author: Suraj Iyer
"""

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.svm import SVC
from sklearn.utils.validation import *
from sklearn.utils.validation import _num_samples, _is_arraylike
import numpy as np


class TrAdaBoostClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator=SVC(), n_iters=10, init_weights=None, verbose=False):
        assert getattr(estimator, 'fit', None) is not None
        assert getattr(estimator, 'predict', None) is not None
        assert isinstance(n_iters, int) and n_iters > 0
        assert _is_arraylike(init_weights) and all(isinstance(w, int) for w in init_weights)
        assert has_fit_parameter(estimator, "sample_weight")

        self.estimator = estimator
        self.n_iters = n_iters
        self.init_weights = init_weights if init_weights else np.ones(n_iters)
        self.verbose = verbose

    def _normalize_weights(self, weights):
        return weights/np.sum(weights)

    def _calculate_error(self, y_true, y_pred, weights):
        assert check_consistent_length(y_true, y_pred)
        n = _num_samples(y_true)
        return np.sum(weights[n:] * np.abs(y_pred - y_true) / np.sum(weights[n:]))

    def fit(self, X_same, y_same=None, X_diff=None, y_diff=None):
        # initialize data
        check_X_y(X_same, y_same)
        check_X_y(X_diff, y_diff)
        X = np.concatenate([X_same, X_diff])
        y = np.concatenate([y_same, y_diff])

        # initialize weights
        n, m = _num_samples(X_same), _num_samples(X_diff)
        n_samples = n + m
        assert len(self.init_weights) == n_samples
        weights = np.ones((self.n_iters, n_samples))
        weights[0] = self.init_weights
        P = np.ones((self.n_iters, n_samples))

        # initialize error vector
        error = np.zeros(self.n_iters)
        beta0 = 1 / (1 + np.sqrt(2 * np.log(n / self.n_iters)))
        beta = np.zeros(self.n_iters)

        # initialize estimator list for each iteration
        estimators = []

        for t in np.arange(self.n_iters):
            P[t] = self._normalize_weights(self.weights_[t])

            # Call learner
            est = clone(self.estimator).fit(X, y, sample_weight=P[t])
            y_same_pred = self.estimator.predict(X_same)

            # calculate the error on same-distribution data (X_same)
            error[t] = self._calculate_error(y_same, y_same_pred, self.weights_[t])
            # error[t] = min([error[t], 0.49])  # error must be less than 0.5
            if error[t] > 0.5 or error[t] == 0:
                # if the error is 0 or > 0.5, stop updating weights
                self.n_iters = t
                weights = weights[:t]
                beta = beta[:t]

                if self.verbose:
                    if error[t] > 0.5:
                        print("Early stopping because error: {} > 0.5".format(error[t]))
                    else:
                        print("Early stopping because error is zero.")
                break
            beta[t] = error[t] / (1 - error[t])

            # Update the new weight vector
            if t < self.n_iters - 1:
                y_diff_pred = est.predict(X_diff)
                weights[t+1][:n] = weights[t][:n] * (beta0 ** np.abs(y_diff_pred - y_diff))
                weights[t+1][n:] = weights[t][n:] * (beta[t] ** -np.abs(y_same_pred - y_same))

            estimators.append(est)

        if self.verbose:
            print("Number of iterations run: {}".format(self.n_iters))
        self.fitted = True
        self.weights_ = weights
        self.beta_ = beta
        self.estimators_ = estimators

        return self

    def _predict_one(self, x):
        """
        Output the hypothesis for a single instance
        :param x: array-like
            target label of a single instance from each iteration in order
        :return: [0, 1]
        """
        x, N = check_array(x), self.n_iters
        # replace 0 by 1 to avoid zero division and remove it from the product
        beta = [self.beta_[t] if self.beta_[t] != 0 else 1 for t in range(np.ceil(N), N)]
        cond = np.prod([b ** -x[t] for b in beta]) >= np.prod([b ** -0.5 for b in beta])
        return int(cond)

    def predict(self, X):
        check_is_fitted(self, 'fitted')
        y_pred_list = np.array([est.predict(X) for est in self.estimators_]).T
        y_pred = map(self._predict_one, y_pred_list)
        return y_pred
