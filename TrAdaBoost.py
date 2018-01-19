# coding: utf-8

"""
    description:
        Scikit-learn compatible implementation of the 
        TrAdaBoost algorithm from the ICML'07 paper
        "Boosting for Transfer Learning"
    author: Suraj Iyer
"""

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.svm import SVC
import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.validation import _num_samples


class TrAdaBoostClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=SVC(), n_iters=10, domain_column='domain', verbose=False):
        assert getattr(base_estimator, 'fit', None) is not None
        assert getattr(base_estimator, 'predict', None) is not None
        assert isinstance(n_iters, int) and n_iters > 0
        assert has_fit_parameter(base_estimator, "sample_weight")

        self.base_estimator = base_estimator
        self.n_iters = n_iters
        self.verbose = verbose
        self.domain_column = domain_column

    def set_params(self, **params):
        return super(TrAdaBoostClassifier, self).set_params(**params)

    def _normalize_weights(self, weights):
        return weights/np.sum(weights)

    def _calculate_error(self, y_true, y_pred, weights):
        check_consistent_length(y_true, y_pred, weights)
        return np.sum(weights * np.abs(y_pred - y_true) / np.sum(weights))

    def fit(self, X, y=None, domain_column=None, init_weights=None):
        # initialize data
        domain_column = domain_column if domain_column else self.domain_column
        assert isinstance(domain_column, str)
        domain_column_idx = X.columns.get_loc(domain_column)
        # mask = (X[domain_column] == 1).values.T[0]
        # X.drop(domain_column, axis=1, inplace=True)
        X, y = check_X_y(X, y)

        # extract same domain and diff domain data
        mask = X[:, domain_column_idx] == 1
        X = np.delete(X, domain_column_idx, 1)  # expensive copy TODO: replace by in place drop

        # initialize weights
        n = mask.sum()
        m = _num_samples(X) - n
        n_samples = n + m
        if init_weights is None:
            init_weights = np.ones(n_samples)
        else:
            assert _num_samples(init_weights) == n_samples
        weights = init_weights
        P = np.empty((self.n_iters, n_samples))

        # initialize error vector
        error = np.empty(self.n_iters)
        beta0 = 1 / (1 + np.sqrt(2 * np.log(n / self.n_iters)))
        beta = np.empty(self.n_iters)

        # initialize estimator list for each iteration
        estimators = []

        for t in np.arange(self.n_iters):
            P[t] = self._normalize_weights(weights)

            # Call learner
            est = clone(self.base_estimator).fit(X, y, sample_weight=P[t])
            y_same_pred = est.predict(X[mask, :])

            # calculate the error on same-distribution data (X_same)
            error[t] = self._calculate_error(y[mask], y_same_pred, weights[mask])
            # error[t] = min([error[t], 0.49])  # error must be less than 0.5
            if self.verbose:
                print('Error_{}: {}'.format(t, error[t]))

            if error[t] > 0.5 or error[t] == 0:
                # if the error is 0 or > 0.5, stop updating weights
                self.n_iters = t
                beta = beta[:t]

                if self.verbose:
                    if error[t] > 0.5:
                        print("Early stopping because error: {} > 0.5".format(error[t]))
                    else:
                        print("Early stopping because error is zero.")
                break

            beta[t] = error[t] / (1 - error[t])
            if self.verbose:
                print('beta_{}: {}'.format(t, beta[t]))

            # Update the new weight vector
            if t < self.n_iters - 1:
                y_diff_pred = est.predict(X[~mask, :])
                weights[~mask] = weights[~mask] * (beta0 ** np.abs(y_diff_pred - y[~mask]))
                weights[mask] = weights[mask] * (beta[t] ** -np.abs(y_same_pred - y[mask]))

            estimators.append(est)

        if self.verbose:
            print("Number of iterations run: {}".format(self.n_iters))

        self.fitted_ = True
        self.diff_sample_weights_ = weights
        self.beta_ = beta
        self.estimators_ = estimators
        self.classes_ = getattr(estimators[0], 'classes_', None)
        self.domain_column_idx_ = domain_column_idx

        return self

    def _predict_one(self, x):
        """
        Output the hypothesis for a single instance
        :param x: array-like
            target label of a single instance from each iteration in order
        :return: 0 or 1
        """
        x, N = check_array(x, ensure_2d=False), self.n_iters
        # replace 0 by 1 to avoid zero division and remove it from the product
        beta = [self.beta_[t] if self.beta_[t] != 0 else 1 for t in range(int(np.ceil(N/2)), N)]
        cond = np.prod([b ** -x[t] for b in beta]) >= np.prod([b ** -0.5 for b in beta])
        return int(cond)

    def predict(self, X, domain_column=None):
        check_is_fitted(self, 'fitted_')

        # remove domain column if exists
        domain_column = domain_column if domain_column else self.domain_column
        assert isinstance(domain_column, str)
        X = np.delete(X, self.domain_column_idx_, 1)  # expensive copy TODO: replace by in place drop

        y_pred_list = np.array([est.predict(X) for est in self.estimators_]).T
        y_pred = np.array(map(self._predict_one, y_pred_list))

        return y_pred

    def predict_proba(self, X):
        # TODO: have to fix how to calculate probability
        # For now, it just returns results of self.predict
        classes = self.classes_[:, np.newaxis]
        pred = (self.predict(X) == classes).T * 1
        return pred
