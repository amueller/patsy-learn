import numpy as np
HAS_PANDAS = True
try:
    import pandas as pd
except ImportError:
    HAS_PANDAS = False
import patsy
from sklearn.naive_bayes import GaussianNB  # for predict_proba
from sklearn.utils.mocking import CheckingClassifier
from sklearn.utils.testing import assert_raise_message, assert_equal
from numpy.testing import assert_array_equal

from patsylearn import PatsyModel, PatsyTransformer


def test_scope_model():
    data = patsy.demo_data("x1", "x2", "x3", "y")

    def myfunc(x):
        tmp = np.ones_like(x)
        tmp.fill(42)
        return tmp

    def check_X(X):
        return np.all(X[:, 1] == 42)

    # checking classifier raises error if check_X doesn't return true.
    # this checks that myfunc was actually applied
    est = PatsyModel(CheckingClassifier(check_X=check_X), "y ~ x1 + myfunc(x2)")
    est.fit(data)

    # test feature names
    assert_equal(est.feature_names_, ["x1", "myfunc(x2)"])


def test_scope_transformer():
    data = patsy.demo_data("x1", "x2", "x3", "y")

    def myfunc(x):
        tmp = np.ones_like(x)
        tmp.fill(42)
        return tmp

    est = PatsyTransformer("x1 + myfunc(x2)")
    est.fit(data)
    data_trans = est.transform(data)
    assert_array_equal(data_trans[:, 1], 42)

    est = PatsyTransformer("x1 + myfunc(x2)")
    data_trans = est.fit_transform(data)
    assert_array_equal(data_trans[:, 1], 42)

    # test feature names
    assert_equal(est.feature_names_, ["x1", "myfunc(x2)"])


def test_error_on_y_transform():
    data = patsy.demo_data("x1", "x2", "x3", "y")
    est = PatsyTransformer("y ~ x1 + x2")
    msg = ("encountered outcome variables for a model"
           " that does not expect them")
    assert_raise_message(patsy.PatsyError, msg, est.fit, data)
    assert_raise_message(patsy.PatsyError, msg, est.fit_transform, data)


def test_intercept_model():
    data = patsy.demo_data("x1", "x2", "x3", "y")

    def check_X_no_intercept(X):
        return X.shape[1] == 2

    # check wether X contains only the two features, no intercept
    est = PatsyModel(CheckingClassifier(check_X=check_X_no_intercept),
                     "y ~ x1 + x2")
    est.fit(data)
    # predict checks applying to new data
    est.predict(data)

    def check_X_intercept(X):
        shape_correct = X.shape[1] == 3
        first_is_intercept = np.all(X[:, 0] == 1)
        return shape_correct and first_is_intercept

    # check wether X does contain intercept
    est = PatsyModel(CheckingClassifier(check_X=check_X_intercept),
                     "y ~ x1 + x2", add_intercept=True)
    est.fit(data)
    est.predict(data)


def test_intercept_transformer():
    data = patsy.demo_data("x1", "x2", "x3", "y")

    # check wether X contains only the two features, no intercept
    est = PatsyTransformer("x1 + x2")
    est.fit(data)
    assert_equal(est.transform(data).shape[1], 2)

    # check wether X does contain intercept
    est = PatsyTransformer("x1 + x2", add_intercept=True)
    est.fit(data)
    data_transformed = est.transform(data)
    assert_array_equal(data_transformed[:, 0], 1)
    assert_equal(est.transform(data).shape[1], 3)


def test_stateful_transform():
    data_train = patsy.demo_data("x1", "x2", "y")
    data_train['x1'][:] = 1
    # mean of x1 is 1
    data_test = patsy.demo_data("x1", "x2", "y")
    data_test['x1'][:] = 0

    # center x1
    est = PatsyTransformer("center(x1) + x2")
    est.fit(data_train)
    data_trans = est.transform(data_test)
    # make sure that mean of training, not test data was removed
    assert_array_equal(data_trans[:, 0], -1)


def test_stateful_transform_dataframe():
    if not HAS_PANDAS:
        return
    data_train = pd.DataFrame(patsy.demo_data("x1", "x2", "y"))
    data_train['x1'][:] = 1
    # mean of x1 is 1
    data_test = pd.DataFrame(patsy.demo_data("x1", "x2", "y"))
    data_test['x1'][:] = 0

    # center x1
    est = PatsyTransformer("center(x1) + x2", return_type='dataframe')
    est.fit(data_train)
    data_trans = est.transform(data_test)

    # make sure result is pandas dataframe
    assert type(data_trans) is pd.DataFrame

    # make sure that mean of training, not test data was removed
    assert_array_equal(data_trans['center(x1)'][:],-1)


def test_stateful_model():
    data_train = patsy.demo_data("x1", "x2", "y")
    data_train['x1'][:] = 1
    # mean of x1 is 1
    data_test = patsy.demo_data("x1", "x2", "y")
    data_test['x1'][:] = 0

    # center x1
    est = PatsyModel(CheckingClassifier(), "y ~ center(x1) + x2")
    est.fit(data_train)

    def check_centering(X):
        return np.all(X[:, 0] == -1)

    est.estimator_.check_X = check_centering
    # make sure that mean of training, not test data was removed
    est.predict(data_test)


def test_predict_probas():
    data = patsy.demo_data("x1", "x2", "y", nlevels=2)
    data['y'] = np.asarray(data['y'], dtype=int)
    data_test = patsy.demo_data("x1", "x2", "y")
    data_test['y'] = np.asarray(data_test['y'], dtype=int)

    est = PatsyModel(GaussianNB(), "y ~ x1 + x2")
    est.fit(data)
    est.predict(data_test)
    est.predict_proba(data_test)
    est.predict_log_proba(data_test)

    if HAS_PANDAS:
        # Make sure that the returned column names are correct
        est = PatsyModel(GaussianNB(), "y ~ x1 + x2", return_type='dataframe')
        est.fit(data)
        est.predict(data_test)
        proba = est.predict_proba(data_test)
        assert isinstance(proba, pd.DataFrame)
        assert all([str(c) in proba.columns[i] for i, c in
                    enumerate(est.estimator_.classes_)])
        est.predict_log_proba(data_test)
