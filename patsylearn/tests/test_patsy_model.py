import numpy as np
HAS_PANDAS = True
try:
    import pandas as pd
except ImportError:
    HAS_PANDAS = False
import patsy
from sklearn.ensemble import RandomForestClassifier  # for predict_proba
from sklearn.svm import SVC  # for decision_function
from sklearn.utils.mocking import CheckingClassifier
from sklearn.utils.testing import assert_raise_message, assert_equal, SkipTest
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
       raise SkipTest("Skipping because pandas is not installed")

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


def test_return_types():
    if not HAS_PANDAS:
        raise SkipTest("Skipping because pandas is not installed")

    # Make sure if return_type is dataframe, actually return a dataframe
    data = patsy.demo_data("x1", "x2", "a", nlevels=3)
    data['a'] = np.asarray([c.strip('a') for c in data['a']], dtype=int)
    data_test = patsy.demo_data("x1", "x2", "a", nlevels=3)
    data_test['a'] = np.asarray([c.strip('a') for c in data_test['a']],
                                dtype=int)

    model = 'a ~ x1 + x2'

    # For classifier with predict_proba
    est = PatsyModel(RandomForestClassifier(), model,
                     return_type='dataframe')
    est.fit(data)

    proba = est.predict_proba(data_test)
    assert isinstance(proba, pd.DataFrame)
    # Make sure column names include class names
    assert all([str(c) in proba.columns[i] for i, c in
                enumerate(est.estimator_.classes_)])

    est.predict_log_proba(data_test)

    # For classifier with decision_function
    est = PatsyModel(SVC(), model, return_type='dataframe')
    est.fit(data)

    decision = est.decision_function(data_test)

    est = PatsyModel(SVC(decision_function_shape='ovr'), model,
                     return_type='dataframe')
    est.fit(data)

    decision = est.decision_function(data_test)
    assert all([str(c) in decision.columns[i] for i, c in
                enumerate(est.estimator_.classes_)])


def test_dataframe_in_pipeline():
    if not HAS_PANDAS:
        raise SkipTest("Skipping because pandas is not installed")

    # Example usage from Pipeline class
    from sklearn.datasets import samples_generator
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # generate some data to play with
    X, y = samples_generator.make_classification(
        n_informative=5, n_redundant=0, random_state=42)
    X_cols = ['c%s' % i for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=X_cols)
    df['y'] = y

    model = 'y ~ ' + ' + '.join(X_cols)

#    anova_filter = PatsyModel(SelectKBest(f_regression, k=5), model,
#                              return_type='dataframe')
    clf = PatsyModel(SVC(kernel='linear'), model, return_type='dataframe')
    scaler = PatsyModel(StandardScaler(), model, return_type='dataframe')
    pipe = Pipeline([('scale', scaler), ('svc', clf)])

    pipe.set_params(svc__C=.1).fit(df, df['y'])
    prediction = pipe.predict(df)
    pipe.score(df)
