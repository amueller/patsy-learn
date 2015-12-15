import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import column_or_1d
from patsy import dmatrix, dmatrices, EvalEnvironment, ModelDesc, INTERCEPT


# TODO: fit_transform fit_predicdt in PatsyModel?
# TODO: Allow pandas dataframe output in Transformer?
# TODO: unsupervised models? missing output variable?

def _drop_intercept(formula, add_intercept):
    """Drop the intercept from formula if not add_intercept"""
    if not add_intercept:
        if not isinstance(formula, ModelDesc):
            formula = ModelDesc.from_formula(formula)
        if INTERCEPT in formula.rhs_termlist:
            formula.rhs_termlist.remove(INTERCEPT)
        return formula
    return formula


class PatsyModel(BaseEstimator):
    """Meta-estimator for patsy-formulas.

    This model is a meta-estimator that takes a patsy-formula and a
    scikit-learn estimator model.
    The input data, a pandas dataframe (or dict-like) is transformed
    according to the formula before it is passed on to the estimator.

    This is currently for supervised models only, as it assumes a label column,
    or left-hand-side in the formula.
    For unsupervised models, simply build a pipeline with PatsyTransformer.

    Parameters
    ----------
    estimator : object
        Scikit-learn estimator.

    formula : string or formula-like
        Pasty formula used to transform the data.

    add_intercept : boolean, default=False
        Wether to add an intersept. By default scikit-learn has built-in
        intercepts for all models, so we don't add an intercept to the data,
        even if one is specified in the formula.

    eval_env : environment or int, default=0
        Envirionment in which to evalute the formula.
        Defaults to the scope in which PatsyModel was instantiated.

    NA_action : string or NAAction, default="drop"
        What to do with rows that contain missing values. You can ``"drop"``
        them, ``"raise"`` an error, or for customization, pass an `NAAction`
        object.  See ``patsy.NAAction`` for details on what values count as
        'missing' (and how to alter this).

    return_type : string, default="ndarray"
        data type that transform method will return. Default is ``"ndarray"``
        for numpy array, but if you would like to get Pandas dataframe (for
        example for using it in scikit transformers with dataframe as input
        use ``"dataframe"``

    Note
    ----
    PastyModel does by default not add an intercept, even if you
    specified it in the formula. You need to set add_intercept=True.

    """
    def __init__(self, estimator, formula, add_intercept=False, eval_env=0,
                 NA_action="drop", return_type='ndarray'):
        self.estimator = estimator
        self.formula = formula
        self.eval_env = eval_env
        self.add_intercept = add_intercept
        self.NA_action = NA_action
        self.return_type = return_type

    def fit(self, data, y=None):
        """Fit the scikit-learn model using the formula.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Contains features and possible labels.
            Column names need to match variables in formula.
        """
        eval_env = EvalEnvironment.capture(self.eval_env, reference=1)
        formula = _drop_intercept(self.formula, self.add_intercept)
        design_y, design_X = dmatrices(formula, data, eval_env=eval_env,
                                       NA_action=self.NA_action)
        self.design_y_ = design_y.design_info
        self.design_X_ = design_X.design_info
        # convert to 1d vector so we don't get a warning
        # from sklearn.
        design_y = column_or_1d(design_y)
        est = clone(self.estimator)
        self.estimator_ = est.fit(design_X, design_y)
        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, data):
        """Predict with estimator using formula.

        Transform the data using formula, then predict on it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        X = np.array(dmatrix(self.design_X_, data))
        return self.estimator_.predict(X)

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, data):
        """Compute predict_proba with estimator using formula.

        Transform the data using formula, then predict probabilities on it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        X = np.array(dmatrix(self.design_X_, data))
        return self.estimator_.predict_proba(X)

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, data):
        """Compute decision_function of estimator using formula.

        Transform the data using formula, then predict on it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        X = np.array(dmatrix(self.design_X_, data))
        return self.estimator_.decision_function(X)

    @if_delegate_has_method(delegate='estimator')
    def transform(self, data):
        """Transform with estimator using formula.

        Transform the data using formula, then transform it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """

        if self.return_type == 'dataframe':
            X = dmatrix(self.design_X_, data, return_type='dataframe')
        else:
            X = np.array(dmatrix(self.design_X_, data))

        return self.estimator_.transform(X)

    @if_delegate_has_method(delegate='estimator')
    def score(self, data):
        """Predict with estimator using formula.

        Transform the data using formula, then predict on it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
            Data needs to contain the label column.
        """
        design_infos = (self.design_y_, self.design_X_)
        design_y, design_X = dmatrices(design_infos, data)
        return self.estimator_.score(design_X, design_y)


class PatsyTransformer(BaseEstimator, TransformerMixin):
    """Transformer using patsy-formulas.

    PatsyTransformer transforms a pandas DataFrame (or dict-like)
    according to the formula and produces a numpy array.

    Parameters
    ----------
    formula : string or formula-like
        Pasty formula used to transform the data.

    add_intercept : boolean, default=False
        Wether to add an intersept. By default scikit-learn has built-in
        intercepts for all models, so we don't add an intercept to the data,
        even if one is specified in the formula.

    eval_env : environment or int, default=0
        Envirionment in which to evalute the formula.
        Defaults to the scope in which PatsyModel was instantiated.

    NA_action : string or NAAction, default="drop"
         What to do with rows that contain missing values. You can ``"drop"``
         them, ``"raise"`` an error, or for customization, pass an `NAAction`
         object.  See ``patsy.NAAction`` for details on what values count as
         'missing' (and how to alter this).

    return_type : string, default="ndarray"
        data type that transform method will return. Default is ``"ndarray"``
        for numpy array, but if you would like to get Pandas dataframe (for
        example for using it in scikit transformers with dataframe as input
        use ``"dataframe"``

    Note
    ----
    PastyTransformer does by default not add an intercept, even if you
    specified it in the formula. You need to set add_intercept=True.

    As scikit-learn transformers can not ouput y, the formula
    should not contain a left hand side.  If you need to transform both
    features and targets, use PatsyModel.
    """
    def __init__(self, formula, add_intercept=False, eval_env=0, NA_action="drop",
                 return_type='ndarray'):
        self.formula = formula
        self.eval_env = eval_env
        self.add_intercept = add_intercept
        self.NA_action = NA_action
        self.return_type = return_type

    def fit(self, data, y=None):
        """Fit the scikit-learn model using the formula.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        self._fit_transform(data, y)
        return self

    def fit_transform(self, data,  y=None):
        """Fit the scikit-learn model using the formula and transform it.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.

        Returns
        -------
        X_transform : ndarray
            Transformed data
        """
        return self._fit_transform(data, y)

    def _fit_transform(self, data, y=None):
        eval_env = EvalEnvironment.capture(self.eval_env, reference=2)
        formula = _drop_intercept(self.formula, self.add_intercept)

        design = dmatrix(formula, data, eval_env=eval_env, NA_action=self.NA_action,
                         return_type='dataframe')
        self.design_ = design.design_info

        if self.return_type == 'dataframe':
            return design
        else:
            return np.array(design)

    def transform(self, data):
        """Transform with estimator using formula.

        Transform the data using formula, then transform it
        using the estimator.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Column names need to match variables in formula.
        """
        if self.return_type == 'dataframe':
            return dmatrix(self.design_, data, return_type='dataframe')
        else:
            return np.array(dmatrix(self.design_, data))
