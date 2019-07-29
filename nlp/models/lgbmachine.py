import joblib
import lightgbm

import pandas as pd
import numpy as np

from typing import Iterable
from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.model_selection import ShuffleSplit
from sklearn.base import BaseEstimator

from models.utils import params_to_numpy


# static model hyper parameters
PARAMS = {
    'seed'                  : 2523252,
    'tree_learner'          : 'serial',
    'device_type'           : 'cpu',
    'is_unbalance'          : False,
    'early_stopping_rounds' : 200,
    'metric'                : 'binary_logloss',
    'eval_metric'           : 'binary_error',
    'n_jobs'                : -1,
}

# distributions of model parameters
SPACE = {
    'num_iterations':
        scope.int(hp.qloguniform('dtree_num_iteration', np.log(50), np.log(500), 1)),
    'num_leaves':
        scope.int(hp.qloguniform('dtree_num_leaves', np.log(20), np.log(100), 1)),
    'learning_rate':
        scope.float(hp.qloguniform('dtree_learning_rate', np.log(1e-3), np.log(1e-1), 1e-4)),
    'max_depth':
        scope.int(hp.qloguniform('dtree_max_depth', np.log(1), np.log(100), 1)),
    'min_data_in_leaf':
        scope.int(hp.qloguniform('dtree_min_data_in_leaf', np.log(10), np.log(100), 1)),
    'max_drop':
        scope.int(hp.qloguniform('dtree_max_drop', np.log(10), np.log(100), 1)),
    'bagging_fraction':
        scope.float(hp.qloguniform('dtree_bagging_fraction', np.log(0.6), np.log(0.9), 0.05)),
    'feature_fraction':
        scope.float(hp.qloguniform('dtree_feature_fraction', np.log(0.6), np.log(0.9), 0.05)),
}


def load_model(model_path: str):
    """
    Load the model
    :param model_path: path to the model `type: Crawler` to load
    :return: Model()
    """
    model = LGBMachine()
    model = model.load(model_path=model_path)
    return model


# -------------- all code for the models can be found in jupyters notebook --------------
class LGBMachine(BaseEstimator):
    """
    Model that measures how close is the url to the company.
    Light Boosting Machine inside
    """

    def __init__(self, **params):
        self._params = params

        self._model = lightgbm.LGBMClassifier(**self.params)
        self._features = None

        # variable for the categorical features
        self._cat_features = None

    @params_to_numpy(1, 2)
    def fit(self, x_train, y_train, categorical_features_inds: Iterable = ()):
        """
        Fit the model
        :param x_train: x sample for training
        :type x_train: pd.DataFrame
        :param y_train: target sample for training
        :param categorical_features_inds: indices of categorical features in the x_train set
        """

        self._cat_features = categorical_features_inds
        if hasattr(x_train, 'columns'):
            self._features = x_train.columns
        else:
            self._features = list(range(len(x_train[0])))

        train_inds, val_inds = next(ShuffleSplit(
            n_splits=1, test_size=0.2, random_state=622333,
        ).split(x_train, groups=list(range(x_train.shape[0]))))

        self.model.fit(X=x_train[train_inds], y=y_train[train_inds],
                       categorical_feature=self.cat_features,
                       eval_set=[(x_train[val_inds], y_train[val_inds])],
                       verbose=False,
                       )

    def predict(self, x_test):
        """ Returns the decision """
        return self.model.predict(x_test, categorical_feature=self.cat_features)

    def predict_proba(self, x_test):
        """ Returns the probability """
        return self.model.predict_proba(x_test, categorical_feature=self.cat_features)[:, 1]

    @property
    def model(self):
        """ Returns internal model """
        return self._model

    @property
    def params(self):
        """ Hyper parameters of the model that have given the best (metric) score while training """
        return self._params

    def get_params(self, deep: bool = True):
        """ Returns parameters of the model """
        return self.params

    def set_params(self, **params):
        """
        Set params to the model
        :param params: hyper parameters
        """
        self.model.set_params(**params)

    @property
    def features(self):
        """ Features of train data """
        return self._features

    @property
    def cat_features(self):
        """ Categorical features of train data """
        return self._cat_features

    @property
    def feature_importances_(self):
        """ Return the feature importances (the higher, the more important the feature) """
        f_imp = pd.DataFrame(
            self.model.feature_importances_, index=self.features, columns=['importance']
        ).sort_values('importance', ascending=False)
        return f_imp

    def save(self, model_path: str):
        """ Save model """
        dict_ = {
            '_model'       : self.model,
            '_features'    : self.features,
            '_cat_features': self.cat_features,

        }
        joblib.dump(dict_, filename=model_path, protocol=4)

    @staticmethod
    def load(model_path: str):
        """ Load model """
        model = LGBMachine()
        dict_ = joblib.load(model_path)
        for key, value in dict_.items():
            setattr(model, key, value)
        return model
