import hyperopt

import numpy as np
import pandas as pd

from hyperopt import Trials, fmin, tpe
from typing import Iterable
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def cross_val(x_train, y_train, estimator, constant_params, cat_features: Iterable[int], params: dict, folds = 5) -> float:
    """
    Train and test Categorical Boosting Machine with hyper parameters
    :param x_train: features of the train set
    :param y_train: target column of the train set
    :param estimator:
    :param constant_params:
    :param cat_features: indices of categorical features in the x_train set
    :param params: hyper parameters for CatBoosting Machine
    :param folds: number of folds for the cross validation
    """

    kf = KFold(n_splits=folds, shuffle=True, random_state=435)
    metric_results = []
    for train_index, val_index in kf.split(x_train):
        model = estimator(**constant_params, **params)
        model.fit(
            x_train.iloc[train_index],
            y_train.iloc[train_index],
            categorical_features_inds = cat_features,
        )
        metric_results.append(accuracy_score(
            y_true = y_train.iloc[val_index],
            y_pred = model.predict(x_train.iloc[val_index])
        ))
    return -np.mean(metric_results)


def optimize_params(x_train, y_train, estimator, constant_params, space, cat_features: Iterable[int], folds: int = 5, evals: int = 10) -> dict:
    """
    Optimize parameters for the model
    :param x_train: features for the train
    :param y_train: target column
    :param estimator:
    :param constant_params:
    :param space: target column
    :param cat_features: categorical features
    :param folds: number of the folds for the cross validation
    :param evals:
    :return:
    """

    trials = Trials()
    best = fmin(
        fn=lambda params: cross_val(
            x_train         = x_train,
            y_train         = y_train,
            estimator       = estimator,
            constant_params = constant_params,
            cat_features    = cat_features,
            folds           = folds,
            params          = params,
        ),
        space     = space,
        algo      = tpe.suggest,
        max_evals = evals,
        trials    = trials,
        rstate    = np.random.RandomState(1234),
    )
    return hyperopt.space_eval(space=space, hp_assignment=best)


def train_test(x_train, y_train, estimator, constant_params, space, cat_features: Iterable[str] = (), evals: int = 10):
    """
    Train the Model
    :param x_train:
    :param y_train:
    :param estimator:
    :param constant_params:
    :param space:
    :param cat_features:
    :param evals:
    :return: model
    """

    # indices of categorical features
    if not cat_features:
        cat_features_inds = [list(x_train.columns).index(item) for item in cat_features]
    else:
        cat_features_inds = 'auto'

    best_params = optimize_params(
        x_train         = pd.DataFrame(x_train),
        y_train         = pd.DataFrame(y_train),
        estimator       = estimator,
        constant_params = constant_params,
        space           = space,
        cat_features    = cat_features_inds,
        folds           = 5,
        evals           = evals,
    )

    return estimator(**best_params)
