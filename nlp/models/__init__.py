from models.lgbmachine import LGBMachine, SPACE as lgb_space, PARAMS as lgb_params
from models.linear_regression import LinearModel, SPACE as lin_space, PARAMS as lin_params
from models.train_test import train_test
from models.blending import BlendingFeatures

from typing import Iterable


def get_lgb(x_train, y_train, cat_features: Iterable[str] = ()):
    """

    :param x_train:
    :param y_train:
    :param cat_features:
    :return:
    """
    return train_test(
        x_train         = x_train,
        y_train         = y_train,
        estimator       = LGBMachine,
        constant_params = lgb_params,
        space           = lgb_space,
        cat_features    = cat_features,
    )


def get_linear(x_train, y_train):
    """

    :param x_train:
    :param y_train:
    :return:
    """
    return train_test(
        x_train         = x_train,
        y_train         = y_train,
        estimator       = LinearModel,
        constant_params = lin_params,
        space           = lin_space,
    )


__all__ = [
    'get_lgb',
    'get_linear',
    'BlendingFeatures',
]
