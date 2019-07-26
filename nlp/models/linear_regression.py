import numpy as np

from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression


# static model hyper parameters
PARAMS = {
    'random_state'      : 2523252,
    'dual'              : False,
    'tol'               : 1e-2,
    'fit_intercept'     : True,
    'intercept_scaling' : 1,
    'max_iter'          : 500,
    'class_weight'      : 'balanced',
    'multi_class'       : 'ovr',
    'n_jobs'            : -1,
    'warm_start'        : False,
    'verbose'           : 0,
}

# distributions of model parameters
SPACE = {
    'penalty':
        hp.choice('penalty_choice', ['l1', 'l2']),
    'C':
        hp.choice('C_choice', [
            scope.float(hp.qloguniform('lin_C_minor', 0, np.log(1e-4), 1e-3)),
            scope.float(hp.qloguniform('lin_C_major', 0, np.log(10 ** 4), 10)),
        ]),
    }


class LinearModel(LogisticRegression):
    def fit(self, *args, categorical_features_inds = None):
        """ Just to skip AttributeException and make one train_test function to lgb and logReg as well """
        return super().fit(*args)
