import pandas as pd
import numpy as np

from scipy.sparse.csr import csr_matrix
from typing import Callable


def params_to_numpy(*param_inds):
    """

    :param param_inds:
    :return:
    """

    def to_numpy(param):
        """

        :param param:
        :return:
        """
        if isinstance(param, list) or isinstance(param, tuple):
            param = np.array(param)
        elif isinstance(param, csr_matrix):
            param = param.toarray()
        elif isinstance(param, pd.DataFrame):
            param = param.values
        return param

    def decorator(func: Callable) -> Callable:
        """  """
        def wrapper(*args, **kwargs):
            """ """
            kwargs = list(kwargs.items())
            for ind in param_inds:
                if ind < len(args):
                    args = args[:ind] + (to_numpy(args[ind]), ) + args[ind+1:]
                elif ind < len(args) + len(kwargs):
                    ix = ind - len(args)
                    kwargs = kwargs[:ix] + [(kwargs[ix][0], to_numpy(kwargs[ix][1])), ] + kwargs[ix+1:]
                else:
                    raise IndexError('Params is out of range!')
            return func(*args, **dict(kwargs))
        return wrapper
    return decorator


if __name__ == '__main__':
    class A:
        @params_to_numpy(1)
        def func(self, x, y = (1, 2, 3), z = (4, 5, 6)):
            return x, y, z
    a = A()
    result = a.func(x=(5, 3, 1))
    print('-'*10)
    print(result)
