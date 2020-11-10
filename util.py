import pickle
import numpy as np
from collections.abc import Hashable


def make_hashable(tup):
    """
    methods like isinstance(tup,Hashable) fail in cases like tup=(3,[])
    so we do this instead
    """
    assert isinstance(tup,tuple)

    def as_str(tup):
        try:
            return str(tup)
        except TypeError: # the absurd case where the str() and repr() functions have bugs
            return f'IDX IN MEMORY:{id(tup)}'

    if any([isinstance(elem,np.dtype) for elem in tup]):
        # sometimes np.dtypes behave weird especially dtype([])
        return as_str(tup)

    try:
        hash(tup)
        return tup
    except TypeError:
        return as_str(tup)


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def normalize(lls):
    """
    dividing by the sum is subtracting the log of the sum when in logspace
    However we need to leave logspace to do the sum since logspace is only
    convenient for products.
    log( sum ( exp (lls)))
    """
    return lls - np.log(np.exp(lls).sum())
