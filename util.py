import pickle
import numpy as np
from collections.abc import Hashable

def safe_eq(a,b):
    """
    a safe version of the "==" operator resilient to the countless numpy bugs and crashes
    """
    if type(a) != type(b):
        return False
    if isinstance(a,(list,tuple)):
        if len(a) != len(b):
            return False
        return all([safe_eq(x,y) for x,y in zip(a,b)])
    try:
        if isinstance(a,np.ndarray):
            return np.array_equal(a,b)
        return bool(a == b)
    except:
        return str(a) == str(b)

def safe_hash(obj):
    """
    - only req: for two objects if they are equal as defined by
        __eq__ then they must have the same __hash__ value
    - hash must be an int
    - methods like isinstance(tup,Hashable) fail in cases like tup=(3,[])
        so we have to do the try/except
    """
    def fallback():
        try:
            return hash(str(obj))
        except TypeError: # the absurd case where the str() and repr() functions have bugs
            raise NotImplementedError # probably we should never reach this point bc this means our value is not even interesting in the first place

    if isinstance(obj,np.dtype):
        # sometimes np.dtypes behave weird especially dtype([])
        # so here we dont even wanna attempt hash() I think
        return fallback()

    try:
        return hash(obj)
    except TypeError:
        return fallback()

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
