import pickle
import numpy as np

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
