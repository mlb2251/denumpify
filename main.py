import mlb
import numpy as np
from collections import defaultdict


def main():
    all_fns = [getattr(np, x) for x in dir(np) if callable(getattr(np, x))]
    by_types = defaultdict(list)
    for f in all_fns:
        by_types[type(f)].append(f)

    del by_types[type]  # things like np.bool, etc
    # theres just one instance of this
    del by_types[np._pytesttester.PytestTester]
    # ufuncs are stuff like np.log() np.exp() etc. Short for "universal function". They act elemwise. Not super relevant to us
    ufuncs = by_types[np.ufunc]
    # cherrypicked from the 20 builtin functions. The rest are things like fromiter(), seterrorobj(), and other things we dont like
    builtins = [np.arange, np.array, np.empty, np.zeros]
    # these are the bulk of the fns we actually care about
    fns = by_types[type(lambda:0)]  # get the <class 'function'> type

    fns = [f for f in fns if not f.__name__.startswith('_')]
    print([f.__name__ for f in fns])
    print(len(fns))
    for f in fns:
        p = Program(f, f.__name__)
    return


if __name__ == '__main__':
    with mlb.debug(False):
        main()
