from pathlib import Path
import numpy as np
from program import *
import mlb
from collections import defaultdict
from util import *


def synth(fns, cfg):


    consts = [-1,0,1]
    vars = ['x','y']

    # load and normalize Primitive frequency data
    freq_path = Path('saved/freq.dict')
    freq = load(freq_path)
    tot = sum(freq.values())
    priors = {name: count/tot for name, count in freq.items()}
    priors['_var'] = 1
    priors['_const'] = 1

    # renormalize now that var/const are added and also convert to log
    tot = sum(priors.values())
    priors = {name: np.log(p/tot) for name, p in priors.items()}



    # set up the prims

    names = [f.__name__ for f in fns]
    prims = []
    for name,fn in zip(names,fns):
        prims.append(PrimFunc(name,fn,TFunc(1), prior=priors[name]))

    prims += [PrimConst(str(i), i, TVal(), prior=priors['_var']) for i in range(-1,2)]
    prims += [PrimVar(name, name, TVal(), prior=priors['_const']) for name in ('x','y')]

    funcs = {p for p in prims if p.is_func}
    exprs = {p() for p in prims if not p.is_func}
    seen = set()

    env = {
        'x':np.ones((2,3)),
        'y':0,
    }

    print("bottom upping")
    # bottom up
    for _ in range(2):
        to_add = set()
        for f in funcs:
            for e in exprs:
                new = f(e)
                if str(new) in seen:
                    continue
                seen.add(str(new))
                res = new.eval(env)
                if isinstance(res,ProgramException):
                    mlb.red(f'{new} -> error: {res}')
                    continue
                try:
                    s = str(res)
                except TypeError:
                    s = "[unable to print result]"
                    mlb.red(f'{new} -> error: {e}')
                    continue

                mlb.green(f'{e} -> {s}')

                to_add.add(new)
                print(len(to_add))
        exprs |= to_add
    
    by_depth = defaultdict(list)
    for e in exprs:
        by_depth[e.depth].append(e)

    print({k:len(v) for k,v in by_depth.items()})

    return