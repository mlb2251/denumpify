from compiletime import *
from old_program import *
import mlb

def simple_bottom_up(fns, cfg):
    priors,names,consts,vars = pre_synth(fns,cfg)

    prims = []
    for name,fn in zip(names,fns):
        prims.append(PrimFunc(name,fn,TFunc(1), prior=priors[name]))

    prims += [PrimConst(str(i), i, TVal(), prior=priors['_const']) for i in consts]
    prims += [PrimVar(name, name, TVal(), prior=priors['_var']) for name in vars]

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