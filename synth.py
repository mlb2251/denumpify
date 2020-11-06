from pathlib import Path
import numpy as np
from program import *
import mlb
from collections import defaultdict
from util import *

from queue import PriorityQueue


def pre_synth(fns,cfg):
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


    names = [f.__name__ for f in fns]

    return priors,names,consts,vars


def thunkify(fns,cfg):
    priors,names,consts,vars = pre_synth(fns,cfg)


    steps = [Step(f) for f in fns]
    steps += [Step(x,is_func=False) for x in consts]
    steps += [Step(x,is_func=False) for x in vars]
    

    found = []
    heap = PriorityQueue()

    p = StackProgram(
                p=[],
                steps=steps,
                ll=0,
                stack_state=[]
                )
    heap.push((p.ll,p))

    while True:
        pass

class StackProgram:
    max_argc = 2
    def __init__(self,
                 p:list[int], # program
                 steps:list, # list of callables, shared universally
                 ll:float,
                 stack_state:list
                 ):
        self.p = p
        self.ll = ll
        self.steps = steps
        self.stack_state = stack_state
        self.idx = 0
    def expand(self, n:int):
        res = []
        while self.idx < len(self.steps):
            step = self.steps[self.idx]
            try:
                stack_state = self.run(step)
            except StackFail:
                self.idx += 1
                continue
            p = StackProgram(
                p=self.p + [self.idx],
                steps = self.steps,
                ll=self.ll + step.prior,
                stack_state = stack_state
            )
            res.append(p)
            self.idx += 1
            if len(p) >= 5:
                break
        return p # a list of zero or more StackProgram
    def run(self, step):
        stack_state = step(self.stack_state)
        if len(stack_state) > self.max_argc:
            raise StackFail
        return stack_state

class Step:
    def __init__(self, val, prior, is_func=True, argc=1):
        self.val = val
        self.is_func = is_func
        self.argc = argc
        self.prior = prior
    def __call__(self, stack_state:list):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        if self.is_func:
            if len(stack_state) < self.argc:
                raise StackFail
            # apply function to the last `argc` things on the stack, leaving everything before that unchanged
            try:
                res = self.val(*stack_state[-self.argc:])
            except:
                raise StackFail
            return stack_state[:-self.argc] + [res]
        
        # if the step is not a function, simply append to the stack
        return stack_state + [self.val]

class StackFail(Exception): pass



def synth(fns, cfg):
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