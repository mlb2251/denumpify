from pathlib import Path
import numpy as np
from program import *
import mlb
from collections import defaultdict
from util import *
from typing import Any
from dataclasses import dataclass, field

from queue import PriorityQueue

class ProgramFail(Exception): pass


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


    steps = [FuncStep(f, priors[f.__name__], f.__name__) for f in fns]
    steps += [ConstStep(x, priors['_const'], str(x)) for x in consts]
    steps += [VarStep(x, priors['_var'], str(x)) for x in vars]

    steps.sort(key=lambda step: step.prior)
    


    env = {
        'x':np.ones((2,3)),
        'y':0,
    }
    search_state = SearchState(
                            steps,
                            n=3,
                            env=env,
                            )

    p = StackProgram(
                p=(),
                search_state=search_state,
                ll=0,
                stack_state=(),
                )

    heap = PriorityQueue()
    heap.put(p.heapify())
    found = [p]


    for _ in range(100):
        base_p = heap.get().p
        print(f'expanding {base_p}')
        ps = base_p.expand()
        if len(ps) == p.search_state.n:
            heap.put(base_p.heapify()) # return it to the heap bc there might be more expansions
        for p in ps:
            found.append(p)
            heap.put(p.heapify())
            print(f'\t{p}')

class SearchState:
    def __init__(self, steps:list, n:int, env:dict):
        self.steps = steps
        self.n = n
        self.env = env
        self.seen_stack_states = set()
        self.max_argc = 2


class StackProgram:
    def __init__(self,
                 p:tuple, # program: tuple of Steps
                 search_state:SearchState,
                 ll:float, # log likelihood of this program
                 stack_state:tuple # stack state after executing this program
                 ):
        self.p = p
        self.search_state = search_state
        self.ll = ll
        self.stack_state = stack_state
        self.steps = search_state.steps # list of Steps, shared globally
        self.idx = 0
    def expand(self,n=None):
        """
        Get the next `n` (or fewer) programs obtained by appending a single Step to this program.
        First tries the program obtained by taking `self` and appending self.steps[self.idx]
        Continues trying programs and incrementing self.idx until `n` programs that don't crash are found.
        Next time this function is called it will pick up where it self off at self.idx
        May return fewer than `n` programs if it tries all steps and fewer than `n` return non-crashing programs
        """
        if n is None:
            n = self.search_state.n
        res = []
        while self.idx < len(self.steps):
            step = self.steps[self.idx]
            try:
                stack_state = self.run(step)
            except ProgramFail:
                self.idx += 1
                continue
            p = StackProgram(
                p=(*self.p,step), # append to tuple
                search_state=self.search_state,
                ll=self.ll + step.prior,
                stack_state=stack_state
            )
            res.append(p)
            self.idx += 1
            if len(res) >= n:
                break
        return res # a list of zero or more StackProgram
    def run(self, step):
        """ 
        Run the program consisting of `self` with `step` appended to it and return result
        May raise ProgramFail
        """
        stack_state = step(self.stack_state, self.search_state.env)
        if len(stack_state) > self.search_state.max_argc:
            raise ProgramFail("global max argc exceeded")
        hashable_state = str(stack_state) if not hashable(stack_state) else stack_state
        if hashable_state in self.search_state.seen_stack_states:
            raise ProgramFail("already seen this stack state")
        self.search_state.seen_stack_states.add(hashable_state)
        return stack_state
    def __repr__(self):
        stack_state = repr(self.stack_state).replace('\n','').replace(' ','').replace('\t','')
        stack_state = mlb.mk_blue(stack_state)
        return f'{self.p} -> {stack_state} (ll={self.ll:.2f})'
    def heapify(self):
        """
        Returns an object that can be pushed to a heap.
        Note that pushing a (prio,Program) tuple directly would raise an exception
            in the instance where two programs have equal priorities
        Note that implementing __lt__ and other methods directly on Program would run
            into issues where things look messier - we only want this ordering to apply
            to heap operations not any other comparisons, and in particular it looks strange
            to make operations like "<" based on self.ll while making operations like "=="
            presumably based on self.p. It just all gets messy so this dataclass makes more sense
        """
        return HeapItem(prio=-self.ll, p=self)


@dataclass(order=True)
class HeapItem:
    prio: int
    p: StackProgram = field(compare=False)

class Step:
    def __init__(self, val, prior, name):
        self.val = val
        self.prior = prior
        self.name = name
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        raise NotImplementedError
    def __repr__(self):
        return self.name

class FuncStep(Step):
    def __init__(self, val, prior, name, argc=1):
        super().__init__(val,prior,name)
        self.argc = argc
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        if len(stack_state) < self.argc:
            raise ProgramFail(f'argc mismatch: expected {self.argc} got {len(stack_state)}')
        # apply function to the last `argc` things on the stack, leaving everything before that unchanged
        try:
            res = self.val(*stack_state[-self.argc:])
        except:
            raise ProgramFail(f'crash during execution')
        return (*stack_state[:-self.argc], res)

class ConstStep(Step):
    def __init__(self, val, prior, name):
        super().__init__(val,prior, name)
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        return (*stack_state, self.val)

class VarStep(Step):
    def __init__(self, val, prior, name):
        super().__init__(val,prior, name)
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        return (*stack_state, env[self.val])





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