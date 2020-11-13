import contextlib
import os
from pathlib import Path
import numpy as np
from program import *
import mlb
from collections import defaultdict
from util import *
from typing import Any
from dataclasses import dataclass, field
from tqdm import tqdm
import inspect

from queue import PriorityQueue

class ProgramFail(Exception): pass

class FunctionInfo:
    stats = defaultdict(int)
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.description = fn.__doc__

        # None indicates we don't know
        self.has_axis = None
        self.has_keepdims = None
        self.argc = None

        try:
            self.params = inspect.signature(fn).parameters
        except ValueError:
            self.params = None
            print(f"can't find signature for {self.name}")
        
        # see what we can learn from inspect.signature()
        if self.params is not None:
            self.argc = 0
            self.has_axis = False
            self.has_keepdims = False
            for name,p in self.params.items():
                self.stats[name] += 1
                if name == 'axis':
                    self.has_axis = True
                if name == 'keepdims':
                    self.has_keepdims = True
                if p.kind == p.POSITIONAL_OR_KEYWORD:
                    if p.default is p.empty:
                        self.argc += 1 # required argument since it has no default
                else:
                    print(f"ignoring unusual param: {self.name}() has arg `{name}` of kind '{p.kind.description}'")

def pre_synth(fns,cfg):
    consts = [-1,0,1]
    vars = ['x','y']

    # load and normalize Primitive frequency data
    freq_path = Path('saved/freq.dict')
    freq = load(freq_path)
    tot = sum(freq.values())
    priors = {name: count/tot for name, count in freq.items()}
    priors['_var'] = .2
    priors['_const'] = .2

    # renormalize now that var/const are added and also convert to log
    tot = sum(priors.values())
    priors = {name: np.log(p/tot) for name, p in priors.items()}

    fns = [FunctionInfo(f) for f in fns]

    counts = [(name,count) for name,count in FunctionInfo.stats.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    print('arg name frequencies:')
    for name,count in counts:
        if count < 5:
            continue
        print(f'\t {name}: {count}')

    return priors,consts,vars,fns


def thunkify(fns,cfg):
    priors,consts,vars,fns = pre_synth(fns,cfg)


    steps = [FuncStep(f, priors[f.name]) for f in fns]
    steps += [ConstStep(x, priors['_const'], str(x)) for x in consts]
    steps += [VarStep(x, priors['_var'], str(x)) for x in vars]

    steps.sort(key=lambda step: step.prior)
    


    env = {
        'x':lambda:np.ones((2,3)),
        'y':lambda:np.array([[1,2,3],[4,5,6]]),
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
                stack_state=StackState(())
                )

    heap = PriorityQueue()
    heap.put(p.heapify())


    print = tqdm.write
    for _ in tqdm(range(2000), disable=True):
        base_p = heap.get().p
        if base_p.deleted:
            continue
        print(f'expanding {base_p}')
        ps = base_p.expand()
        if len(ps) == p.search_state.n:
            heap.put(base_p.heapify()) # return it to the heap bc there might be more expansions
        for p in ps:
            if len(p.p) < search_state.max_program_size:
                heap.put(p.heapify()) # only use as a potential parent program if its small enough
            print(f'\t{p}')
    
    results = [p for p in search_state.seen_stack_states.values() if len(p.stack_state) == 1]
    results.sort(key=lambda p: -p.ll)
    for res in reversed(results):
        print(str(res))


class SearchState:
    def __init__(self, steps:list, n:int, env:dict):
        self.steps = steps
        self.n = n
        self.env = env
        self.seen_stack_states = {} # {hashed_state->StackProgram}
        self.max_stack_size = 3
        self.max_program_size = 3


class StackProgram:
    def __init__(self,
                 p:tuple, # program: tuple of Steps
                 search_state:SearchState,
                 ll:float, # log likelihood of this program
                 stack_state# stack state after executing this program
                 ):
        self.p = p
        self.search_state = search_state
        self.ll = ll
        self.stack_state = stack_state
        self.steps = search_state.steps # list of Steps, shared globally
        self.idx = 0
        self.deleted = False
    @property
    def ll_upperbound(self):
        """
        upper bound on the ll of any child that could come from expand()
        """
        if self.idx >= len(self.steps):
            return -np.inf # there are no more children
        return self.ll + self.steps[self.idx].prior
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
                p = self.take_step(step)
            except ProgramFail:
                self.idx += 1
                continue
            res.append(p)
            self.idx += 1
            if len(res) >= n:
                break
        return res # a list of zero or more StackProgram
    def take_step(self, step):
        """ 
        Run the program consisting of `self` with `step` appended to it and return result
        May raise ProgramFail
        """
        stack_state = step(self.stack_state, self.search_state.env)
        # reject if stack has too many items (heuristic)
        if len(stack_state) > self.search_state.max_stack_size:
            raise ProgramFail("global max stacksize exceeded")
        # hash the state and add it to the list of seen states

        new_ll = self.ll + step.prior

        if stack_state in self.search_state.seen_stack_states:
            if self.search_state.seen_stack_states[stack_state].ll >= new_ll:
                # found this stack state previously and ll was higher
                raise ProgramFail("already seen this stack state")
            # found this stack state previously but ll was lower so we should delete the old one
            self.search_state.seen_stack_states[stack_state].deleted = True

        # we did it! This is a great new program and we can initialize it and return it!
        p = StackProgram(
            p=(*self.p,step), # append to tuple
            search_state=self.search_state,
            ll=new_ll,
            stack_state=stack_state
        )
        self.search_state.seen_stack_states[stack_state] = p
        return p
    def __repr__(self):
        stack_state = repr(self.stack_state).split('||')
        stack_state = mlb.mk_green('||').join([mlb.mk_blue(s) for s in stack_state])
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
        return HeapItem(prio=-self.ll_upperbound, p=self)


@dataclass(order=True)
class HeapItem:
    prio: float
    p: StackProgram = field(compare=False)


class StackState:
    """an immutable stack class. push() and pop() dont do inplace modifications
    """
    def __init__(self, stack: tuple):
        self.stack = tuple(stack)
    def push(self, item):
        stk = (*self.stack,item) # appending to a tuple
        return StackState(stk)
    def pop(self, n):
        """
        Return: (unpopped:StackState,popped:tuple)
        """
        if n == 0:
            return self, ()
        unpopped = self.stack[:-n] # things that arent popped
        popped = self.stack[-n:] # things that are popped
        return StackState(unpopped), popped
    def __len__(self):
        return len(self.stack)
    def __hash__(self):
        """
        - only req: for two objects if they are equal as defined by
           __eq__ then they must have the same __hash__ value
        - hash must be an int
        - methods like isinstance(tup,Hashable) fail in cases like tup=(3,[])
            so we have to do the try/except
        """
        def fallback(tup):
            try:
                return hash(str(tup))
            except TypeError: # the absurd case where the str() and repr() functions have bugs
                return hash(tuple(x.__class__ for x in tup))

        if any([isinstance(elem,np.dtype) for elem in self.stack]):
            # sometimes np.dtypes behave weird especially dtype([])
            return fallback(self.stack)

        try:
            return hash(self.stack)
        except TypeError:
            return fallback(self.stack)

    def __eq__(self, other):
        def eq(a,b):
            if type(a) != type(b):
                return False
            if isinstance(a,(list,tuple)):
                if len(a) != len(b):
                    return False
                return all([eq(x,y) for x,y in zip(a,b)])
            if isinstance(a,np.ndarray):
                return np.array_equal(a,b)
            try:
                return bool(a == b)
            except:
                return str(a) == str(b)
        return eq(self.stack,other.stack)
    def __repr__(self):
        res = []
        for item in self.stack:
            try:
                res.append(repr(item))
            except TypeError:
                res.append('[unable to print]')
        res = '(' + ' || '.join(res) + ')'

        res = res.replace('\n','').replace(' ','').replace('\t','')
        res = res.replace('||',' || ')
        return res


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
    def __init__(self, fn_info, prior):
        super().__init__(val=fn_info.fn, prior=prior, name=fn_info.name)
        self.argc = fn_info.argc
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        if self.argc is not None and len(stack_state) < self.argc:
            raise ProgramFail(f'argc mismatch: expected {self.argc} got {len(stack_state)}')
            
        # if argc is not known, try argc=0,1,2,... up to the number of things on the stack (inclusive)
        if self.argc is not None:
            argcs = [self.argc]
        else:
            argcs = list(range(0,len(stack_state)+1))

        with contextlib.redirect_stdout(os.devnull):
            with contextlib.redirect_stderr(os.devnull):

                for argc in argcs:
                    stack_state_prefix, args = stack_state.pop(argc)
                    try:
                        # apply function to the last `argc` things on the stack, leaving everything before that unchanged
                        #res = self.val(*stack_state[-argc:])
                        res = self.val(*args)
                        repr(res)
                    except:
                        continue # program crashed
                    # if we reach this point the program didnt crash!
                    return stack_state_prefix.push(res)
                # if we reach this point the program crashed on every attempt
                raise ProgramFail(f'crash during execution for all argcs attempted')
class ConstStep(Step):
    def __init__(self, val, prior, name):
        super().__init__(val,prior, name)
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        return stack_state.push(self.val)

class VarStep(Step):
    def __init__(self, val, prior, name):
        super().__init__(val,prior, name)
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        return stack_state.push(env[self.val]())





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