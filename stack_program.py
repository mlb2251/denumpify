import numpy as np
import mlb
from dataclasses import dataclass, field
import os
import contextlib
from util import *
from copy import deepcopy



class ProgramFail(Exception): pass

class SearchState:
    def __init__(self, steps:list, n:int, env:dict):
        self.steps = steps
        self.step_from_name = {step.name:step for step in steps}
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
    def take_step(self, step, error_if_seen=True, update_if_new=True):
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

        update = update_if_new
        if stack_state in self.search_state.seen_stack_states:
            if self.search_state.seen_stack_states[stack_state].ll >= new_ll:
                # found this stack state previously and ll was higher
                update = False
                if error_if_seen:
                    raise ProgramFail("already seen this stack state")

        # we did it! This is a great new program and we can initialize it and return it!
        p = StackProgram(
            p=(*self.p,step), # append to tuple
            search_state=self.search_state,
            ll=new_ll,
            stack_state=stack_state
        )
        # update seen_stack_states
        if update:
            if stack_state in self.search_state.seen_stack_states:
                # found this stack state previously but ll was lower so we should delete the old one
                self.search_state.seen_stack_states[stack_state].deleted = True
            self.search_state.seen_stack_states[stack_state] = p
        return p
    def __repr__(self):
        return f'{self.p} -> {self.stack_state} (ll={self.ll:.2f})'
    def pretty(self):
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
    @staticmethod
    def parse(s:str, search_state: SearchState):
        """
        Raises:
            ParseError
            ProgramFail
        """
        # canonicalize and split into step names
        s = canonicalize(s)
        step_names = s[1:-1].split(', ')
        # look up actual Step objects using names
        steps = []
        for step_name in step_names:
            if step_name not in search_state.step_from_name:
                hint = ' (hint: are you missing an arg count?)' if '.' not in step_name else ''
                raise ParseError(f"unrecognized step name {step_name} in {s}{hint}")
            steps.append(search_state.step_from_name[step_name])
        # initialize stack program
        p = StackProgram(
                    p=(),
                    search_state=search_state,
                    ll=0,
                    stack_state=StackState(())
                    )
        # apply steps 
        for step in steps:
            try:
                p = p.take_step(step, error_if_seen=False, update_if_new=False) # may raise ProgramFail
            except ProgramFail as e:
                raise ProgramFail(f"During execution of {s} on step {step.name} with program {p} encountered exception: {e}")
        return p

        


def canonicalize(s:str):
    """
    meshgrid.0, -1, cov.1 -> (meshgrid.0, -1, cov.1)
    meshgrid.0 -1 cov.1 -> (meshgrid.0, -1, cov.1)
    (meshgrid.0 -1 cov.1) -> (meshgrid.0, -1, cov.1)
    meshgrid.0,    -1,     cov.1 -> (meshgrid.0, -1, cov.1)
    meshgrid.0,-1,cov.1 -> (meshgrid.0, -1, cov.1)
    meshgrid.0,-1, cov.1 -> (meshgrid.0, -1, cov.1)
    """
    s_original = s
    s = s.strip()
    # spaces become redundant commas
    s = s.replace(' ',',')
    # strip any parens
    if s.startswith('('):
        if not s.endswith(')'):
            raise ParseError(f'starts with open paren but doesnt end with close paren: {s_original}')
        s = s[1:-1] # strip parens
    # split by commas
    items = s.split(',')
    items = [x for x in items if len(x) != 0]
    return '(' + ', '.join(items) + ')'

    

class ParseError(Exception): pass

@dataclass(order=True)
class HeapItem:
    prio: float
    p: StackProgram = field(compare=False)


class StackState:
    """an immutable stack class. push() and pop() dont do inplace modifications
    """
    def __init__(self, stack: tuple):
        stack = tuple(stack)
        # this first deepcopy detatches us from our parent so we cant accidentally modify them
        # and trigger their need to restore() (which we'd have to warn them about which we cant do)
        # also its pretty harmless given we're already deepcopying to get our backup copy for restoring
        self.stack = deepcopy(stack)
        # this second deepcopy is to handle restoring ourselves when we accidentally
        # do an inplace modification of ourselves
        self.stack_backup = deepcopy(stack)
    def restore(self):
        if not safe_eq(self.stack,self.stack_backup):
            # our stack has been modified by some inplace operation! Lets restore the original
            self.stack = deepcopy(self.stack_backup)
            return True
        return False
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
        return safe_eq(self.stack,other.stack)
    def __repr__(self):
        res = []
        for item in self.stack:
            try:
                res.append(repr(item))
            except TypeError:
                res.append('[unable to print]')
                raise # eh lets set this to raise bc we dont actually want these showing up
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
    def __init__(self, fn_info, prior, argc):
        name = f'{fn_info.name}.{argc}'
        super().__init__(val=fn_info.fn, prior=prior, name=name)
        self.argc = argc
    def __call__(self, stack_state:tuple, env:dict):
        """
        Takes a stack state and returns a new one
        important: dont modify the stack_state list that you are passed
        """
        if len(stack_state) < self.argc:
            raise ProgramFail(f'argc mismatch: expected {self.argc} got {len(stack_state)}')
            
        with contextlib.redirect_stdout(os.devnull):
            with contextlib.redirect_stderr(os.devnull):
                stack_state_prefix, args = stack_state.pop(self.argc)
                try:
                    # apply function to the last `argc` things on the stack, leaving everything before that unchanged
                    res = self.val(*args)
                    repr(res) # this crashes sometimes so we want to try it
                except Exception as e:
                    raise ProgramFail(f'crash during execution of program: {e}')
                finally:
                    # in case of failure, we want to restore the stack state to whatever it originally was even if inplace operations happened
                    restored1 = stack_state.restore()
                    restored2 = stack_state_prefix.restore() # I dont think this should be affected but good to be safe in case theres some pointers between items in the tuple. Unsure but this is safer.
                    if restored1 or restored2:
                        raise ProgramFail(f'program resulted in an inplace modification')
                    #str(env['x']()) # *** TEMP just to debug errror
                return stack_state_prefix.push(res)
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
