from compiletime import pre_synth,ArgInfo
import numpy as np
import mlb
from dataclasses import dataclass,field
import os
from typing import *
import contextlib
from copy import deepcopy
from util import *
from fastcore.basics import ifnone

cfg = None

def pcfg_bottom_up(fns,_cfg):
    global cfg
    cfg = _cfg
    priors,consts,_,fns = pre_synth(fns,cfg)
    
    priors = {k:int(v) for k,v in priors.items()}


    if cfg.prior == 'uniform':
        priors = {k:-1 for k,v in priors.items()}

    env = {
        'x': np.random.rand(2,3),
        'y': np.array([[1,2,3],[4,5,6]]),
    }

    terminals = []
    nonterminals = []

    for f in fns:
        argcs = f.argcs if f.argcs is not None else [0,1,2,3]
        for argc in argcs:
            nonterm = Nonterminal([ArgInfo() for _ in range(argc)],
                                  priors.get(f.name,priors['_default']),
                                  f.fn)
            if f.name == 'correlate':
                nonterm.arginfos = [a._and(lambda x: isinstance(x,np.ndarray)) for a in nonterm.arginfos]
            elif f.name == 'vdot':
                nonterm.arginfos = [a._and(lambda x: not isinstance(x,type)) for a in nonterm.arginfos]
            nonterminals.append(nonterm)
    
    
    for const in consts:
        nonterm = Nonterminal([],priors['_const'],lambda const=const:const, name=str(const))
        terminals.append(Expr(nonterm,[]))
    for var in env:
        # NEVER change this from str(var). if its named anything other than "x", "y", etc
        # then we won't be able to find it in Expr.run to modify the environment
        nonterm = Nonterminal([],priors['_var'],lambda env=env,var=var:env[var], name=str(var))
        terminals.append(Expr(nonterm,[]))


    nonterminals_vars_consts = nonterminals + [e.nonterminal for e in terminals]

    assert not any([t.exception for  t in terminals])

    seen = {Val(t.val):t for t in terminals}

    if cfg.prior == 'custom':
        progs = user_inputs()
        progs = [parse(prog,nonterminals_vars_consts) for prog in progs]
        uni = get_best_unigram(progs,nonterminals_vars_consts)
        update_priors(uni)
        int_priors(nonterminals_vars_consts)

    target_stop = -cfg.w
    bup_enumerate(terminals,nonterminals,target_stop,seen)


    from analysis import HSpace,IOIntervention
    hspace = HSpace.new(terminals)

    intvs = [
        IOIntervention({
        'x': np.zeros((2,3)),
        'y': np.zeros((2,3)),
        }),
        IOIntervention({
        'x': np.eye(4),
        'y': np.ones((2,3)),
        }),
        IOIntervention({
        'x': np.eye(4),
        'y': 2,
        }),
        IOIntervention({
        'x': np.random.rand(2,3),
        'y': np.zeros((3,3)),
        }),
        IOIntervention({
        'x': np.array([[1,2,3],[1,2,3]]),
        'y': np.zeros((3,3)),
        }),
    ]

    print(hspace.repr_head())
    for i,intv in enumerate(intvs):
        hspace = hspace.split(intv)
        print(i, hspace.repr_head())

    repl(terminals,nonterminals,nonterminals_vars_consts, seen, target_stop)

def bup_enumerate(terminals,nonterminals, target_stop, seen, target_start=0):
    assert target_stop < 0
    target = target_start
    while target >= target_stop:
        terminals.sort(key=lambda x:-x.ll)
        frozen_terminals = terminals[:] # this shallow copy by "[:]" is v imp
        check_len = len(frozen_terminals)
        for nonterminal in nonterminals:
            assert len(frozen_terminals) == check_len
            if nonterminal.prior <= target:
                continue

            args_cands = get_arg_candidates(
                min_ll = target - nonterminal.prior, # minimum increases a little
                arginfos_remaining = nonterminal.arginfos,
                args_so_far = [],
                terminals = frozen_terminals,
                )
            for args_cand in args_cands:
                # args_cand :: [Terminal]
                assert nonterminal.prior + sum([term.ll for term in args_cand]) >= target
                expr = Expr(nonterminal, args_cand)
                if expr.exception:
                    continue
                check_seen = Val(expr.val)
                if check_seen in seen:
                    # TODO check relative lls maybe just to be safe
                    #assert seen[check_seen].ll >= expr.ll
                    if not cfg.no_obs_eq:
                        continue # observational equivalence
                seen[check_seen] = expr
                # TODO do some observational equiv checking here (careful to assert relative lls so you dont throw out an optimal choice)
                terminals.append(expr)



        print(f"Finished {target =} to yield a total of {len(terminals) =}")
        target -= 1
    terminals.sort(key=lambda x:-x.ll)
    for t in terminals[::-1]:
        print(t.pretty())
    

def repl(terminals,nonterminals, nonterminals_vars_consts, seen, target_stop):
    import readline
    str_terms = {t:str(t) for t in terminals}
    while True:
        try:
            line = input('>>> ').strip()
        except EOFError:
            break
        if line == '':
            continue
        if line == 'enum':
            target_stop -= 1
            bup_enumerate(terminals,nonterminals,target_stop,seen,target_start=target_stop)
            continue
        if line == 'seen':
            for val in seen:
                print(val.val)
            continue
        if line.startswith('?'):
            # treat line as a search query
            line = line[1:].strip()
            args = line.split(' ')
            args = [x for x in args if x != '']
            if len(args) == 0:
                continue
            found = str_terms
            # repeatedly narrow to only include results where the string `arg` shows up somewhere in str(p)
            for arg in args:
                found = {t:s for t,s in found.items() if arg in s}
            for t in found:
                print(t.pretty())
        else:
            # treat line as a program to execute
            try:
                e = parse(line,nonterminals_vars_consts)
            except ParseError as exc:
                mlb.red(exc)
                continue

            check_seen = Val(e.val)
            seen_str = mlb.mk_green(f'[seen this val from {seen[check_seen]} (ll={seen[check_seen].ll}) ]') if check_seen in seen else mlb.mk_red('[not seen]')
            print(f'{e.pretty()} {seen_str}')


def user_inputs():
    return [
        'concatenate(tup(y,x))',
        'repeat(y,2,1)',
        'all(eq(x,y))',
        'dot(transpose(x),y)'
    ]



class Val:
    def __init__(self,val):
        self.val = val
    def __hash__(self):
        return safe_hash(self.val)
    def __eq__(self,other):
        return safe_eq(self.val,other.val)
    def __repr__(self):
        return repr(self.val).replace('\n','').replace(' ','').replace('\t','')

def get_arg_candidates(
    min_ll, # lowest allowed ll, dont return candidates below this
    arginfos_remaining, # list of ArgInfo objects
    args_so_far, # list of Terminal objects
    terminals, # list of Terminal objects
    ):
    if len(arginfos_remaining) == 0:
        yield args_so_far # this is the base case that actually returns a real value!
        return
    arginfo = arginfos_remaining[0]
    for terminal in terminals:
        if terminal.ll < min_ll:
            return # generator stops. Note we assume terminals are sorted by ll
        if not arginfo.allow_terminal(terminal):
            continue # skip terminals that dont fit any specified requirements of this argument
        # TODO itd be easy to add an extra ll penalty here much like allow_terminal but a cost instead of outright rejection
        yield from get_arg_candidates(
            min_ll = min_ll-terminal.ll, # minimum increases a little
            arginfos_remaining = arginfos_remaining[1:],
            args_so_far = args_so_far + [terminal],
            terminals = terminals,
            )




@dataclass
class Nonterminal:
    def __init__(self,arginfos, prior, fn, name=None):
        self.arginfos = arginfos
        self.prior = prior
        self.fn = fn
        self.argc = len(arginfos)
        self.name_noargc = ifnone(name,f'{fn.__name__}')
        self.name = ifnone(name,f'{self.name_noargc}.{self.argc}')
    def __repr__(self):
        return self.name_noargc
    def __hash__(self):
        return hash(self.name)


class Expr:
    def __init__(self,nonterminal, args, ):
        assert len(args) == nonterminal.argc
        self.nonterminal = nonterminal
        self.args = args # [Expr]
        self.ll = self.nonterminal.prior + sum([a.ll for a in self.args])
        #print(f"ayy {self.nonterminal} {self.args}")
        with contextlib.redirect_stdout(os.devnull):
            with contextlib.redirect_stderr(os.devnull):
                try:
                    self.val = self.nonterminal.fn(*[a.val for a in self.args])
                    repr(self.val) # if this fails we throw it out
                    self.val_backup = deepcopy(self.val) # likewise if this fails we throw it out
                    self.exception = False
                except Exception as e:
                    self.val = e # not important
                    self.exception = True
                finally:
                    for a in self.args:
                        a.restore()
        #print("lmao")
    def restore(self):
        if not safe_eq(self.val,self.val_backup):
            # our stack has been modified by some inplace operation! Lets restore the original
            self.val = deepcopy(self.val_backup)
            return True
        return False
    def size(self):
        return 1 + sum(arg.size() for arg in self.args)
    def record_usages(self,prod_dict):
        """
        increment the slot in prod_dict[Nonterminal -> int] 
        corresponding to ur nonterminal and recurse on your children
        """
        prod_dict[self.nonterminal] += 1
        for arg in self.args:
            arg.record_usages(prod_dict)
    def __repr__(self):
        if len(self.args) == 0:
            return repr(self.nonterminal)
        args = ','.join([repr(a) for a in self.args])
        return f'{self.nonterminal}({args})'
    def pretty(self):
        if isinstance(self.val,slice):
            rep = slice_repr(self.val)
        else:
            rep = repr(self.val)
        val = rep.replace('\n','').replace(' ','').replace('\t','')
        val = mlb.mk_blue(val)
        return f'{self} (ll={self.ll}) -> {val}'
    def run(self, env):
        """
        a scrappy method that doesnt use memoization and just recursively evaluates
        the expr in the context of the provided env. Returns an actual python value
        or an Exception.
        """
        # override the environment temporarily
        if self.nonterminal.name in env: # ie if it's named "x" or "y"
            return env[self.nonterminal.name]
        
        # calculate values of args
        args = [arg.run(env) for arg in self.args]

        # return Exception if any child returned Exception
        for arg in args:
            if isinstance(arg,Exception):
                return arg

        # execute the expression
        with contextlib.redirect_stdout(os.devnull):
            with contextlib.redirect_stderr(os.devnull):
                try:
                    res = self.nonterminal.fn(*args)
                    repr(res) # if this fails we throw it out
                    return res
                except Exception as e:
                    return e


class RuntimeException(Exception): pass
        

def slice_repr(s):
    assert isinstance(s,slice)
    if s.start is None and s.stop is None and s.step is None:
        # ':' case
        return ':'
    if s.step is None:
        # 'start:stop' case
        start = ifnone(s.start,'')
        stop = ifnone(s.stop,'')
        return f'{start}:{stop}'
    # 'start:stop:step' case
    start = ifnone(s.start,'')
    stop = ifnone(s.stop,'')
    step = ifnone(s.step,'')
    return f'{start}:{stop}:{step}'



class ParseError(Exception):
    def __init__(self,*args):
        super().__init__(' '.join(args))

import re

def parse(s:str,nonterms:List[Nonterminal], nt_of_name=None, names=None) -> Expr:
    """
    note nonterms should actually be nonterminals_vars_consts
    ie it should have the Nonterminal instance for each var and const

    removes all whitespace during preproc (just as a simple solution, but
    watch out if it gives you weird errors)

    may raise ParseError
    """
    if names is None:
        names = [nonterm.name_noargc for nonterm in nonterms]
    if nt_of_name is None:
        nt_of_name = {(nt.name_noargc,nt.argc):nt for nt in nonterms}

    # fn calls become tuples: (fn_name,[args])
    #     example: foo(bar(x,y),z) -> ('foo',[('bar',['x','y']),'z'])

    # strip whitespace, remove spaces, do some basic checks
    s = s.strip()
    if '\n' in s:
        raise ParseError("why is there a newline")
    if s.count('(') != s.count(')'):
        raise ParseError("unequal number of opening and closing parens")
    if len(s) == 0:
        raise ParseError("empty string")

    # figure out what funciton / const / var is being used next
    match = re.match(r'(\w|-)+',s)
    if match is None:
        raise ParseError("Expected string to start with an identifier:",s)
    name = s[:match.end()]
    s = s[match.end():]
    if name not in names:
        raise ParseError("unrecognized nonterminal:",name)
    
    # handle var and const
    if len(s) == 0:
        # since this was the last token of the string
        # its a const or var so we wanna call the Nonterminal
        # with zero arguments
        if (name,0) not in nt_of_name:
            argcs = [argc for (fname,argc) in nt_of_name if fname == name]
            raise ParseError(f"found {name}() but can't call it with 0 args. Allowed argcs: {argcs}")
        expr = Expr(nt_of_name[(name,0)],[])
        if expr.exception:
            raise ParseError(f"exception during execution of subtree {expr}: {expr.val}")
        return expr
    
    # handle function call
    if s[0] != '(':
        raise ParseError("expected open paren, got:",s)
    if s[-1] != ')':
        raise ParseError("expected this to end with close paren:",s)

    s = s[1:-1]
    # split by commas, except when already inside parens
    args = []
    while True:
        s = s.strip()
        if s == '':
            break # only happens if zero args
        i = next_comma(s) # find net comma, but aware of paren depth
        if i == -1: # no more commas, so we should parse the rest into an Expr
            args.append(parse(s,nonterms,nt_of_name,names))
            break
        else: # parse next arg into an Expr
            args.append(parse(s[:i],nonterms,nt_of_name,names))
            s = s[i+1:] # skip the ',' and continue

    if (name,len(args)) not in nt_of_name:
        argcs = [argc for (fname,argc) in nt_of_name if fname == name]
        raise ParseError(f'{name} applied to wrong number of args, got {len(args)} but expected one of these: {argcs}')

    expr = Expr(nt_of_name[(name,len(args))],args)
    if expr.exception:
        raise ParseError(f"exception during execution of subtree {expr}: {expr.val}")
    return expr
    

def next_comma(s):
    depth = 0
    for i,char in enumerate(s):
        if char == ',' and depth == 0:
            return i
        if char == '(':
            depth += 1
        if char == ')':
            depth -= 1
            if depth < 0:
                raise ParseError("too many closing parens")
    if depth > 0:
        raise ParseError("not enough closing parens")
    return -1 # indicates no more commans


def parse_indexer(s):
    """
    parse the contents of a [...]
    assumes s has been stripped of its brackets already
    pretty limited but supports stuff like
        [:]
        [:,:]
        [:4,6:]
        [None:]
        [x] # maybe
    """
    raise NotImplementedError

def update_priors(priors_dict:Dict[Nonterminal,float]):
    for nt,prior in priors_dict.items():
        nt.prior = prior

def int_priors(prods:List[Nonterminal]):
    for prod in prods:
        prod.prior = int(prod.prior)

def get_best_unigram(progs:List[Expr], prods:List[Nonterminal], uniform_factor=.2):
    """
    Get the best unigram model based on the set of programs. Smooth it out based on the
    uniform_factor.
    Args:
        progs (list[Expr]): list of programs to calculate the best PCFG for
        prods (list[Nonterminal]): list of production rules (nonterminals)
        uniform_factor (float, default=.2): how much a uniform pcfg is favored, so
            we dont assign some productions exactly to zero. The final pcfg returned
            will be pcfg * (1-uniform_factor) + uniform_pcfg * uniform_factor.
    Returns:
        dict[Nonterminal -> ll]
    """
    usage_counts = {prod:0 for prod in prods}

    for prog in progs:
        prog.record_usages(usage_counts)

    total = sum(usage_counts.values())
    # add bias in favor of uniform distribution by removing 20% from each usage
    # count then evenly redistributing that. Smooths everythign slightly.
    bias = total*uniform_factor/len(usage_counts)
    biased_usage_counts = {prod:(1-uniform_factor)*usage + bias for prod,usage in usage_counts.items()}
    total = sum(usage_counts.values())

    pcfg = {prod:np.log(usages/total) for prod,usages in biased_usage_counts.items()}

    return pcfg


def get_best_bigram():
    """
    honestly wouldnt be much different from the unigram algorithm. You just have a usage_count
    dict for each context (ie a context = who ur parent is) then you still only normalize within
    each usage_count dict. You could imagine just having the Expr.record_usage function take a dict
    of dicts and update the correct one by telling its child stuff abt its context.

    we dont even care if stuff doesnt type check bc thatll get masked out anyways
    """







