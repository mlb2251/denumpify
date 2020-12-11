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

def pcfg_bottom_up(fns,cfg):
    priors,consts,_,fns = pre_synth(fns,cfg)


    priors = {k:int(v) for k,v in priors.items()}


    # TODO TEMP

    #priors = {k:-1 for k,v in priors.items()}
    step = 1
    target = 0

    env = {
        'x': np.ones((2,3)),
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
        nonterm = Nonterminal([],priors['_const'],lambda:const, name=str(const))
        terminals.append(Expr(nonterm,[]))
    for var in env:
        nonterm = Nonterminal([],priors['_var'],lambda:env[var], name=str(var))
        terminals.append(Expr(nonterm,[]))
    
    assert not any([t.exception for  t in terminals])

    seen = set()

    while target >= -9:
        print(f"{target =} {len(terminals) =}")
        terminals.sort(key=lambda x:-x.ll)
        frozen_terminals = terminals[:] # this shallow copy by "[:]" is v imp
        for nonterminal in nonterminals:
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
                    continue # observational equivalence
                seen.add(check_seen)
                # TODO do some observational equiv checking here (careful to assert relative lls so you dont throw out an optimal choice)
                terminals.append(expr)



        target -= 1
    terminals.sort(key=lambda x:-x.ll)
    for t in terminals[::-1]:
        print(t.pretty())
    
    repl(terminals,seen)
    print("hi")

def repl(terminals, seen):
    import readline
    str_terms = {t:str(t) for t in terminals}
    while True:
        try:
            line = input('>>> ').strip()
        except EOFError:
            break
        if line == '':
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
            mlb.red('interpreter not yet implemented')
            continue
            try:
                #p = StackProgram.parse(line,search_state)
                val = eval(line)
            except (Exceptions,ParseError) as e:
                mlb.red(e)
                continue

            seen_str = mlb.mk_green('[seen this stack state]') if p.stack_state in seen else mlb.mk_red('[not seen]')
            print(f'{p.pretty()} {seen_str}')



class Val:
    def __init__(self,val):
        self.val = val
    def __hash__(self):
        return safe_hash(self.val)
    def __eq__(self,other):
        return safe_eq(self.val,other.val)
    

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




# class Terminal:
#     def __init__(self,expr,val):
#         self.expr = expr
#         self.val = expr.val
#         self.ll = expr.ll

#     @staticmethod
#     def from_nonterminal(nonterminal, args):
#         # args :: [Terminal]
#         assert len(args) == nonterminal.argc
#         expr = Expr(nonterminal,[term.expr for term in args])
#         return Terminal(expr, ll)




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
                    self.val = Exception # not important
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
        val = repr(self.val).replace('\n','').replace(' ','').replace('\t','')
        val = mlb.mk_blue(val)
        return f'{self} (ll={self.ll}) -> {val}'
        
    # @staticmethod
    # def from_nonterminal(nonterminal, args):
    #     # args :: [Terminal]
    #     return Expr(nonterminal,args)





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







