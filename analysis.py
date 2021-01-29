
from collections import defaultdict
from typing import Type

from fastcore.basics import ifnone
from pcfg_bottom_up import Val


class Intervention:
  def __init__(self,predicate):
    self.predicate = predicate

class HTree:
  def __init__(self,children:dict):
    self.children = children # dict from `Val -> HSpace|HTree` where the Val is the shared output val that led us to split the space in this way
  # def flat(self):
  #   res = []
  #   for c in self.children:
  #     if isinstance(c,HTree):
  #       res.extend(c.flat())
  #     elif isinstance(c,HSpace):
  #       res.append(c)
  #     else:
  #       raise TypeError
        
  def __repr__(self):
    hs = [(len(hspace),output) for output,hspace in self.children.items()]
    hs.sort(key=lambda h: h[0])
    hs = [f'{h[0]}: {h[1]}' for h in hs]
    return 'HTree:\n' + '\n\t'.join(hs)
  # def split(self,env:dict):
  #   self.children = [child.split(env) for child in self.children]
    

from copy import deepcopy

class IOIntervention:
  def __init__(self,env:dict):
    self.env = env
  def __call__(self,expr):
    res = expr.run(deepcopy(self.env))
    if isinstance(res,Exception):
      res = Exception
    return Val(res)


class HSpace:
  def __init__(self, groups:dict, interventions:tuple):
    self.interventions = interventions
    self.groups = groups # {[O] -> [Expr]}

    self.num_exprs = sum(len(exprs) for exprs in self.groups.values())
    self.largest_group = max(len(exprs) for exprs in self.groups.values())
    self.avg_group = self.num_exprs/len(self.groups)
  def __repr__(self):
    return f'HSpace(largest={self.largest_group}, avg={self.avg_group:.3f}, num_exprs={self.num_exprs}, num_groups={len(self.groups)})'
  def head(self, n=10):
    hs = [(len(exprs),outputs) for outputs,exprs in self.groups.items()]
    hs.sort(key=lambda h: -h[0])
    return hs[:n]
  def repr_head(self,n=10):
    hs = [f'{length}: {outputs}' for (length,outputs) in self.head(n)]
    return f'{self}:\n\t' + '\n\t'.join(hs)
    
  
  @staticmethod
  def new(exprs):
    return HSpace(groups={():exprs}, interventions=())

  def split(self,intervention):
    """
    """
    groups = defaultdict(list) # (O,O,O,...) -> [Expr]
    for outputs,exprs in self.groups.items():
      for e in exprs:
        res = intervention(e)
        groups[(*outputs,res)].append(e)
    return HSpace(groups, (*self.interventions,intervention))

  # def narrow(self, intervention, out):
  #   exprs = []
  #   for e in self.exprs:
  #     res = intervention(e)
  #     if res == out:
  #       exprs.append(e)
  #   return HSpace(exprs, interventions = self.interventions + [intervention])


