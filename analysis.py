
from collections import defaultdict

class HTree:
  def __init__(self,children:dict):
    self.children = children # dict from `Val -> HSpace|HTree` where the Val is the shared output val that led us to split the space in this way
    
from pcfg_bottom_up import Val
class HSpace:
  def __init__(self,exprs):
    self.exprs = exprs # [Expr]

  def narrow(self,env:dict, out:Val):
    exprs = []
    for e in self.exprs:
      res = e.run(env)
      if Val(res) == out:
        exprs.append(e)
    return HSpace(exprs)

  def split(self,env:dict):
    """
    Take a set of exprs (List[Expr]) and an environment (dict)
    and execute each expr in the environment. Group them all
    by output value (use Val.__eq__ which is safe_eq).
    """
    subspaces = defaultdict(list) # Val -> Expr
    for e in self.exprs:
      res = e.run(env)
      if isinstance(res,Exception):
        res = Exception
      subspaces[Val(res)].append(e)
    return HTree({output:HSpace(exprs) for output,exprs in subspaces.items()})

