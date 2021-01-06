
from collections import defaultdict

def partition_hspace(exprs, env):
  from pcfg_bottom_up import Val
  """
  Take a set of exprs (List[Expr]) and an environment (dict)
  and execute each expr in the environment. Group them all
  by output value (use Val.__eq__ which is safe_eq).
  """
  exceptions = []
  vals = defaultdict(list) # Val -> Expr
  for e in exprs:
    res = e.run(env)
    if isinstance(res,Exception):
      exceptions.append(e)
      continue
    vals[Val(res)].append(e)
  
  return vals,exceptions

    
