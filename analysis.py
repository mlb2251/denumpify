


def partition_hspace(exprs, env):
  """
  Take a set of exprs (List[Expr]) and an environment (dict)
  and execute each expr in the environment. Group them all
  by output value (use Val.__eq__ which is safe_eq).
  """
  for e in exprs:
    res = e.run(env)
    
