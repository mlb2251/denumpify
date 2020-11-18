

class Expr:
    def __init__(self,ll=None,from_prim=None):
        if ll is None and from_prim is not None:
            ll = from_prim.prior

        self.prim = from_prim
        self.ll = ll

        # eval stuff
        self._eval_envs = []
        self._eval_vals = []
        self._saved_size = None
        self._saved_depth = None
        self.last_eval = None
    def eval(self, env:dict):
        if env in self._eval_envs:
            return self._eval_vals[self._eval_envs.index(env)]
        res = self._eval(env)
        self._eval_envs.append(env)
        self._eval_vals.append(res)
        self.last_eval = res
        return res
    def _eval(self, env:dict):
        raise NotImplementedError
    @property
    def size(self):
        if self._saved_size is None:
            self._saved_size = self._size()
        return self._saved_size
    def _size(self):
        raise NotImplementedError
    @property
    def depth(self):
        if self._saved_depth is None:
            self._saved_depth = self._depth()
        return self._saved_depth
    def _depth(self):
        raise NotImplementedError
        

class ProgramException(Exception): pass

class App(Expr):
    def __init__(self, fn_prim, args, **kwargs):
        super().__init__(**kwargs)
        assert fn_prim.is_func
        assert len(args) == fn_prim.tp.argc
        assert all([isinstance(arg, Expr) for arg in args])

        self.fn = fn_prim.val
        self.tp = fn_prim.tp
        self.name = fn_prim.name
        self.args = args

    def _eval(self,env:dict):
        args = [arg.eval(env) for arg in self.args]
        try:
            return self.fn(args)
        except Exception as e:
            return ProgramException(e)
    def __repr__(self):
        args = ','.join([repr(arg) for arg in self.args])
        return  f'{self.name}({args})'
    def _size(self):
        return 1 + sum([arg.size for arg in self.args])
    def _depth(self):
        return 1 + max([arg.depth for arg in self.args])



class Var(Expr):
    def __init__(self, name:str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
    def _eval(self,env:dict):
        try:
            return env[self.name]
        except KeyError as e:
            return ProgramException(e)
    def __repr__(self):
        return self.name
    def _size(self):
        return 1
    def _depth(self):
        return 1

class Const(Expr):
    def __init__(self, val, **kwargs):
        super().__init__(**kwargs)
        self.val = val
    def _eval(self,env:dict):
        return self.val
    def __repr__(self):
        return repr(self.val)
    def _size(self):
        return 1
    def _depth(self):
        return 1


class Prim:
    def __init__(self, name, val, tp, prior=0):
        self.name = name
        self.val = val
        self.tp = tp
        self.id = name
        self.prior = prior
    
    @property
    def is_func(self):
        return isinstance(self,PrimFunc)

    def __repr__(self):
        return self.name

class PrimConst(Prim):
    def __call__(self):
        return Const(self.val, from_prim=self)
class PrimVar(Prim):
    def __call__(self):
        return Var(self.val, from_prim=self)
class PrimFunc(Prim):
    def __call__(self, *args):
        ll = self.prior + sum([arg.ll for arg in args]) if self.prior is not None else None
        return App(self, args, ll=ll, from_prim=self)


class Type:
    pass

class TAny(Type):
    pass

class TFunc(Type):
    def __init__(self,argc):
        self.argc = argc
def tfunc(*args, **kwargs):
    return TFunc(*args, **kwargs)

class TVal(Type):
    pass


"""
vstack = Primitive(np.vstack, 1, 'vstack')

vstack()

"""
