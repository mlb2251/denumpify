

class Expr:
    pass

class App(Expr):
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

        assert self.fn.is_func
        assert len(self.args) == self.fn.tp.argc
    def eval(self,env:dict):
        self.fn.val([arg.eval(env) for arg in self.args])
    def __repr__(self):
        args = ','.join([repr(arg) for arg in self.args])
        return  f'{self.fn}({args})'


class Var(Expr):
    def __init__(self, name:str):
        self.name = name
    def eval(self,env:dict):
        return env[self.name]
    def __repr__(self):
        return self.name

class Const(Expr):
    def __init__(self,val):
        self.val = val
    def eval(self,env:dict):
        return self.val
    def __repr__(self):
        return repr(self.val)

class Prim:
    def __init__(self, name, val, tp):
        self.name = name
        self.val = val
        self.tp = tp

    
    @property
    def is_func(self):
        return isinstance(self.tp,TFunc)

    def __call__(self, *args):
        if self.is_func:
            return App(self, args)
        return Const(self)
    def __repr__(self):
        return self.name


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
