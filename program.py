

class Expr:
    pass


class Application(Expr):
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __call__(self, *args):
        self.fn.val([arg])


class Index:
    def __init__(self, i):
        self.i = i


class Primitive:
    def __init__(self, fn, argc, name=None):
        if name is None:
            if hasattr(fn, '__name__'):
                name = fn.__name__
            else:
                name = 'anon'

        self.val = val
        self.name = name
        self.argc = argc
        # self.complete = (self.argc == 0)
        # self.args = None

    def __call__(self, *args):
        if len(args) != self.argc:
            raise ValueError(
                f'wrong number of arguments to fn {self.name}. Expected {self.argc}, Got {len(args)}')
        # if self.complete:
        #     raise ValueError(
        #         f'doesnt make sense to apply function when expression doesnt take any args or has already been applied to args: {self.name}')

        # self.args = args
        # self.complete = True
        expr = Application(self, args)
        return expr


"""
vstack = Primitive(np.vstack, 1, 'vstack')

vstack()

"""
