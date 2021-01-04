from pathlib import Path
from util import *
import numpy as np
import sys
from collections import defaultdict
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import inspect
import itertools
import operator

class UnusualTrafficException(Exception):
    pass

class FunctionInfo:
    stats = defaultdict(int)
    MAX_ARITY = 4
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.description = fn.__doc__

        # None indicates we don't know
        self.has_axis = None
        self.has_keepdims = None
        self.has_varargs = None
        self.argcs = None
        self.arginfos = None

        try:
            self.params = inspect.signature(fn).parameters
        except ValueError:
            self.params = None
            print(f"can't find signature for {self.name}")
        
        # see what we can learn from inspect.signature()
        if self.params is not None:
            min_argc = 0
            self.has_axis = False
            self.has_keepdims = False
            self.has_varargs = False
            for name,p in self.params.items():
                self.stats[name] += 1
                if name == 'axis':
                    self.has_axis = True
                if name == 'keepdims':
                    self.has_keepdims = True
                if p.kind == p.VAR_POSITIONAL:
                    self.has_varargs = True
                elif p.kind == p.POSITIONAL_OR_KEYWORD:
                    if p.default is p.empty:
                        min_argc += 1 # required argument since it has no default
                else:
                    print(f"ignoring unusual param: {self.name}() has arg `{name}` of kind '{p.kind.description}'")
            max_argc = min([len(self.params), FunctionInfo.MAX_ARITY])
            if self.has_varargs:
                max_argc = FunctionInfo.MAX_ARITY
            self.argcs = list(range(min_argc,max_argc+1))

from typing import Callable
class ArgInfo:
    def __init__(self,allow_terminal=(lambda expr:True)):
        self.allow_terminal = allow_terminal
    def _and(self,fn):
        old = self.allow_terminal # decided to pull this out instead of closuring it
        return ArgInfo(lambda expr: old(expr) and fn(expr))
    def _or(self,fn):
        old = self.allow_terminal # decided to pull this out instead of closuring it
        return ArgInfo(lambda expr: old(expr) or fn(expr))

def pre_synth(fns,cfg):
    consts = [-1,0,1,2,None]
    vars = ['x','y']

    # load and normalize Primitive frequency data
    freq_path = Path('saved/freq.dict')
    freq = load(freq_path)
    tot = sum(freq.values())
    priors = {name: count/tot for name, count in freq.items()}
    priors['_var'] = .2
    priors['_const'] = .2
    priors['_default'] = .1
    priors['to_tuple'] = .05
    priors['index'] = .2
    priors['slice'] = .05

    # renormalize now that var/const are added and also convert to log
    tot = sum(priors.values())
    priors = {name: np.log(p/tot) for name, p in priors.items()}

    fns = [FunctionInfo(f) for f in fns]

    counts = [(name,count) for name,count in FunctionInfo.stats.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    print('arg name frequencies:')
    for name,count in counts:
        if count < 5:
            continue
        print(f'\t {name}: {count}')
    

    return priors,consts,vars,fns



def get_freqs(fns, cfg):
    driver = Driver(headless=(not cfg.no_headless))
    freq_path = Path('saved/freq.dict')
    names = [f.__name__ for f in fns]

    if freq_path.exists():
        print(f'loading existing {freq_path}')
        freq = load(freq_path)
        print(f"continue at fn number {len(freq)}")
    else:
        print("creating new freq")
        freq = dict()

    for name in names:
        if name in freq:
            continue  # eg when loading from a partially complete freq dict
        print(name)
        query = f'"np.{name}"'
        try:
            num_results = driver.get_num_results(query)
        except UnusualTrafficException:
            print("Unusual traffice error, saving...")
            save(freq, freq_path)
            driver.driver.close()
            sys.exit(0)
        print(f"-> {num_results}")
        freq[name] = num_results

    print("saving results...")
    save(freq, freq_path)
    driver.driver.close()

def get_numpy_fns():
    all_fns = [getattr(np, x) for x in dir(np) if callable(getattr(np, x))]
    by_types = defaultdict(list)
    for f in all_fns:
        by_types[type(f)].append(f)

    del by_types[type]  # things like np.bool, etc

    # theres just one instance of this
    del by_types[np._pytesttester.PytestTester]

    # ufuncs are stuff like np.log() np.exp() etc. Short for "universal function". They act elemwise. Not super relevant to us
    ufuncs = by_types[np.ufunc]

    # cherrypicked from the 20 builtin functions. The rest are things like fromiter(), seterrorobj(), and other things we dont like
    builtins = [np.arange, np.array, np.empty, np.zeros]

    # these are the bulk of the fns we actually care about
    fns = by_types[type(lambda:0)]  # get the <class 'function'> type
    fns += builtins

    # ignore ones starting with underscore
    fns = [f for f in fns if not f.__name__.startswith('_')]

    custom = [
        tup, # make a tuple
        slice, # make a slice
        operator.eq,
        operator.ne,
        operator.add,
        operator.sub,
        operator.mul,
        operator.getitem, # indexing and slicing. Takes an object and an index/slice
    ]
    fns += custom

    # note: setbufsize causes wild crashes lol
    reject = {'lookfor', 'info', 'source', 'printoptions', 'set_printoptions','setbufsize', 'seterr', 'seterrcall','set_string_function', 'get_printoptions', 'getbufsize','get_include','deprecate'}
    fns = [f for f in fns if f.__name__ not in reject]

    return fns

def tup(*args):
    return tuple(args)
def index(obj,*slices):
    """
    Easier slicing for use in DSL. Roughly this is obj[*slices] though
    of course you cant use star expansion in a slice (you need to instead
    use a tuple of slices as your indexer). And furthermore we'll add some


    dont worry a[(1,)] is the same as a[1] i think so this works for normal
    indexing too thats just the 1-tuple case


    slice(None)  -> :       # one arg and it's None: no constraints
    slice(3)     -> :3      # one arg: interpret as end=...
    slice(None,3) -> same as prev
    slice(3,4)    -> 3:4    # two arg: interpret as start=... end=...
    slice(3,None) -> 3: # since it says "start at 3 but have no end"

    """
    return obj[slices]
def _eq(a,b):
    return a == b
def _add(a,b):
    return a + b

class Driver:
    def __init__(self, headless=False):
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument("--incognito")
        options.add_argument('--disable-extensions')
        chrome_app = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        # alt you can just throw chromedriver anywhere in path and itll figure it out
        chrome_driver = '/Users/matthewbowers/python/headless/chromedriver'
        options.binary_location = chrome_app

        self.driver = webdriver.Chrome(chrome_driver, options=options)

        # if searching for an element fails always retry for up to 1s in case it just
        # hasn't loaded yet. Issue: youll want to turn this off if ur ever checking if an element exists
        # bc whenever that fails it'll take a full second to fail which is a lot. So if failing to find
        # something is expected then this will massively slow you down.
        self.driver.implicitly_wait(1)
        print("initialized chrome driver")

    def check_unusual_traffic(self, raised=True):
        if 'Our systems have detected unusual traffic from your computer network' in self.driver.page_source:
            if raised:
                raise UnusualTrafficException
            return True
        return False

    def get_num_results(self, query):
        """
        Returns number of google search results for a query like "test" or "\"test\"" (latter would be a verbatim search)
        """
        print("searching google")
        self.driver.get(f"http://www.google.com/search?q={query}")

        self.check_unusual_traffic()

        print("finding number of results")
        results = self.driver.find_element_by_id('result-stats').text
        # 'About 4,150,000,000 results (0.59 seconds) '
        _about, num, _results, *ignore_rest = results.split(' ')
        if _about != 'About':  # edge case where it just says "10 results" instead of "About 10 results"
            assert _about.isdigit()
            num_results = int(_about.replace(',', ''))
            return num_results
        assert _about == 'About'
        assert _results == 'results'
        num_results = int(num.replace(',', ''))
        return num_results