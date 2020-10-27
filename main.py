from pathlib import Path
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import os
import sys
import selenium
import argparse
from bs4 import BeautifulSoup
import requests
import mlb
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.ion()


parser = argparse.ArgumentParser(description='Denumpify')
parser.add_argument('mode', nargs='?',
                    choices=['synth', 'freq', 'plot'],
                    default='synth',
                    help='mode to run in [default: synth]')
parser.add_argument('--no-debug', action='store_true',
                    help='disable mlb.debug')
parser.add_argument('--no-headless', action='store_true',
                    help='disable headless browsing')
cfg = parser.parse_args()


class UnusualTrafficException(Exception):
    pass


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

    # ignore ones starting with underscore
    fns = [f for f in fns if not f.__name__.startswith('_')]

    return fns


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


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


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


def synth(fns, cfg):
    pass


def main():
    fns = get_numpy_fns()
    print(f"got {len(fns)} numpy fns")

    if cfg.mode == 'freq':
        get_freqs(fns, cfg)
    elif cfg.mode == 'plot':
        freq_path = Path('saved/freq.dict')
        # f, ax = plt.subplots(figsize=(6, 15))
        freq = load(freq_path)
        tot = sum(freq.values())
        data = [(name, count/tot*100) for name, count in freq.items()]
        data.sort(key=(lambda x: -x[1]))
        data = data[:40]

        xs = [x[0] for x in data]
        ys = [x[1] for x in data]
        sns.barplot(x=ys, y=xs)

        # ax.set(xlim=(0, 24), ylabel="",
        #        xlabel="Automobile collisions per billion miles")
        plt.xlabel("Search results (percent)")
        plt.show()
        breakpoint()
    elif cfg.mode == 'synth':
        synth(fns, cfg)
    else:
        raise ValueError(cfg.mode)


if __name__ == '__main__':
    with mlb.debug((not cfg.no_debug)):
        main()
