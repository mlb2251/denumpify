import mlb
import numpy as np
from collections import defaultdict
import pickle

import requests
from bs4 import BeautifulSoup
import argparse
import selenium
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options


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

    def get_num_results(self, query):
        """
        Returns number of google search results for a query like "test" or "\"test\"" (latter would be a verbatim search)
        """
        print("searching google")
        self.driver.get(f"http://www.google.com/search?q={query}")

        print("finding number of results")
        if 'Our systems have detected unusual traffic from your computer network' in self.driver.page_source:
            print("fuck")
            breakpoint()
        results = self.driver.find_element_by_id('result-stats').text
        # 'About 4,150,000,000 results (0.59 seconds) '
        _about, num, _results, *ignore_rest = results.split(' ')
        assert _about == 'About'
        assert _results == 'results'
        num_results = int(num.replace(',', ''))
        return num_results


def main():
    driver = Driver(headless=False)

    fns = get_numpy_fns()
    print(f"got {len(fns)} numpy fns")

    names = [f.__name__ for f in fns]

    freq = dict()

    for name in names:
        print(name)
        query = f'"np.{name}"'
        num_results = driver.get_num_results(query)
        print(f"-> {num_results}")
        freq[name] = num_results

    print("ay")

    # req = requests.get('http://www.google.com/search',
    #                    params=dict(q='\'test\'',
    #                                aqs='chrome.0.69i59j69i57j69i59j69i60j69i61j69i60l3.1924j0j7',
    #                                sourceid='chrome',
    #                                ie='UTF-8')
    #                    )
    # soup = BeautifulSoup(req.text)

    # html = soup.prettify()
    # with open('out.html', 'w') as f:
    #     f.write(html)

    print(soup.find('div', dict(id='result-stats')).text)

    for f in fns:
        p = Program(f, f.__name__)
    return


if __name__ == '__main__':
    with mlb.debug(False):
        main()
