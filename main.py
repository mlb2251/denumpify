from pathlib import Path
import os
from pcfg_bottom_up import pcfg_bottom_up
import sys
import argparse
import mlb
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from compiletime import *
from synth import *
from util import *

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
        pcfg_bottom_up(fns,cfg)
    else:
        raise ValueError(cfg.mode)


if __name__ == '__main__':
    with mlb.debug((not cfg.no_debug)):
        main()
