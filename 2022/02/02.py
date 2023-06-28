#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/2
"""

import numpy as np


# -- part 1 --

def convert(val: str) -> int:
    char = val.decode()
    if char in ['A', 'X']:
        return 1  # rock
    elif char in ['B', 'Y']:
        return 2  # paper
    else:
        return 3  # scissors

items = np.loadtxt('input', converters=convert, delimiter=' ')

wins = ((items[:,1] - items[:,0]) % 3 == 1)
draws = items[:,1] == items[:,0]

score = 6*wins + 3*draws + items[:,1]

print(np.sum(score))


# -- part 2 --
losses = items[:,1] == 1
draws = items[:,1] == 2
wins = items[:,1] == 3

items_mine = (items[:,0] + items[:,1] - 3) % 3 + 1

score = 6*wins + 3*draws + items_mine

print(np.sum(score))

import IPython; IPython.embed()
