#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/1
"""

import numpy as np

cals = []
with open('input') as f:
    total = 0
    for line in f:
        if line == '\n':
            cals.append(total)
            total = 0
        else:
            total += int(line)

# part 1
print(max(cals))

# part 2
print(np.sum(np.sort(cals)[-3:]))

import IPython; IPython.embed()
