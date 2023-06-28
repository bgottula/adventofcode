#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/3
"""

import numpy as np


# -- part 1 --

with open('input') as f:
    priorities = []
    for line in f:
        s1 = {c for c in line[:len(line)//2]}
        s2 = {c for c in line[len(line)//2:]}
        c = (s1 & s2).pop()
        priorities.append(ord(c) - 38 if c.isupper() else ord(c) - 96)

print(np.sum(priorities))


# -- part 2 --

with open('input') as f:
    priorities = []
    group = []
    for line in f:
        group.append({c for c in line.strip()})
        if len(group) == 3:
            c = set.intersection(*group).pop()
            group = []
            priorities.append(ord(c) - 38 if c.isupper() else ord(c) - 96)

print(np.sum(priorities))
