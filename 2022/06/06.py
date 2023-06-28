#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/6
"""

from collections import deque


# -- part 1 --

d = deque(maxlen=4)
idx = 0
with open('input') as f:
    while True:
        c = f.read(1)
        idx += 1
        if not c:
            print('Failed to find marker.')
            break
        d.append(c)
        if len(set(d)) == 4:
            break

print(''.join(d))
print(idx)


# -- part 2 --

d = deque(maxlen=14)
idx = 0
with open('input') as f:
    while True:
        c = f.read(1)
        idx += 1
        if not c:
            print('Failed to find marker.')
            break
        d.append(c)
        if len(set(d)) == 14:
            break

print(''.join(d))
print(idx)
