#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/4
"""

import re


num_subsets = 0
num_overlaps = 0
with open('input') as f:
    for line in f:
        line = line.strip()
        m = re.search(r'([0-9]+)-([0-9]+),([0-9]+)-([0-9]+)', line)

        s1 = set(range(int(m.group(1)), int(m.group(2)) + 1))
        s2 = set(range(int(m.group(3)), int(m.group(4)) + 1))

        if s1 <= s2 or s2 <= s1:
            num_subsets += 1

        if s1 & s2:
            num_overlaps += 1

print(num_subsets)
print(num_overlaps)
