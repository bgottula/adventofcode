#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/5
"""

import re
from collections import deque

num_stacks = 9

with open('input') as f:
    lines = f.readlines()

stacks = [deque() for _ in range(num_stacks)]
lists = [[] for _ in range(num_stacks)]

for line in lines[7::-1]:
    m = re.search(' '.join(['(\[[A-Z]\]|   )']*num_stacks), line)
    for stack_idx, (stack, list) in enumerate(zip(stacks, lists)):
        if (s := m.group(stack_idx + 1)) != '   ':
            stack.append(s[1])
            list.append(s[1])


for line in lines[10:]:
    m = re.search('move ([0-9]+) from ([1-9]) to ([1-9])', line)
    num_to_move = int(m.group(1))
    from_idx = int(m.group(2)) - 1
    to_idx = int(m.group(3)) - 1

    for _ in range(num_to_move):
        stacks[to_idx].append(stacks[from_idx].pop())

    lists[to_idx] += lists[from_idx][-num_to_move:]
    del lists[from_idx][-num_to_move:]

print(''.join([stack[-1] for stack in stacks]))
print(''.join([l[-1] for l in lists]))
