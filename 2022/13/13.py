#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/13
"""

from bisect import bisect_left
from enum import IntEnum
import functools


class Order(IntEnum):
    IN_ORDER = -1
    OUT_OF_ORDER = +1
    UNKNOWN = 0


def compare(left, right, level=0) -> Order:
    indent = '  ' * level
    print(f'{indent}- Compare {left} vs {right}')

    if isinstance(left, int) and isinstance(right, int):
        if left < right:
            print(f'{indent}  - Left side is smaller, so inputs are in the right order')
            return Order.IN_ORDER
        elif right < left:
            print(f'{indent}  - Right side is smaller, so inputs are not in the right order')
            return Order.OUT_OF_ORDER
        else:
            return Order.UNKNOWN
    elif isinstance(left, int) and isinstance(right, list):
        return compare([left], right, level=level+1)
    elif isinstance(left, list) and isinstance(right, int):
        return compare(left, [right], level=level+1)

    # both are lists
    idx = 0
    while True:
        if len(left) < idx + 1 and len(right) < idx + 1:
            return Order.UNKNOWN
        elif len(left) < idx + 1:
            print(f'{indent}  - Left side ran out of items, so inputs are in the right order')
            return Order.IN_ORDER
        elif len(right) < idx + 1:
            print(f'{indent}  - Right side ran out of items, so inputs are not in the right order')
            return Order.OUT_OF_ORDER

        if (result := compare(left[idx], right[idx], level=level+1)) != Order.UNKNOWN:
            return result

        idx += 1


packets = []
with open('input') as f:
    for line in f:
        l = line.strip()
        if len(l) == 0:
            continue
        packets.append(eval(l))


# -- part 1 --
pair_idx = 1
right_order_indices = []
while True:
    print(f'== Pair {pair_idx} ==')
    left_idx = 2 * (pair_idx - 1)
    right_idx = 2 * (pair_idx - 1) + 1
    try:
        result = compare(packets[left_idx], packets[right_idx])
    except IndexError:
        break
    if result == Order.IN_ORDER:
        right_order_indices.append(pair_idx)
    pair_idx += 1

print(right_order_indices)
print(sum(right_order_indices))


# -- part 2 --
packets.append([[2]])
packets.append([[6]])
sorted_packets = sorted(packets, key=functools.cmp_to_key(compare))

for idx, item in enumerate(sorted_packets):
    if item == [[2]]:
        idx_2 = idx + 1

    if item == [[6]]:
        idx_6 = idx + 1

print(idx_2 * idx_6)

import IPython; IPython.embed()
