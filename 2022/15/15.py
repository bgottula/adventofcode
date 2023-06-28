#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/15
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Optional
import numpy as np

# load up the puzzle input
sensors = []
beacons = []
filename = 'input'
with open(filename) as f:
    for line in f:
        m = re.search(
            'Sensor at x=(-?[0-9]+), y=(-?[0-9]+): '
            'closest beacon is at x=(-?[0-9]+), y=(-?[0-9]+)',
            line
        )
        sensors.append((int(m.group(1)), int(m.group(2))))
        beacons.append((int(m.group(3)), int(m.group(4))))
sensors = np.array(sensors)
beacons = np.array(beacons)
l1_norms = np.sum(np.abs(sensors - beacons), axis=1)


def calc_beacon_free_columns(y: int) -> int:
    """Calculate the number of beacon-free columns on row y."""
    spans = []
    for s, b, l1 in zip(sensors, beacons, l1_norms):

        y_diff = abs(y - s[1])
        if y_diff >= l1:
            continue

        x_min = s[0] - (l1 - y_diff)
        x_max = s[0] + (l1 - y_diff)

        # handle case where beacon is on this row
        if b[1] == y:
            if x_min == x_max:
                assert b[0] == x_min
                continue

            if b[0] == x_min:
                x_min += 1
            elif b[0] == x_max:
                x_max -= 1

        spans.append((x_min, x_max))

        print(f'Sensor {s}: {x_min = }, {x_max = }')

    spans = np.sort(np.array(spans), axis=0)

    # merge overlapping spans
    merged = []
    for span in spans:
        if not merged:
            merged.append(span)

        if span[0] <= merged[-1][1]:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    merged = np.array(merged)

    return np.sum(merged[:,1] - merged[:,0] + 1)


def find_distress_beacon(y: int, x_bounds: tuple[int, int]) -> Optional[int]:
    """Find the distress beacon on this row.

    Returns:
        The column index if the distress beacon is found or None.
    """
    spans = []
    for s, l1 in zip(sensors, l1_norms):

        y_diff = abs(y - s[1])
        if y_diff >= l1:
            continue

        x_min = s[0] - (l1 - y_diff)
        x_max = s[0] + (l1 - y_diff)

        # limit to allowed x-range
        if x_max < x_bounds[0]:
            continue
        if x_min > x_bounds[1]:
            continue
        x_min = max(x_bounds[0], x_min)
        x_max = min(x_bounds[1], x_max)

        spans.append((x_min, x_max))

    spans = np.sort(np.array(spans), axis=0)

    # merge overlapping spans
    merged = []
    for span in spans:
        if not merged:
            merged.append(span)

        if span[0] <= merged[-1][1]:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    merged = np.array(merged)

    num_distress_free = np.sum(merged[:,1] - merged[:,0] + 1)
    if num_distress_free == (x_bounds[1] - x_bounds[0] + 1):
        return None

    if len(merged) == 1:
        # single span should have length one less than allowed bounds
        assert (merged[0,-1] - merged[0,0] + 1) == (x_bounds[1] - x_bounds[0])
        if merged[0,0] != x_bounds[0]:
            return x_bounds[0]
        elif merged[0,-1] != x_bounds[1]:
            return x_bounds[1]
        else:
            print(f'Something went wrong, only one span but all wrong')
            import IPython; IPython.embed()

    # should have at most two spans total or there's a bug or the puzzle lied
    assert len(merged) == 2

    # distress beacon should be between the two segments
    return merged[0,-1] + 1



# -- part 1 --
if filename == 'example':
    y = 10
elif filename == 'input':
    y = 2000000
num_beacon_free = calc_beacon_free_columns(y)
print(num_beacon_free)


# -- part 2 --
bound = 20 if filename == 'example' else 4000000
for y in range(bound + 1):
    if y % 10000 == 0:
        print(f'{y = }')
    if x := find_distress_beacon(y, (0, bound)):
        print(f'Found it at ({x},{y})')
        break
tuning_frequency = 4000000 * x + y
print(tuning_frequency)

import IPython; IPython.embed()
