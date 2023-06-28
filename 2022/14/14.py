#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/14
"""

from __future__ import annotations
from dataclasses import dataclass
import re
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Point:
    x: int
    y: int


cave = np.zeros((175, 1000), dtype=np.uint8)


def add_line_segments(points: list[Point]):
    for previous, current in zip(points, points[1:]):
        if previous.x == current.x:
            cave[min(previous.y, current.y):max(previous.y, current.y) + 1, previous.x] = 255
        else:
            cave[previous.y, min(previous.x, current.x):max(previous.x, current.x) + 1] = 255


def deposit_sand():
    loc = Point(500, 0)

    while True:
        moved = False

        # drop down as far as possible
        if (y := np.argmax(cave[loc.y + 1:, loc.x] > 0)) == 0:
            if np.all(cave[loc.y + 1:, loc.x] == 0):
                # drops off the map
                return False
        else:
            moved = True
            loc.y += y

        if cave[loc.y + 1, loc.x - 1] == 0:
            moved = True
            loc.x -= 1
            loc.y += 1
        elif cave[loc.y + 1, loc.x + 1] == 0:
            moved = True
            loc.x += 1
            loc.y += 1

        if moved == False:
            cave[loc.y, loc.x] = 128
            return True


with open('input') as f:
    for line in f:
        points = []
        while m := re.search('([0-9]+),([0-9]+)', line):
            x = int(m.group(1))
            y = int(m.group(2))
            points.append(Point(x, y))
            line = line[len(m.group(0)) + 4:]
        add_line_segments(points)


# -- part 1 --

# num_sand_grains = 0
# while deposit_sand():
#     num_sand_grains += 1
# print(num_sand_grains)


# -- part 2 --
cave[np.max(np.where(cave == 255)[0]) + 2, :] = 255

num_sand_grains = 0
while True:
    deposit_sand()
    num_sand_grains += 1
    if cave[0, 500] > 0:
        break
print(num_sand_grains)


plt.imshow(cave)
plt.xlim((450, 550))
plt.show()

import IPython; IPython.embed()
