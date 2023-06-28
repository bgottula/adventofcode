#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/18
"""

import numpy as np

# 😈
import sys
sys.setrecursionlimit(15000)


lava = np.loadtxt('input', delimiter=',', dtype=int)

sides = [
    np.array([0, 0, +1]),
    np.array([0, 0, -1]),
    np.array([+1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, +1, 0]),
    np.array([0, -1, 0]),
]

# Part 1
external_surface_area = 0
for lava_cube in lava:
    lava_surface_area = 6
    for side in sides:
        if np.any(np.logical_and.reduce(lava == (lava_cube + side), axis=1)):
            lava_surface_area -= 1
    external_surface_area += lava_surface_area

print(f'Total surface area: {external_surface_area}')


# Part 2

# leave a gap of one cube between lava extremities and bounds of the space
lower_bound = np.min(lava, axis=0) - 1
upper_bound = np.max(lava, axis=0) + 1

# total volume of the 3D space around the lava
space_volume = np.prod(upper_bound - lower_bound + 1)

# to be filled with coordinates of "air"; starting with -100 dummy coordinate values
air = np.full((space_volume, 3), -100, dtype=int)

start_location = lower_bound
air[0, :] = 0  # mark starting location as being "air"
air_idx = 1

def part2(current_location: np.ndarray, surface_area_so_far: int, depth: int = 0) -> int:
    """Calculate external surface area of lava."""

    new_surface_area = 0
    for side in sides:
        new_location = current_location + side

        # check if the new location is beyond the search volume
        if np.any(new_location > upper_bound) or np.any(new_location < lower_bound):
            continue

        # there's air in this direction, but it's already been visited
        if np.any(np.logical_and.reduce(air == new_location, axis=1)):
            continue
        # there's a bit of lava in this direction
        if np.any(np.logical_and.reduce(lava == new_location, axis=1)):
            new_surface_area += 1
            continue

        # nothing detected here, so must be unvisited air
        global air_idx
        air[air_idx, :] = new_location
        air_idx += 1
        try:
            new_surface_area += part2(new_location, surface_area_so_far, depth + 1)
        except RecursionError:
            print(f'Got RecursionError at {depth=}')
            sys.exit(1)

    return surface_area_so_far + new_surface_area


external_surface_area = part2(start_location, 0)
print(f'External surface area: {external_surface_area}')


import IPython; IPython.embed()
