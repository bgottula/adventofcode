#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/22
"""

# Part 1 seems fairly straightforward, if a little tedious.
#
# I think the key design decision is what data structure to use to represent
# the map. The easiest I can think of is a 2D Numpy array with a simple mapping
# from the ASCII art characters to integers, such as
# ' ': 0
# '.': 1
# '#': 2
#
# For each move:
# 1) Get the current row or column.
# 2) Compute how many spaces can be moved until hitting a wall.
# 3) Determine how many spaces are actually moved.
# 4) Account for wrapping.
#
# Wrapping seems like the annoying part in terms of bookkeeping. Using a
# horizontal move as an example, let's say the row we're on is as follows:
#
#     ....#........
#
# We're starting just right of the wall, and need to move 10 spaces to the
# right. The index of the starting location is 9, but we'll first truncate the
# blank spaces from the start and end of the array so it instead looks like
# this:
#
# ....#........
#
# Now the starting index is 5. Since the row is 13 spaces wide, we compute
# (5 + 10) % 13 = 15 % 13 = 2
#
# So the end position in the truncated array is 2. Now, add the spaces that
# were truncated from the front of the array back again to get a final position
# of 6.
#
# To compute the max number of spaces we can move before hitting a wall, repeat
# the row such that there are two copies side by side:
#
# ...#...........#........
#
# If the move is to the right, the starting position is in the left copy.
# Otherwise, the starting position is in the right copy. Now simply compute
# the number of spaces between the starting position and the first wall in the
# movement direction. In the special case where there are no walls, the max
# number of moves is undefined (infinite).

import enum
import re
import numpy as np


class Direction(enum.IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# mapping from ASCII art map characters to integers
d = {" ": 0, ".": 1, "#": 2}

with open("example") as f:
    map_rows = []
    for line in f:
        if "." in line:
            map_rows.append([d[c] for c in line.rstrip("\n")])
        elif line:
            path = line.rstrip("\n")

width = max(map(len, map_rows))
map_array = np.array([row + [0] * (width - len(row)) for row in map_rows])


direction = Direction.RIGHT
position = [0, np.argmax(map_array[0, :] == 1)]
while True:
    # Get next move magnitude
    spaces = re.search(r"\d+", path).group()
    path = path[len(spaces) :]
    spaces = int(spaces)
    print(f"Moving {spaces} spaces to the {direction}")

    if direction in [Direction.LEFT, Direction.RIGHT]:
        pos = position[1]
        row = map_array[position[0], :]
    else:
        pos = position[0]
        row = map_array[:, position[1]]

    offset = np.argmax(row > 0)
    pos_trunc = pos - offset
    row_trunc = row[row > 0]
    row_dupe = np.tile(row_trunc, 2)
    if np.all(row != 2):
        max_move = None
    elif direction in [Direction.RIGHT, Direction.DOWN]:
        max_move = np.argmax(row_dupe[pos_trunc:] == 2) - 1
    else:
        pos_trunc_offset = pos_trunc + row_trunc.size
        max_move = np.argmax(np.flip(row_dupe[:pos_trunc_offset]) == 2)

    spaces = max(spaces, max_move)
    if direction in [Direction.LEFT, Direction.UP]:
        spaces *= -1
    pos_trunc = (pos_trunc + spaces) % pos_trunc.size

    if direction in [Direction.LEFT, Direction.RIGHT]:
        position[1] = pos_trunc + offset
    else:
        position[0] = pos_trunc + offset

    if not path:
        break

    direction += 1 if path[0] == "R" else -1
    direction %= 4
    path = path[1:]


password = 1000 * (position[0] + 1) + 4 * (position[1] + 1) + direction
print(password)

print(f"Row: {position[0] + 1}")
print(f"Column: {position[1] + 1}")
print(f"Direction: {direction}")

import IPython

IPython.embed()
