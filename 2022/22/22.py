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

from __future__ import annotations
import enum
import re
import numpy as np


class Direction(enum.IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class MapVal(enum.IntEnum):
    EMPTY = 0
    OPEN = 1
    WALL = 2


class InvalidMove(Exception):
    """Move is invalid"""


class HitWall(Exception):
    """Hit a wall"""


moves = {
    Direction.RIGHT: np.array([0, 1]),
    Direction.DOWN: np.array([1, 0]),
    Direction.LEFT: np.array([0, -1]),
    Direction.UP: np.array([-1, 0]),
}


# mapping from ASCII art map characters to integers
d = {" ": 0, ".": 1, "#": 2}

with open("/home/rgottula/devel/advent_of_code/2022/22/input") as f:
    map_rows = []
    for line in f:
        if "." in line:
            map_rows.append([d[c] for c in line.rstrip("\n")])
        elif line:
            path = line.rstrip("\n")

width = max(map(len, map_rows))
map_array = np.array([row + [0] * (width - len(row)) for row in map_rows])
face_size = int(np.sqrt(np.sum(map_array > 0) // 6))

# Part 1

# direction = Direction.RIGHT
# position = [0, np.argmax(map_array[0, :] == 1)]
# while True:
#     # Get next move magnitude
#     move_mag = re.search(r"\d+", path).group()
#     path = path[len(move_mag) :]
#     move_mag = int(move_mag)
#     # print(f"Attempting to move {move_mag} spaces {direction.name}")

#     if direction in [Direction.LEFT, Direction.RIGHT]:
#         pos = position[1]
#         row = map_array[position[0], :]
#     else:
#         pos = position[0]
#         row = map_array[:, position[1]]

#     offset = np.argmax(row > 0)
#     pos_trunc = pos - offset
#     row_trunc = row[row > 0]
#     row_dupe = np.tile(row_trunc, 2)
#     if np.all(row != 2):
#         max_move = float('inf')
#     elif direction in [Direction.RIGHT, Direction.DOWN]:
#         max_move = np.argmax(row_dupe[pos_trunc:] == 2) - 1
#     else:
#         pos_trunc_offset = pos_trunc + row_trunc.size
#         max_move = np.argmax(np.flip(row_dupe[:pos_trunc_offset]) == 2)

#     move_mag_limited = min(move_mag, max_move)

#     if direction in [Direction.LEFT, Direction.UP]:
#         move = -move_mag_limited
#     else:
#         move = move_mag_limited
#     pos_trunc_new = (pos_trunc + move) % row_trunc.size

#     if direction in [Direction.LEFT, Direction.RIGHT]:
#         position[1] = pos_trunc_new + offset
#     else:
#         position[0] = pos_trunc_new + offset

#     # print(f'End position: {position}')

#     if not path:
#         break

#     direction = Direction((direction + 1 if path[0] == 'R' else direction - 1) % 4)
#     path = path[1:]


# password = 1000 * (position[0] + 1) + 4 * (position[1] + 1) + direction
# print(password)

# print(f"Row: {position[0] + 1}")
# print(f"Column: {position[1] + 1}")
# print(f"Direction: {direction}")






# ====================== Part 2 ========================
#
# Okay part 2 is quite interesting. I think my existing technique ought to
# work with some modifications:
#
# 1) Rather than just taking a row or column, I'll need to assemble a 1D array
#    from different strips (rows and columns) of the input map.
#
#    I can't pre-assemble a 2D array that is 200x200 in size that just extends
#    around the sides of the cube, because the extensions from different
#    directions aren't the same. This is a shame because it would simplify
#    things pretty substantially.
#
#    Maybe there's some kind of stride-tricks wizardry I can use here? Hmmm.
#
# 2) After performing the move, I'll have an index into a 1-D array that I'll
#    need to transform back to x, y coordinates and a potentially modified
#    direction on the 2D map. That sounds trickier than the first part. Will
#    need to do quite a bit of bookkeeping.
#
# Hmm. This seems almost harder than just staying on the 2D map and making
# moves one square at a time brute-force style. The main problem with brute-
# force approach is figuring out where to go when the next move goes off the
# edge of the map. The algorithm to compute this for an arbitrary map is not
# obvious, but I think there may be some consistent rules:
#
# A) If the map continues in the direction of travel, proceed as usual.
# B) If a blank space is reached, try to turn the corner 90 degrees in either
#    direction.
# C) If (B) fails, or if the edge of the array is reached, shift over by [TBD]
#    rows or columns and re-enter the array moving in the opposite direction
#    until the first non-empty square is reached.
#
# I think these rules may work in the general case. I just need to fill in the
# details of how (B) and (C) work exactly.
#
# For (B) and (C), it's necessary to have some notion of how far the current
# position is from the edges of the current face. If we're in case (B) or (C)
# we already know we're at one of the edges, so we just need to know how far
# we are from the edges to our left and right (relative to the direction we
# want to move). Our position on the face is probably just the X and Y
# coordinates on the full map modulo the size of one cube face.
#
# Now for (B), let's say we are trying to move right and we are at the bottom
# row of the current face. First, let's try turning right. The next position
# will be one square right, one square down, and the new direction will be down.
# Test the new location to see if it's a valid place to move or a wall. If a
# wall, we're done. If it's a valid place to move, great! Move there. If it's
# empty, now try turning left instead. This will be a much wider turn, since
# we're at the bottom edge of this face. The next position will be N squares
# right and N squares up, where N is the size of the cube face. The new
# direction will be Up.
#
# For (C), first move perpendicular to the direction of travel by 2N squares in
# whichever direction stays within the bounds of the array (there should only
# be one direction that meets this criteria) where N is the size of the cube
# face. Then, modify the position index i in the same direction to get
#
# j = i - 2*(i % N) + N - 1
#
# which effectively reflects the position across the middle of that face.
# Set the direction to the opposite of the initial direction. Finally, move in
# that direction until the first non-empty square is found.
#
# Another idea is to apply the 2D map provided onto surfaces of a 3D array.
# This would be a better approximation of the geometry and should make it at
# least somewhat easier to figure out the position wrapping around the corners
# of the cube. However I would still need to flatten the cube back into a 2D
# map at the end in order to correctly determine the direction and position,
# unless I can think of some other trick. Furthermore, on the 3D cube it may
# be a bit trickier to think about directions in a way that is intuitive and
# consistent.


def in_array(pos: np.ndarray) -> bool:
    """Check if position is within array bounds or not.

    Can't rely on IndexError because negative indices are valid in Python.
    """
    if not (0 <= pos[0] < map_array.shape[0]):
        return False
    if not (0 <= pos[1] < map_array.shape[1]):
        return False
    return True


def try_turn(pos: np.ndarray, direction: Direction) -> tuple[np.ndarray, Direction]:
    """Try to make a 90 degree turn to another cube face.

    Assumes that the current position is at the edge of a cube face with the
    direction facing off the edge onto an empty square.

    Args:
        pos: Current position.
        direction: Current direction.

    Returns:
        Tuple containing the new position and direction after this move.

    Raises:
        InvalidMove when any move of this type goes off the map.
        HitWall when a turn hits a wall.
    """

    pos_face = pos % face_size

    if direction == Direction.RIGHT:
        # Turn left
        step = pos_face[0] + 1
        move = np.array([-step, step])
        pos_left = pos + move

        # Turn right
        step = face_size - pos_face[0]
        move = np.array([step, step])
        pos_right = pos + move

    elif direction == Direction.DOWN:

        # Turn left
        step = face_size - pos_face[1]
        move = np.array([step, step])
        pos_left = pos + move

        # Turn right
        step = pos_face[1] + 1
        move = np.array([step, -step])
        pos_right = pos + move

    elif direction == Direction.LEFT:

        # Turn left
        step = face_size - pos_face[0]
        move = np.array([step, -step])
        pos_left = pos + move

        # Turn right
        step = pos_face[0] + 1
        move = np.array([-step, -step])
        pos_right = pos + move

    elif direction == Direction.UP:

        # Turn left
        step = pos_face[1] + 1
        move = np.array([-step, -step])
        pos_left = pos + move

        # Turn right
        step = face_size - pos_face[1]
        move = np.array([-step, step])
        pos_right = pos + move

    else:
        raise ValueError(f'No such direction {direction}')


    # Try to turn left
    if in_array(pos_left):
        val = map_array[tuple(pos_left)]
        if val == MapVal.WALL:
            raise HitWall()
        elif val == MapVal.OPEN:
            return pos_left, Direction((direction - 1) % 4)

    # Try to turn right
    if in_array(pos_right):
        val = map_array[tuple(pos_right)]
        if val == MapVal.WALL:
            raise HitWall()
        elif val == MapVal.OPEN:
            return pos_right, Direction((direction + 1) % 4)

    # Neither turn is valid
    raise InvalidMove()


def try_wrap(pos: np.ndarray, direction: Direction) -> tuple[np.ndarray, Direction]:
    """Try to wrap around from one edge of the array to another.

    Assumes that the current position is at the edge of a cube face with the
    direction facing off the edge onto an empty square or off the end of the
    array.

    Args:
        pos: Current position.
        direction: Current direction.

    Returns:
        Tuple containing the new position and direction after this move.

    Raises:
        HitWall when the move can't be made due to a wall.
    """
    if direction in [Direction.UP, Direction.DOWN]:
        move = np.array([0, 2*face_size])
    else:
        move = np.array([2*face_size, 0])

    pos_trial = position + move
    if not in_array(pos_trial):
        pos_trial = position - move

    if direction in [Direction.UP, Direction.DOWN]:
        pos_trial[1] += -2*(pos_trial[1] % face_size) + face_size - 1
    else:
        pos_trial[0] += -2*(pos_trial[0] % face_size) + face_size - 1

    # Flip direction upside down
    direction = Direction((direction + 2) % 4)

    # Find the first non-empty square in the new direction
    if direction == Direction.RIGHT:
        pos_trial[1] = np.argmax(map_array[pos_trial[0], :] > 0)
    elif direction == Direction.DOWN:
        pos_trial[0] = np.argmax(map_array[:, pos_trial[1]] > 0)
    elif direction == Direction.LEFT:
        pos_trial[1] = map_array.shape[1] - np.argmax(np.flip(map_array[pos_trial[0], :]) > 0) - 1
    elif direction == Direction.UP:
        pos_trial[0] = map_array.shape[0] - np.argmax(np.flip(map_array[:, pos_trial[1]]) > 0) - 1

    if map_array[tuple(pos_trial)] == MapVal.WALL:
        raise HitWall()

    return pos_trial, direction


direction = Direction.RIGHT
position = np.array([0, np.argmax(map_array[0, :] == 1)])

while True:
    # Get next move magnitude
    move_mag = re.search(r"\d+", path).group()
    path = path[len(move_mag) :]
    moves_remaining = int(move_mag)

    print(f"At {position} attempting to move {move_mag} spaces {direction.name}")

    while moves_remaining:

        # find the next potential position on the map
        trial_pos = position + moves[direction]

        if not in_array(trial_pos):
            try:
                position, direction = try_wrap(position, direction)
            except HitWall:
                break
            else:
                moves_remaining -= 1
                continue

        val = map_array[tuple(trial_pos)]
        if val == MapVal.WALL:
            break
        elif val == MapVal.EMPTY:
            try:
                position, direction = try_turn(position, direction)
            except HitWall:
                break
            except InvalidMove:
                try:
                    position, direction = try_wrap(position, direction)
                except HitWall:
                    break
        else:
            position = trial_pos

        moves_remaining -= 1


    if not path:
        break

    direction = Direction((direction + 1 if path[0] == 'R' else direction - 1) % 4)
    path = path[1:]


password = 1000 * (position[0] + 1) + 4 * (position[1] + 1) + direction
print(password)

# print(f"Row: {position[0] + 1}")
# print(f"Column: {position[1] + 1}")
# print(f"Direction: {direction}")

import IPython

IPython.embed()



# Part 2 guesses:
#
# 1) 35313 -- too high. Found bug in handling of negative indices into array.
# 2) 161192 -- didn't submit, but presumably also too high
