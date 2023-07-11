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


# mapping from ASCII art map characters to integers
d = {" ": MapVal.EMPTY.value, ".": MapVal.OPEN.value, "#": MapVal.WALL.value}


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


def compute_face_size(map_array: np.ndarray) -> int:
    return int(np.sqrt(np.sum(map_array > 0) // 6))


def load_map(filename) -> tuple[np.ndarray, str]:

    with open(filename) as f:
        map_rows = []
        for line in f:
            if "." in line:
                map_rows.append([d[c] for c in line.rstrip("\n")])
            elif line:
                path = line.rstrip("\n")

    width = max(map(len, map_rows))
    map_array = np.array([row + [0] * (width - len(row)) for row in map_rows])

    return map_array, path


def do_part_1(map_array: np.ndarray, path: str) -> int:

    direction = Direction.RIGHT
    position = [0, np.argmax(map_array[0, :] == 1)]
    while True:
        # Get next move magnitude
        move_mag = re.search(r"\d+", path).group()
        path = path[len(move_mag) :]
        move_mag = int(move_mag)
        # print(f"Attempting to move {move_mag} spaces {direction.name}")

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
            max_move = float('inf')
        elif direction in [Direction.RIGHT, Direction.DOWN]:
            max_move = np.argmax(row_dupe[pos_trunc:] == 2) - 1
        else:
            pos_trunc_offset = pos_trunc + row_trunc.size
            max_move = np.argmax(np.flip(row_dupe[:pos_trunc_offset]) == 2)

        move_mag_limited = min(move_mag, max_move)

        if direction in [Direction.LEFT, Direction.UP]:
            move = -move_mag_limited
        else:
            move = move_mag_limited
        pos_trunc_new = (pos_trunc + move) % row_trunc.size

        if direction in [Direction.LEFT, Direction.RIGHT]:
            position[1] = pos_trunc_new + offset
        else:
            position[0] = pos_trunc_new + offset

        # print(f'End position: {position}')

        if not path:
            break

        direction = Direction((direction + 1 if path[0] == 'R' else direction - 1) % 4)
        path = path[1:]

    # print(f"Row: {position[0] + 1}")
    # print(f"Column: {position[1] + 1}")
    # print(f"Direction: {direction}")

    return 1000 * (position[0] + 1) + 4 * (position[1] + 1) + direction


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
#
# Well, turns out my rules A, B, and C are incomplete. There's at least one
# situation that this does not handle. On the example map, it's when heading
# off the left edge (middle four squares) or going off the bottom edge from the
# right-most four squares. The left-edge case isn't handled by any of the steps,
# and the bottom-edge case is handled by rule (C) but incorrectly. I'm
# struggling to think of what rule I can apply to handle this case.
#
#
# Another idea is to apply the 2D map provided onto surfaces of a 3D array.
# This would be a better approximation of the geometry and should make it at
# least somewhat easier to figure out the position wrapping around the corners
# of the cube. However I would still need to flatten the cube back into a 2D
# map at the end in order to correctly determine the direction and position,
# unless I can think of some other trick. Furthermore, on the 3D cube it may
# be a bit trickier to think about directions in a way that is intuitive and
# consistent.
#
#
# Backing up a bit, here's some geometry fundamentals.
#
# A cube has 11 "nets":
# https://math.stackexchange.com/a/4470796
#
# A polyhedron can be represented as a graph. The vertices and edges of the
# polyhedron are the same as the vertices and edges of the graph representation,
# the only difference is that the vertices can be moved around in the graph so
# it's a bit easier to look at. Typically a cube graph is shown with a small
# square inset within a larger square, where the vertices of the small square
# have edges to the corresponding vertices of the larger square. Kinda looks
# like a top-down view of a pyramid where the top has been flattened.
#
# Cut exactly 7 edges of the cube to form a net. This leaves 5 intact edges,
# which are the ones that are unfolded to flatten the cube into the 2D net.
# The *cut* edges must form a spanning tree, meaning that the graph with only
# the cut edges remaining can be drawn as a tree that includes all of the
# verticies.
#
#
# Observations:
# * The two halfs of a cut edge in the net are always an odd number of edges
#   apart around the perimeter of the net. For example, if I start with edge
#   AB, the paired edge AB will be 1, 3, 5, 7, 9, 11, or 13 edges away.
# * If the pairing edge is 1 edge away (it's the nearest neighbor), the pair
#   must form a right angle. Furthermore, the angle must be 90-degrees as
#   measured from the outside of the net (as opposed to 270-degrees).
# * If there are two edges in the net that are parallel, exactly 4 tiles apart,
#   and in the same row or column they always form a pair. There will be at
#   most one such pair in a net. They will always be exactly 7 edges apart.
# * To find all the edge pairings it seems to work best to start by identifying
#   all of the corner pairs and work out from those. Typically the next edge
#   pair will be just beyond a corner pair, but check the two rules below to
#   confirm.
#
# Some conjectures (true for all cube nets):
# * If the pair is 3, 7, or 11 edges apart, they must be the same orientation
#   (both vertical or both horizontal).
# * If the pair is 1, 5, 9, or 13 edges apart, they must be perpendicular
#   orientations.
#
# It may be easiest to start with the corner pairs and then work out from those.
# without first identifying the corner pairs it may be more difficult to
# correctly determine what edge pairs with some arbitrarily chosen edge.


def in_array(map_array: np.ndarray, pos: np.ndarray) -> bool:
    """Check if position is within array bounds or not.

    Can't rely on IndexError because negative indices are valid in Python.
    """
    if not (0 <= pos[0] < map_array.shape[0]):
        return False
    if not (0 <= pos[1] < map_array.shape[1]):
        return False
    return True


def test_in_array():
    map_array = np.reshape(np.arange(100), (5,20))
    assert in_array(map_array, np.array([3, 2]))
    assert in_array(map_array, np.array([0, 0]))
    assert in_array(map_array, np.array([4, 19]))
    assert not in_array(map_array, np.array([5, 19]))
    assert not in_array(map_array, np.array([4, 20]))
    assert not in_array(map_array, np.array([-1, 0]))
    assert not in_array(map_array, np.array([0, -1]))



def try_turn(
        map_array: np.ndarray,
        pos: np.ndarray,
        direction: Direction,
    ) -> tuple[np.ndarray, Direction]:
    """Try to make a 90 degree turn to another cube face.

    Assumes that the current position is at the edge of a cube face with the
    direction facing off the edge onto an empty square.

    Args:
        map_array: Map.
        pos: Current position.
        direction: Current direction.

    Returns:
        Tuple containing the new position and direction after this move.

    Raises:
        InvalidMove when any move of this type goes off the map.
        HitWall when a turn hits a wall.
    """

    face_size = compute_face_size(map_array)
    pos_face = pos % face_size

    # Should be on an open square
    assert map_array[tuple(pos)] == MapVal.OPEN

    # Shouldn't be trying to make turns if we're not at the edge of a face
    ahead = pos + moves[direction]
    assert in_array(map_array, ahead)
    assert map_array[tuple(ahead)] == MapVal.EMPTY

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
    if in_array(map_array, pos_left):
        val = map_array[tuple(pos_left)]
        if val == MapVal.WALL:
            raise HitWall()
        elif val == MapVal.OPEN:
            return pos_left, Direction((direction - 1) % 4)

    # Try to turn right
    if in_array(map_array, pos_right):
        val = map_array[tuple(pos_right)]
        if val == MapVal.WALL:
            raise HitWall()
        elif val == MapVal.OPEN:
            return pos_right, Direction((direction + 1) % 4)

    # Neither turn is valid
    raise InvalidMove()


def test_try_turn():
    map_array, _ = load_map("example")

    # Try some right turns
    position, direction = try_turn(map_array, np.array([7, 11]), Direction.RIGHT)
    assert np.all(position == np.array([8, 12]))
    assert direction == Direction.DOWN

    position, direction = try_turn(map_array, np.array([6, 11]), Direction.RIGHT)
    assert np.all(position == np.array([8, 13]))
    assert direction == Direction.DOWN

    position, direction = try_turn(map_array, np.array([5, 11]), Direction.RIGHT)
    assert np.all(position == np.array([8, 14]))
    assert direction == Direction.DOWN

    # this starts on a Wall, which is illegal
    try:
        try_turn(map_array, np.array([4, 11]), Direction.RIGHT)
    except AssertionError:
        pass
    else:
        assert False

    # Try some left turns back the other direction
    position, direction = try_turn(map_array, np.array([8, 12]), Direction.UP)
    assert np.all(position == np.array([7, 11]))
    assert direction == Direction.LEFT

    position, direction = try_turn(map_array, np.array([8, 13]), Direction.UP)
    assert np.all(position == np.array([6, 11]))
    assert direction == Direction.LEFT

    position, direction = try_turn(map_array, np.array([8, 14]), Direction.UP)
    assert np.all(position == np.array([5, 11]))
    assert direction == Direction.LEFT

    # this one should hit a wall
    try:
        try_turn(map_array, np.array([8, 15]), Direction.UP)
    except HitWall:
        pass
    else:
        assert False  # expected an exception but didn't get one

    # Try a right turn facing left
    position, direction = try_turn(map_array, np.array([10, 8]), Direction.LEFT)
    assert np.all(position == np.array([7, 5]))
    assert direction == Direction.UP

    # Try a left turn facing down
    position, direction = try_turn(map_array, np.array([7, 4]), Direction.DOWN)
    assert np.all(position == np.array([11, 8]))
    assert direction == Direction.RIGHT

    # Try some invalid turns
    try:
        try_turn(map_array, np.array([4, 1]), Direction.UP)
    except InvalidMove:
        pass
    else:
        assert False

    try:
        try_turn(map_array, np.array([7, 1]), Direction.DOWN)
    except InvalidMove:
        pass
    else:
        assert False

    try:
        try_turn(map_array, np.array([1, 11]), Direction.RIGHT)
    except InvalidMove:
        pass
    else:
        assert False

    # This goes off the top of the map, which is an invalid starting position
    try:
        try_turn(map_array, np.array([0, 8]), Direction.UP)
    except AssertionError:
        pass
    else:
        assert False


def try_wrap(
        map_array: np.ndarray,
        pos: np.ndarray,
        direction: Direction,
    ) -> tuple[np.ndarray, Direction]:
    """Try to wrap around from one edge of the array to another.

    Assumes that the current position is at the edge of a cube face with the
    direction facing off the edge onto an empty square or off the end of the
    array.

    Args:
        map_array: Map.
        position: Current position.
        direction: Current direction.

    Returns:
        Tuple containing the new position and direction after this move.

    Raises:
        HitWall when the move can't be made due to a wall.
        InvalidMove if this move can't be performed.
    """
    face_size = compute_face_size(map_array)

    # Should be on an open square
    assert map_array[tuple(pos)] == MapVal.OPEN

    # Shouldn't be trying wrap around if we're not at the edge of a face
    ahead = pos + moves[direction]
    if in_array(map_array, ahead):
        assert map_array[tuple(ahead)] == MapVal.EMPTY

    if direction in [Direction.UP, Direction.DOWN]:
        move = np.array([0, 2*face_size])
    else:
        move = np.array([2*face_size, 0])

    pos_trial = pos + move
    if not in_array(map_array, pos_trial):
        pos_trial = pos - move
        if not in_array(map_array, pos_trial):
            raise InvalidMove

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


def test_try_wrap():
    map_array, _ = load_map("example")

    # Try going off the top edge
    try:
        try_wrap(map_array, np.array([0, 8]), Direction.UP)
    except HitWall:
        pass
    else:
        assert False

    position, direction = try_wrap(map_array, np.array([0, 9]), Direction.UP)
    assert np.all(position == np.array([4, 2]))
    assert direction == Direction.DOWN

    position, direction = try_wrap(map_array, np.array([0, 10]), Direction.UP)
    assert np.all(position == np.array([4, 1]))
    assert direction == Direction.DOWN

    # Starts on a wall, which is invalid
    try:
        try_wrap(map_array, np.array([0, 11]), Direction.UP)
    except AssertionError:
        pass
    else:
        assert False

    # Try going off the left edge. I think this may be a case that isn't
    # handled at all, come to think of it. Let's just uh...see what happens.
    position, direction = try_wrap(map_array, np.array([4, 0]), Direction.LEFT)

    import IPython; IPython.embed()



def do_part_2(map_array: np.ndarray, path: str) -> int:

    direction = Direction.RIGHT
    position = np.array([0, np.argmax(map_array[0, :] == 1)])

    while True:
        # Get next move magnitude
        move_mag = re.search(r"\d+", path).group()
        path = path[len(move_mag) :]
        moves_remaining = int(move_mag)

        # print(f"At {position} attempting to move {move_mag} spaces {direction.name}")

        while moves_remaining:

            # find the next potential position on the map
            trial_pos = position + moves[direction]

            if not in_array(map_array, trial_pos):
                try:
                    position, direction = try_wrap(map_array, position, direction)
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
                    position, direction = try_turn(map_array, position, direction)
                except HitWall:
                    break
                except InvalidMove:
                    try:
                        position, direction = try_wrap(map_array, position, direction)
                    except HitWall:
                        break
            else:
                position = trial_pos

            moves_remaining -= 1

        if not path:
            break

        direction = Direction((direction + 1 if path[0] == 'R' else direction - 1) % 4)
        path = path[1:]

    # print(f"Row: {position[0] + 1}")
    # print(f"Column: {position[1] + 1}")
    # print(f"Direction: {direction}")

    return 1000 * (position[0] + 1) + 4 * (position[1] + 1) + direction


def do_part_2_hack(map_array: np.ndarray, path: str) -> int:
    """Hacky approach that is hard-coded to work for the input map only.

    Unfortunately not a general solution but it works. The general solutions
    I was able to think of would have been very tedious and time consuming
    to implement. There are likely far more elegant general solutions that I
    haven't thought of.
    """

    direction = Direction.RIGHT
    position = np.array([0, np.argmax(map_array[0, :] == 1)])

    while True:
        # Get next move magnitude
        move_mag = re.search(r"\d+", path).group()
        path = path[len(move_mag) :]
        moves_remaining = int(move_mag)

        # print(f"At {position} attempting to move {move_mag} spaces {direction.name}")

        while moves_remaining:

            # find the next potential position on the map
            trial_pos = position + moves[direction]

            # A bunch of hard-coded logic to handle going off the edge of the map
            if not in_array(map_array, trial_pos):
                y, x = trial_pos
                if direction == Direction.UP:
                    assert 50 <= x < 150
                    assert y == -1
                    if 50 <= x < 100:
                        offset = x - 50
                        trial_pos_2 = np.array([offset + 150, 0])
                        if map_array[tuple(trial_pos_2)] == MapVal.WALL:
                            break
                        position = trial_pos_2
                        direction = Direction.RIGHT
                    else:
                        offset = x - 100
                        trial_pos_2 = np.array([199, offset])
                        if map_array[tuple(trial_pos_2)] == MapVal.WALL:
                            break
                        position = trial_pos_2
                        direction = Direction.UP
                elif direction == Direction.RIGHT:
                    assert 0 <= y < 50
                    try:
                        position, direction = try_wrap(map_array, position, direction)
                    except HitWall:
                        break
                elif direction == Direction.DOWN:
                    assert 0 <= x < 50
                    offset = x
                    trial_pos_2 = np.array([0, offset + 100])
                    if map_array[tuple(trial_pos_2)] == MapVal.WALL:
                        break
                    position = trial_pos_2
                    direction = Direction.DOWN
                else:
                    assert 100 <= y < 200
                    if 100 <= y < 150:
                        try:
                            position, direction = try_wrap(map_array, position, direction)
                        except HitWall:
                            break
                    else:
                        offset = y - 150
                        trial_pos_2 = np.array([0, offset + 50])
                        if map_array[tuple(trial_pos_2)] == MapVal.WALL:
                            break
                        position = trial_pos_2
                        direction = Direction.DOWN

                moves_remaining -= 1
                continue

            val = map_array[tuple(trial_pos)]
            if val == MapVal.WALL:
                break
            elif val == MapVal.EMPTY:
                try:
                    position, direction = try_turn(map_array, position, direction)
                except HitWall:
                    break
                except InvalidMove:
                    try:
                        position, direction = try_wrap(map_array, position, direction)
                    except HitWall:
                        break
            else:
                position = trial_pos

            moves_remaining -= 1

        if not path:
            break

        direction = Direction((direction + 1 if path[0] == 'R' else direction - 1) % 4)
        path = path[1:]

    # print(f"Row: {position[0] + 1}")
    # print(f"Column: {position[1] + 1}")
    # print(f"Direction: {direction}")

    return 1000 * (position[0] + 1) + 4 * (position[1] + 1) + direction


def make_pairing_arrays(map_array: np.ndarray):
    """Initial attempt to find edge pairs in any cube net"""
    face_size = compute_face_size(map_array)
    shrunken_map = map_array[::face_size, ::face_size] > 0


def main():

    # test_in_array()
    # test_try_turn()
    # test_try_wrap()

    map_array, path = load_map("/home/rgottula/devel/advent_of_code/2022/22/input")

    # part1_password = do_part_1(map_array, path)
    # print(f'Part 1 password: {part1_password}')

    # part2_password = do_part_2(map_array, path)
    # print(f'Part 2 password: {part2_password}')

    part2_password = do_part_2_hack(map_array, path)
    print(f'Part 2 password: {part2_password}')


if __name__ == "__main__":
    main()



# Part 2 guesses:
#
# 1) 35313 -- too high. Found bug in handling of negative indices into array.
# 2) 161192 -- didn't submit, but presumably also too high
