#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/24
"""


from __future__ import annotations
from dataclasses import dataclass
import enum
import numpy as np


class Direction(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


moves = {
    Direction.UP: np.array([-1, 0]),
    Direction.DOWN: np.array([+1, 0]),
    Direction.LEFT: np.array([0, -1]),
    Direction.RIGHT: np.array([0, +1]),
}


@dataclass
class Blizzard:
    position: np.ndarray
    direction: Direction


class MapVal(enum.IntEnum):
    BLIZZARD_UP = Direction.UP.value
    BLIZZARD_DOWN = Direction.DOWN.value
    BLIZZARD_LEFT = Direction.LEFT.value
    BLIZZARD_RIGHT = Direction.RIGHT.value
    OPEN = enum.auto()
    WALL = enum.auto()
    ELF = enum.auto()


def load_map(filename) -> np.ndarray:

    # mapping from ASCII art map characters to integers
    ascii_to_mapval = {
        "^": MapVal.BLIZZARD_UP.value,
        "v": MapVal.BLIZZARD_DOWN.value,
        "<": MapVal.BLIZZARD_LEFT.value,
        ">": MapVal.BLIZZARD_RIGHT.value,
        ".": MapVal.OPEN.value,
        "#": MapVal.WALL.value,
        "E": MapVal.ELF.value,
    }

    with open(filename) as f:
        map_rows = []
        for line in f:
            map_rows.append([ascii_to_mapval[c] for c in line.rstrip("\n")])

    width = max(map(len, map_rows))
    map_array = np.array([row + [0] * (width - len(row)) for row in map_rows])

    return map_array


def get_blizzard_list(map_array: np.ndarray) -> list[Blizzard]:
    blizzards = []
    for d in Direction:
        blizzards += [Blizzard(pos, d) for pos in np.argwhere(map_array == d.value)]
    return blizzards


def make_binary_map(blizzards: list[Blizzard], map_shape: tuple[int, int]) -> np.ndarray:
    """Create a boolean map array where True elements are open spaces."""

    # Create the base map
    map_array = np.ones((map_shape[0] + 1, map_shape[1]), dtype=bool)

    # Add walls
    map_array[0, :] = False
    map_array[-1, :] = False
    map_array[-2, :] = False
    map_array[:, 0] = False
    map_array[:, -1] = False

    # Open the entrance and exit
    map_array[0, 1] = True
    map_array[-2, -2] = True

    # Add all the blizzards
    for blizzard in blizzards:
        map_array[tuple(blizzard.position)] = False

    return map_array


def make_full_map(
        blizzards: list[Blizzard],
        elf_pos: np.ndarray | None,
        map_shape: tuple[int, int]
    ) -> np.ndarray:
    """Re-create the full map array."""

    # Create the base map
    map_array = MapVal.OPEN.value * np.ones(map_shape, dtype=int)

    # Add walls
    map_array[0, :] = MapVal.WALL.value
    map_array[-1, :] = MapVal.WALL.value
    map_array[:, 0] = MapVal.WALL.value
    map_array[:, -1] = MapVal.WALL.value

    # Open the entrance and exit
    map_array[0, 1] = MapVal.OPEN.value
    map_array[-1, -2] = MapVal.OPEN.value

    if elf_pos is not None:
        map_array[tuple(elf_pos)] = MapVal.ELF.value

    # Add all the blizzards
    for blizzard in blizzards:
        val = map_array[tuple(blizzard.position)]
        if val == MapVal.OPEN:
            map_array[tuple(blizzard.position)] = blizzard.direction.value
        elif val in [MapVal.BLIZZARD_UP, MapVal.BLIZZARD_DOWN, MapVal.BLIZZARD_LEFT, MapVal.BLIZZARD_RIGHT]:
            map_array[tuple(blizzard.position)] = max(MapVal).value + 1
        elif val > max(MapVal):
            map_array[tuple(blizzard.position)] += 1
        else:
            raise RuntimeError(f'Illegal map value encountered: {MapVal(val).name}')

    return map_array


def mapval_to_char(mapval: MapVal) -> str:
    mapval_to_ascii = {
        MapVal.BLIZZARD_UP.value: "^",
        MapVal.BLIZZARD_DOWN.value: "v",
        MapVal.BLIZZARD_LEFT.value: "<",
        MapVal.BLIZZARD_RIGHT.value: ">",
        MapVal.OPEN.value: ".",
        MapVal.WALL.value: "#",
        MapVal.ELF.value: "E",
    }
    try:
        return mapval_to_ascii[mapval]
    except KeyError:
        return f'{mapval - max(MapVal) + 1}'


def print_map(
        blizzards: list[Blizzard],
        elf_pos: np.ndarray | None,
        map_shape: tuple[int, int],
    ) -> None:
    map_array = make_full_map(blizzards, elf_pos, map_shape)
    for row in map_array:
        print(''.join([mapval_to_char(v) for v in row]))


def move_blizzard(blizzard: Blizzard, map_shape: tuple[int, int]) -> Blizzard:
    new_pos = blizzard.position + moves[blizzard.direction]
    # wrap around to the other side if needed
    new_pos = (new_pos - 1) % (np.array(map_shape) - 2) + 1
    return Blizzard(new_pos, blizzard.direction)


def move_blizzards(
        blizzards: list[Blizzard],
        map_shape: tuple[int, int]
    ) -> list[Blizzard]:
    """Move the blizzards around the map."""
    return [move_blizzard(b, map_shape) for b in blizzards]


def solveit(
        blizzards: list[Blizzard],
        map_shape: tuple[int, int],
        forward: bool,
    ) -> tuple[int, list[Blizzard]]:

    if forward:
        elves = {(0, 1)}
    else:
        elves = {(map_shape[0] - 1, map_shape[1] - 2)}
    moves = np.array([
        [-1, +1, +0, +0, +0],
        [+0, +0, -1, +1, +0],
    ])

    minutes = 0
    while True:
        minutes += 1
        blizzards = move_blizzards(blizzards, map_shape)
        map_array = make_binary_map(blizzards, map_shape)
        elves_new = set()
        for elf in elves:
            potential_elves = np.array((elf,)).T + moves
            allowed = map_array[tuple(potential_elves)]
            for new_elf in potential_elves[:, allowed].T:
                if forward and new_elf[0] == (map_shape[0] - 1):
                    return minutes, blizzards
                elif not forward and new_elf[0] == 0:
                    return minutes, blizzards
                elves_new |= {tuple(new_elf)}
        elves = elves_new
        print(f'minutes: {minutes} num elves: {len(elves)} total map size: {map_shape[0] * map_shape[1]}')


def main():
    map_array = load_map("input")
    blizzards = get_blizzard_list(map_array)

    # Part 1
    minutes_1, blizzards = solveit(blizzards, map_array.shape, forward=True)
    print(f'Part 1: {minutes_1}')
    minutes_2, blizzards = solveit(blizzards, map_array.shape, forward=False)
    minutes_3, _ = solveit(blizzards, map_array.shape, forward=True)
    minutes_total = minutes_1 + minutes_2 + minutes_3
    print(f'Part 2: {minutes_total}')

    # print('Initial Map')
    # print_map(blizzards, None, map_array.shape)
    # for minute in range(5):
    #     blizzards = move_blizzards(blizzards, map_array.shape)
    #     print(f'\nAfter {minute + 1} minutes')
    #     print_map(blizzards, None, map_array.shape)

    import IPython; IPython.embed()


if __name__ == "__main__":
    main()


# Load the map
# Generate a list or array containing all of the blizzard positions and directions
# Write a function that updates the list of blizzard positions and directions one step
# Write a function that takes a list of blizzard positions and turns it into a binary
#    map array showing spaces that are open and spaces that are blocked, either by a
#    blizzard or a wall.
# Write a function that does a breadth-first search to find the route through the map.
#    At each new step, generate the map of blocked and free locations for that minute in
#    time. Make sure to do this operation only once for each new step.
#
# How to do the breadth-first search:
# * Keep a list of all positions the elf could be in at this time step.
# * Update the blizzards on the map.
# * For each elf position in the list, determine what new positions are valid
#       on the next time step and add those to a new list.
# * Iterate until one of the elves lands on the exit tile.
#
# Optimizations:
# * The blizzard positions are likely periodic with a period that's related to
#      the shape of the map. For example, if the map was 5x5 I'd expect the
#      blizzard positions to be periodic with a period of 5 minutes. Could
#      leverage this to cut down on blizzard movement computation.
