#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/23
"""

from __future__ import annotations
import enum
import numpy as np


class MapVal(enum.IntEnum):
    OPEN = 0
    ELF = 1


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3


look_vectors = {
    Direction.NORTH: np.array([[-1, -1, -1], [-1, 0, +1]]),
    Direction.SOUTH: np.array([[+1, +1, +1], [-1, 0, +1]]),
    Direction.WEST: np.array([[-1, 0, +1], [-1, -1, -1]]),
    Direction.EAST: np.array([[-1, 0, +1], [+1, +1, +1]]),
}

all_neighbors = np.array([
    [-1, -1, -1, 0, 0, +1, +1, +1],
    [-1, 0, +1, -1, +1, -1, 0, +1]
])

moves = {
    Direction.NORTH: np.array([-1, 0]),
    Direction.SOUTH: np.array([+1, 0]),
    Direction.WEST: np.array([0, -1]),
    Direction.EAST: np.array([0, +1]),
}


def load_map(filename) -> np.ndarray:

    # mapping from ASCII art map characters to integers
    ascii_to_mapval = {".": MapVal.OPEN.value, "#": MapVal.ELF.value}

    with open(filename) as f:
        map_rows = []
        for line in f:
            map_rows.append([ascii_to_mapval[c] for c in line.rstrip("\n")])

    width = max(map(len, map_rows))
    map_array = np.array([row + [0] * (width - len(row)) for row in map_rows])

    return map_array


def expand_map(map_array: np.ndarray) -> np.ndarray:
    """Roughly double the size of the map in each dimension."""
    shape = map_array.shape
    return np.pad(map_array, ((shape[0] // 2, shape[0] // 2), (shape[1] // 2, shape[1] // 2)))


def pad_map(map_array: np.ndarray) -> np.ndarray:
    """Expand map as needed such that no elf is on the edge."""
    return np.pad(
        map_array,
        (
            (int(np.any(map_array[0, :])), int(np.any(map_array[-1, :]))),
            (int(np.any(map_array[:, 0])), int(np.any(map_array[:, -1]))),
        )
    )


def crop_map(map_array: np.ndarray) -> np.ndarray:
    """Crop down to smallest size that includes all elves."""
    coords = np.argwhere(map_array)
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    return map_array[x_min:x_max+1, y_min:y_max+1]


def print_map(map_array: np.ndarray) -> None:

    mapval_to_ascii = {MapVal.OPEN.value: ".", MapVal.ELF.value: "#"}
    map_array = crop_map(map_array)
    for row in map_array:
        print(''.join([mapval_to_ascii[c] for c in row]))


def check_example_round(map_array: np.ndarray, round_num: int):
    """Check the map after a round against the examples."""
    try:
        reference_map = load_map(f'example_round_{round_num+1:02d}')
    except FileNotFoundError:
        return
    reference_map = crop_map(reference_map)
    map_array = crop_map(map_array)
    if not np.all(map_array == reference_map):
        print(f'Round {round_num + 1} does not match example!')


def check_for_neighbors(
        map_array: np.ndarray,
        elf: np.ndarray,
        offsets: np.ndarray
    ) -> bool:
    look_indices = elf[..., np.newaxis] + offsets
    inspected_tiles = map_array[tuple(look_indices)]
    return np.sum(inspected_tiles) > 0


def solve(map_array: np.ndarray, max_rounds: int | None = None) -> tuple[int, int]:

    directions = np.arange(4)
    map_array = crop_map(map_array)
    rounds_completed = 0

    while True:

        map_array = pad_map(map_array)

        current_locations = np.argwhere(map_array)
        proposed_locations = []
        for elf in current_locations:

            # default move is to stay put
            proposed_locations.append(tuple(elf))

            if not check_for_neighbors(map_array, elf, all_neighbors):
                # no neighbors, no move
                continue

            for dir in directions:
                if not check_for_neighbors(map_array, elf, look_vectors[dir]):
                    proposed_locations[-1] = tuple(elf + moves[dir])
                    break

        proposed_locations = np.array(proposed_locations)

        # resolve conflicts to generate final locations of all elves
        final_locations = []
        for current, proposed in zip(current_locations, proposed_locations):
            if np.sum(np.all(proposed == proposed_locations, axis=1)) > 1:
                final_locations.append(current)
            else:
                final_locations.append(proposed)
        final_locations = np.array(final_locations)

        # update the map
        map_array[:] = MapVal.OPEN.value
        map_array[final_locations[:,0], final_locations[:,1]] = MapVal.ELF.value

        directions = np.roll(directions, -1)

        # print(f'== End of Round {rounds_completed + 1} ==')
        # print_map(map_array)

        # check_example_round(map_array, rounds_completed)

        rounds_completed += 1

        if max_rounds is not None and rounds_completed >= max_rounds:
            break

        if np.all(final_locations == current_locations):
            break

    return np.sum(crop_map(map_array) == 0), rounds_completed


def main():
    map_array = load_map("input")

    # Part 1
    num_empty, _ = solve(map_array, 10)
    print(num_empty)

    # Part 2 (takes a few minutes)
    _, num_rounds = solve(map_array)
    print(num_rounds)

    import IPython; IPython.embed()

if __name__ == "__main__":
    main()
