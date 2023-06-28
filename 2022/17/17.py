#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/17
"""

from __future__ import annotations
import itertools
import numpy as np


# @profile
# def try_placement(chamber: np.ndarray, rock: np.ndarray, x: int, y: int) -> bool:
#     t1 = try_placement_1(chamber, rock, x, y)
#     t2 = try_placement_2(chamber, rock, x, y)
#     if t1 != t2:
#         print('mismatch!')
#         import IPython; IPython.embed()
#     return t1


# @profile
# def try_placement_1(chamber: np.ndarray, rock: np.ndarray, x: int, y: int) -> bool:
#     """Test rock placement. Returns True if placement is possible, false otherwise."""
#     c = chamber.copy()
#     try:
#         c[y:y + rock.shape[0], x:x+rock.shape[1]] |= rock
#     except ValueError:
#         # hit a wall
#         return False

#     if np.sum(c) != np.sum(chamber) + np.sum(rock):
#         # overlapping with an existing rock
#         return False

#     return True


# @profile
def try_placement(chamber: np.ndarray, rock: np.ndarray, x: int, y: int) -> bool:
    """Test rock placement. Returns True if placement is possible, false otherwise."""
    slice = chamber[y:y + rock.shape[0], x:x+rock.shape[1]]

    if slice.shape != rock.shape:
        # probably hit a wall
        return False

    slice += rock

    if np.any(slice > 1):
        # overlapping with an existing rock
        slice -= rock
        return False

    slice -= rock
    return True


def place(chamber: np.ndarray, rock: np.ndarray, x: int, y: int) -> None:
    chamber[y:y + rock.shape[0], x:x+rock.shape[1]] |= rock


def find_highest(chamber: np.ndarray) -> int:
    """Find the Y coordinate of the highest bit of rock in the chamber."""
    return chamber.shape[0] - np.argmax(np.flipud(np.logical_or.reduce(chamber, axis=1))) - 1


def find_new_highest(current_highest: int, rock: np.ndarray, rock_y: int) -> int:
    """Find the new highest rock coordinate"""
    rock_highest = rock_y + rock.shape[0] - 1
    return max(rock_highest, current_highest)


def show(chamber: np.ndarray) -> None:
    """Print out an ascii-art representation of the chamber contents."""
    for row in np.flipud(chamber):
        if np.any(row):
            print('|' + ''.join(['#' if c else '.' for c in row]) + '|')


def crosscorrelate(x: np.ndarray, y: np.ndarray, max_lags: int) -> np.ndarray:
    return np.correlate(x, y[max_lags//2:-max_lags//2], mode='valid')


def predict_height(rock_idx: int, heights: np.ndarray) -> int:

    # These constants are only correct for Part 2
    periodic_start_idx = 216  # sequence is aperiodic before this
    period = 1755
    period_height = 2747

    if rock_idx < periodic_start_idx:
        return heights[rock_idx]

    k = rock_idx - periodic_start_idx
    return (k // period) * period_height + heights[k % period + periodic_start_idx]


# @profile
def main():
    rocks = [
        np.array([
            [1, 1, 1, 1],
        ], dtype=int),

        np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=int),

        np.flipud(np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=int)),

        np.array([
            [1],
            [1],
            [1],
            [1],
        ], dtype=int),

        np.array([
            [1, 1],
            [1, 1],
        ], dtype=int),
    ]

    with open('input') as f:
        gas_jets_pattern = f.readline().strip()

    # this will iterate over the gas jet pattern forever, use next()
    gas_jet_iter = itertools.cycle(gas_jets_pattern)
    rock_iter = itertools.cycle(rocks)


    # make a very tall array, hope that it's tall enough for the entire puzzle
    # axis 0 is vertical, where index 0 is the floor
    # axis 1 is horizontal, where index 0 is against the left wall
    num_rocks_to_drop = 2022
    chamber_height = int(1.6 * num_rocks_to_drop)  # pile grows by about 1.5589 units per rock on average
    chamber_width = 7
    chamber = np.zeros((chamber_height, chamber_width), dtype=np.int8)

    # line the floor with "rock"
    chamber[0, :] = True

    tower_heights = [0]
    y_highest_rock = 0
    for idx in range(2022):
    # for idx in itertools.count(start=1):

        if idx % 100 == 0:
            print(f'Rock {idx}')

        if y_highest_rock >= chamber.shape[0] - 10:
            # TODO: Could double the size of the array each time we hit this
            print('About to hit ceiling of the chamber! Ending at rock number {idx}.')
            break

        rock = next(rock_iter)

        # starting coordinates of the rock's bottom-left cell
        x = 2  # two units from left wall
        y = y_highest_rock + 4  # leave gap of 3 vertical units

        while True:

            # gas jet left/right phase
            jet = next(gas_jet_iter)
            if jet == '>':
                x_trial = x + 1
            else:
                x_trial = x - 1

            if try_placement(chamber, rock, x_trial, y):
                x = x_trial

            if try_placement(chamber, rock, x, y - 1):
                y -= 1
            else:
                place(chamber, rock, x, y)
                y_highest_rock = find_new_highest(y_highest_rock, rock, y)
                tower_heights.append(y_highest_rock)
                # if idx < 10:
                #     print(f'\nPlaced rock {idx + 1}: ')
                #     show(chamber)
                break


    print(f'Tower of rocks is {find_highest(chamber)} units tall after rock 2022')

    part2 = predict_height(1000000000000, np.array(tower_heights))
    print(f'Tower of rocks is {part2} units tall after rock 1000000000000')


if __name__ == "__main__":
    main()
