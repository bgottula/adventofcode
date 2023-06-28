#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/9
"""

from typing import NamedTuple
import numpy as np

class Position(NamedTuple):
    x: int
    y: int

head = Position(0, 0)
tail = Position(0, 0)
tail_positions = {tail}

with open('input') as f:
    for line in f:
        direction = line[0]
        magnitude = int(line.strip()[2:])

        for _ in range(magnitude):

            # move head
            if direction == 'U':
                head = head._replace(y=head.y + 1)
            elif direction == 'D':
                head = head._replace(y=head.y - 1)
            elif direction == 'L':
                head = head._replace(x=head.x - 1)
            elif direction == 'R':
                head = head._replace(x=head.x + 1)
            else:
                raise RuntimeError(f'Unexpected direction {direction}')

            x_diff = head.x - tail.x
            y_diff = head.y - tail.y

            assert abs(x_diff) <= 2
            assert abs(y_diff) <= 2

            # move tail
            if x_diff == 2:
                tail = tail._replace(x=tail.x + 1, y=head.y)
            elif x_diff == -2:
                tail = tail._replace(x=tail.x - 1, y=head.y)
            elif y_diff == 2:
                tail = tail._replace(x=head.x, y=tail.y + 1)
            elif y_diff == -2:
                tail = tail._replace(x=head.x, y=tail.y - 1)

            assert abs(head.x - tail.x) <= 1
            assert abs(head.y - tail.y) <= 1

            tail_positions.add(tail)

print(len(tail_positions))


# -- part 2 --

from dataclasses import dataclass

@dataclass
class Position:
    x: int
    y: int

rope = [Position(0,0) for _ in range(10)]

head = rope[0]
tail = rope[-1]
tail_positions = {(tail.x, tail.y)}


def move_knot(knot_idx: int):
    assert knot_idx > 0

    leader = rope[knot_idx - 1]
    follower = rope[knot_idx]

    x_diff = leader.x - follower.x
    y_diff = leader.y - follower.y

    assert abs(x_diff) <= 2
    assert abs(y_diff) <= 2

    if abs(x_diff) <= 1 and abs(y_diff) <= 1:
        return

    if x_diff != 0:
        follower.x += np.sign(x_diff)
    if y_diff != 0:
        follower.y += np.sign(y_diff)

    assert abs(leader.x - follower.x) <= 1
    assert abs(leader.y - follower.y) <= 1


with open('input') as f:
    for line in f:
        direction = line[0]
        magnitude = int(line.strip()[2:])

        for _ in range(magnitude):

            # move head
            if direction == 'U':
                head.y += 1
            elif direction == 'D':
                head.y -= 1
            elif direction == 'L':
                head.x -= 1
            elif direction == 'R':
                head.x += 1
            else:
                raise RuntimeError(f'Unexpected direction {direction}')

            for knot_idx in range(1, 10):
                move_knot(knot_idx)

            tail_positions.add((tail.x, tail.y))

print(len(tail_positions))
