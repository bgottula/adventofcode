#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/21
"""


# I think this can be solved with a simple recursive algorithm.

# First, parse the file into some data structure. Perhaps a dict where the name
# is used as a key, and the value is a either an integer or a named tuple
# containing the names of the operands and the operator.
#
# Next, set up a recursion starting with root. Look up each operand by calling
# the recursive function, which returns an integer value. Then perform the
# operation on the two operands and return it.

from __future__ import annotations
import operator
import re
from typing import Callable, NamedTuple

class Operation(NamedTuple):
    operand1: str
    operator: Callable[[int, int], int]
    operand2: str


def op_lookup(char: str) -> Callable[[int, int], int]:
    if char == '+':
        return operator.add
    elif char == '-':
        return operator.sub
    elif char == '*':
        return operator.mul
    elif char == '/':
        return operator.floordiv
    else:
        raise ValueError(f'Unrecognized operator {char}')


input = {}
with open('input') as f:
    for line in f:
        key = line[:4]

        try:
            val = int(line[6:])
        except ValueError:
            s = re.search('([a-z]{4}) ([+\-*\/]) ([a-z]{4})', line[6:])
            val = Operation(
                s[1],
                op_lookup(s[2]),
                s[3]
            )

        input[key] = val



def compute(key: str) -> int:
    """Compute the value that the named monkey yells."""

    val = input[key]

    if isinstance(val, int):
        return val

    op1 = compute(val.operand1)
    op2 = compute(val.operand2)

    return val.operator(op1, op2)


class UnknownHumn(Exception):
    """We don't know what this human should shout yet"""


def compute2(key: str, res: int | None = None) -> int:
    """Compute the value that the named monkey yells."""

    val = input[key]

    if isinstance(val, int):
        if key == 'humn':
            if res is not None:
                print(f'humn shouts {res}!')
            else:
                raise UnknownHumn()
        return val

    try:
        op1 = compute2(val.operand1)
    except UnknownHumn:
        if res is None and key != 'root':
            # Nothing more we can do here quite yet
            raise

    try:
        op2 = compute2(val.operand2)
    except UnknownHumn:
        if res is None and key != 'root':
            # Nothing more we can do here quite yet
            raise

    try:
        return val.operator(op1, op2)
    except NameError:
        pass

    if 'op1' in locals():
        # compute op2 using op1 and res
        if key == 'root':
            op2 = op1
        elif val.operator == operator.add:
            op2 = res - op1
        elif val.operator == operator.sub:
            op2 = op1 - res
        elif val.operator == operator.mul:
            op2 = res // op1
        elif val.operator == operator.floordiv:
            op2 = op1 // res
        else:
            raise ValueError(f'Unknown operator {val.operator}')

        # Now we're passing the value *down* the tree
        compute2(val.operand2, op2)

    elif 'op2' in locals():
        # compute op1 using op2 and res
        if key == 'root':
            op1 = op2
        elif val.operator == operator.add:
            op1 = res - op2
        elif val.operator == operator.sub:
            op1 = res + op2
        elif val.operator == operator.mul:
            op1 = res // op2
        elif val.operator == operator.floordiv:
            op1 = res * op2
        else:
            raise ValueError(f'Unknown operator {val.operator}')

        # Now we're passing the value *down* the tree
        compute2(val.operand1, op1)

    else:
        raise RuntimeError('Neither op1 nor op2 are defined')


# Part 1
# root = compute('root')
# print(root)


# Part 2
compute2('root')


# import IPython; IPython.embed()
