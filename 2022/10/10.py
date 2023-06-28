#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/10
"""

import re

X = 1
X_history = [X]
strength = [0]
display = []

def show_display():
    assert len(display) == 240
    for col in range(0, 240, 40):
        print(''.join(display[col:col+40]))

def addx(V):
    global X
    noop()
    noop()
    X += V

def noop():
    cycle = len(strength)
    strength.append(cycle * X)
    X_history.append(X)
    column = (cycle - 1) % 40
    display.append('#' if abs(column - X) <= 1 else '.')

with open('input') as f:
    for line in f:
        m = re.search('([a-z]+)', line)
        instruction = m.group(1)

        if instruction == 'noop':
            noop()
        elif instruction == 'addx':
            m = re.search('[a-z]+ (-?[0-9]+)', line)
            operand = int(m.group(1))
            addx(operand)
        else:
            raise RuntimeError(f'Unknown instruction {instruction}')

# for idx, (x, s) in enumerate(zip(X_history, strength)):
#     print(f'{idx:6d} {x:6d} {s:6d}')

print(sum(strength[20:221:40]))

show_display()
