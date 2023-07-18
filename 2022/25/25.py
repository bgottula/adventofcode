#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/25
"""

total = 0
values = []
with open("input") as f:
    for line in f:
        line = line.rstrip("\n")

        val = 0
        for idx, c in enumerate(reversed(line)):
            if c == "2":
                val += 5**idx * 2
            elif c == "1":
                val += 5**idx * 1
            elif c == "0":
                val += 5**idx * 0
            elif c == "-":
                val += 5**idx * -1
            elif c == "=":
                val += 5**idx * -2
            else:
                print("something wrong")

        # print(val)
        values.append(val)
        total += val


def dec2snafu(val: int) -> str:
    s = {
        2: "2",
        1: "1",
        0: "0",
        -1: "-",
        -2: "=",
    }

    result = ""
    place = 0
    while True:
        q = (val + 2) % 5 - 2
        val -= 5**place * q
        val //= 5
        result = s[q] + result
        if val == 0:
            break

    return result


# print(total)
print(dec2snafu(total))

import IPython

IPython.embed()
