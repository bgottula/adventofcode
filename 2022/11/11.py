#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/11
"""

from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np

@dataclass
class Monkey:
    items: list[int]
    operation: str
    divisor: int
    recipients: tuple[int, int]
    monkeys: list[Monkey]
    num_inspected: int = 0

    def test(self, item) -> bool:
        return (item % self.divisor) == 0

    def inspect_items(self):
        for item in self.items:
            self.num_inspected += 1
            # print(f'  Monkey inspects item with a worry level of {item}.')
            old = item
            item = eval(self.operation)
            # print(f'    {self.operation} applied, new level is {item}.')

            # -- part 1 --
            # item = item // 3

            # -- part 2 --
            item %= wrap

            # print(f'    Monkey gets bored with item. Worry level is divided by 3 to {item}.')
            self.monkeys[self.recipients[not self.test(item)]].items.append(item)
            # print(f'    Item is thrown to monkey {self.recipients[not self.test(item)]}.')
        self.items = []

monkeys = []

with open('input') as f:
    lines = f.readlines()

line_idx = 0
divisors = set()
while True:
    items = eval('[' + lines[line_idx + 1][18:] + ']')
    operation = lines[line_idx + 2][19:].strip()
    m = re.search('.*Test: divisible by ([0-9]+)', lines[line_idx + 3])
    divisor = int(m.group(1))
    divisors.add(divisor)
    m = re.search('.*If true: throw to monkey ([0-9]+)', lines[line_idx + 4])
    recipient_true = int(m.group(1))
    m = re.search('.*If false: throw to monkey ([0-9]+)', lines[line_idx + 5])
    recipient_false = int(m.group(1))
    recipients = (recipient_true, recipient_false)

    monkeys.append(Monkey(items, operation, divisor, recipients, []))

    print(f'Added monkey {monkeys[-1]}')

    line_idx += 7
    if line_idx + 1 >= len(lines):
        break

wrap = np.prod(list(divisors))

for monkey in monkeys:
    monkey.monkeys = monkeys

for round_idx in range(10000):
    for idx, monkey in enumerate(monkeys):
        # print(f'Monkey {idx}:')
        monkey.inspect_items()

    # print(f'\nAfter round {round_idx + 1}:')
    # for monkey in monkeys:
    #     print(monkey.items)

    # print(f'After round {round_idx + 1}: {[monkey.num_inspected for monkey in monkeys]}')

    if (round_idx + 1) % 100 == 0:
        print(f'Round {round_idx + 1} ({round_idx / 10000:.2%})')

    if (round_idx + 1) in [1, 20, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        print(f'After round {round_idx + 1}: {[monkey.num_inspected for monkey in monkeys]}')

inspections = sorted([monkey.num_inspected for monkey in monkeys])
print()
print(inspections[-2] * inspections[-1])
