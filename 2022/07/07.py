#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/7
"""

from __future__ import annotations
import re
from typing import NamedTuple, Optional


class File(NamedTuple):
    name: str
    size: int


class Dir:
    def __init__(self, name: str, parent: Optional[Dir] = None):
        self.name = name
        self.parent = parent if parent is not None else self
        self.files = {}
        self.subdirs = {}

    def add_subdir(self, name: str) -> Dir:
        if name not in self.subdirs:
            self.subdirs[name] = Dir(name, self)
        return self.subdirs[name]

    def add_file(self, name: str, size: int) -> File:
        if name not in self.files:
            self.files[name] = File(name, size)
        return self.files[name]

    def compute_size(self) -> int:
        size = 0
        for file in self.files.values():
            size += file.size
        for subdir in self.subdirs.values():
            size += subdir.compute_size()
        return size

    def print(self, prefix: str = '') -> None:
        print(f'{prefix}- {self.name} (dir)')
        for subdir in self.subdirs.values():
            subdir.print(prefix=prefix + '  ')
        for file in self.files.values():
            print(f'{prefix}  - {file.name} (file, size={file.size})')

root = Dir('/')
cd = root
all_dirs = {root}
with open('input') as f:
    for line in f:
        if line[0] == '$':  # shell command
            cmd = line[2:4]
            if cmd == 'cd':
                dirname = line.strip()[5:]
                if dirname == '/':
                    cd = root
                if dirname == '..':
                    cd = cd.parent
                else:
                    cd = cd.add_subdir(dirname)
                    all_dirs.add(cd)
            elif cmd == 'ls':
                pass
            else:
                raise RuntimeError(f'Unexpected command {cmd}')
        else:  # ls output
            if line[0:3] == 'dir':
                cd.add_subdir(line.strip()[4:])
            else:
                m = re.search('([0-9]+) (.*)', line)
                size = int(m.group(1))
                name = m.group(2)
                cd.add_file(name, size)

root.print()

total_size = 0
for dir in all_dirs:
    if (size := dir.compute_size()) <= 100000:
        total_size += size

print(total_size)


# -- part 2 --
total_disk_space = 70000000
unused_needed = 30000000
used_space = root.compute_size()
unused_space = total_disk_space - used_space
min_to_delete = unused_needed - unused_space

smallest = used_space
for dir in all_dirs:
    size = dir.compute_size()
    if size >= min_to_delete and size < smallest:
        smallest = size

print(smallest)

import IPython; IPython.embed()
