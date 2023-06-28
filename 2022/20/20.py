#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/20
"""

import numpy as np

# This won't be modified
file = np.loadtxt('input', dtype=int)
file *= 811589153
file = list(file)

decrypted = file.copy()
# print(decrypted)

# values in the encrypted file are not unique, so keep a separate array of unique indices
indices = list(range(len(file)))

for round in range(10):

    # Do the mixing operations
    for file_idx, file_val in enumerate(file):

        # where to remove a value from the list
        del_idx = indices.index(file_idx)

        # where to re-insert the value in the list
        ins_idx = (del_idx + file_val) % (len(file) - 1)

        # move the item in the indices list
        val = indices.pop(del_idx)
        indices.insert(ins_idx, val)

        # move the item in the decrypted file list
        val = decrypted.pop(del_idx)
        decrypted.insert(ins_idx, val)

    # print(f'After round {round + 1}')
    # print(decrypted)


coords = [
    decrypted[(decrypted.index(0) + 1000) % len(decrypted)],
    decrypted[(decrypted.index(0) + 2000) % len(decrypted)],
    decrypted[(decrypted.index(0) + 3000) % len(decrypted)],
]

print(coords)
print(sum(coords))


import IPython; IPython.embed()
