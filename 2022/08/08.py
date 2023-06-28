#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/8
"""

import numpy as np

heights = []
with open('input') as f:
    for line in f:
        heights.append([int(c) for c in line.strip()])

# Mask value False means tree is not visible.
heights = np.array(heights)
visible = np.zeros(heights.shape, dtype=bool)

for height in range(10):

    # left to right
    v = np.argmax(heights >= height, axis=1)
    visible[np.arange(heights.shape[0]), v] = True

    # right to left
    v = heights.shape[1] - np.argmax(np.fliplr(heights) >= height, axis=1) - 1
    visible[np.arange(heights.shape[0]), v] = True

    # top to bottom
    v = np.argmax(heights >= height, axis=0)
    visible[v, np.arange(heights.shape[1])] = True

    # bottom to top
    v = heights.shape[0] - np.argmax(np.flipud(heights) >= height, axis=0) - 1
    visible[v, np.arange(heights.shape[1])] = True

print(np.sum(visible))



# -- part 2 --

scores_per_dir = []

# looking left
score = np.zeros_like(heights)
hc = heights.copy()
hc[:,-1] = 9
for idx in range(heights.shape[1] - 1):
    score[:, idx] = np.argmax(hc[:, idx+1:] >= hc[:,[idx]], axis=1) + 1
scores_per_dir.append(score)

# looking right
score = np.zeros_like(heights)
hc = np.fliplr(heights.copy())
hc[:,-1] = 9
for idx in range(heights.shape[1] - 1):
    score[:, idx] = np.argmax(hc[:, idx+1:] >= hc[:,[idx]], axis=1) + 1
score = np.fliplr(score)
scores_per_dir.append(score)

# looking down
score = np.zeros_like(heights)
hc = heights.copy()
hc[-1,:] = 9
for idx in range(heights.shape[0] - 1):
    score[idx, :] = np.argmax(hc[idx+1:, :] >= hc[[idx],:], axis=0) + 1
scores_per_dir.append(score)

# looking up
score = np.zeros_like(heights)
hc = np.flipud(heights.copy())
hc[-1,:] = 9
for idx in range(heights.shape[0] - 1):
    score[idx, :] = np.argmax(hc[idx+1:, :] >= hc[[idx],:], axis=0) + 1
score = np.flipud(score)
scores_per_dir.append(score)

scores = np.prod(scores_per_dir, axis=0)

print(np.max(scores))

import IPython; IPython.embed()
