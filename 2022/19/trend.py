#!/usr/bin/env python3

# plot trend in runtime vs minutes remaining

import numpy as np
import matplotlib.pyplot as plt

mins = np.array([
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
])

runtime = np.array([
    0.095,
    0.12,
    0.22,
    0.47,
    1.3,
    4.2,
    14.1,
    50.5,
])

plt.semilogy(mins, runtime)
plt.xlabel('Minutes Remaining')
plt.ylabel('Runtime [s]')
plt.grid(True)
plt.show()
