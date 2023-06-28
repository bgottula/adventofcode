#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/12
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

# read in the map from file
map = []
with open('input') as f:
    for line in f:
        map.append([ord(c) - 96 for c in line.strip()])
map = np.array(map)

# fixup start and end
start = tuple(np.argwhere(map == -13)[0])
end = tuple(np.argwhere(map == -27)[0])
map[map == -13] = 1
map[map == -27] = 26

# display the map
plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(map)
ax.plot(start[1], start[0], 'r.')
ax.plot(end[1], end[0], 'r.')


class NoSolutionException(Exception):
    """Raised when there's no solution."""


def solve_recursive(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Recursive shortest path solver.

    This recursive approach is probably guaranteed to find the shortest path
    but only works on very small maps because it explores all possible valid
    paths in a depth-first fashion to find the shortest one. For even a modest
    sized map the total number of valid paths can be extremely large, so it
    becomes laughably impractical to try all of them.

    Args:
        path: The segments of a valid path from the start with a length of at
            least one.

    Returns:
        The shortest valid path that starts with the provided `path` and ends
        at the end point on the map.

    Raises:
        NoSolutionException if there is no valid extension of the provided
        `path` that reaches the end point.
    """
    loc = path[-1]

    right = (loc[0], loc[1] + 1)
    left = (loc[0], loc[1] - 1)
    up = (loc[0] - 1, loc[1])
    down = (loc[0] + 1, loc[1])

    # show progress in plot (this slows things down even more)
    dmap = map.copy()
    for p in path:
        dmap[p] = 27
    img.set_data(dmap)
    fig.canvas.draw_idle()
    plt.pause(0.0001)

    # try all candidate paths
    candidate_paths = []
    for next_loc in [left, right, up, down]:

        if next_loc in path:
            # revisiting the same square certainly won't be the shortest path
            continue

        if next_loc[0] < 0 or next_loc[1] < 0:
            # can't go this direction because it's off the map
            continue

        try:
            allowed = map[next_loc] <= map[loc] + 1
        except IndexError:
            # can't go this direction because it's off the map
            continue

        if not allowed:
            # can't go this direction because it's too steep
            continue

        # Reached the end, we're done. No need to compare with the other
        # candidate paths
        if next_loc == end:
            return path + [next_loc]

        # get the best (shortest) path that continues in this direction
        try:
            candidate_paths.append(solve_recursive(path + [next_loc]))
        except NoSolutionException:
            pass

    # no direction from here yields a path to the end
    if len(candidate_paths) == 0:
        raise NoSolutionException

    # find the direction that has the shortest path to the end
    return min(candidate_paths, key=len)


total_nodes = 1
reached_locations = set()
part = 2  # part 1 or part 2 of day 12

@dataclass
class Node:
    path: list[tuple[int, int]]
    children: Optional[list[Node]] = None

    def _add_children(self) -> Optional[list[tuple[int, int]]]:
        global total_nodes
        global reached_locations
        self.children = []
        loc = self.path[-1]

        right = (loc[0], loc[1] + 1)
        left = (loc[0], loc[1] - 1)
        up = (loc[0] - 1, loc[1])
        down = (loc[0] + 1, loc[1])

        for next_loc in [left, right, up, down]:

            if next_loc in reached_locations:
                # there's a shorter path to get to this same location
                continue

            if next_loc[0] < 0 or next_loc[1] < 0:
                # can't go this direction because it's off the map
                continue

            try:
                if part == 1:
                    allowed = map[next_loc] <= map[loc] + 1
                else:
                    allowed = map[next_loc] >= map[loc] - 1
            except IndexError:
                # can't go this direction because it's off the map
                continue

            if not allowed:
                # can't go this direction because it's too steep
                continue

            # show this path on the map
            dmap = map.copy()
            for p in (self.path + [next_loc]):
                dmap[p] = 27
            img.set_data(dmap)
            fig.canvas.draw_idle()
            plt.pause(0.0001)

            # add new leaf node
            reached_locations.add(next_loc)
            self.children.append(Node(self.path + [next_loc]))
            total_nodes += 1

            if (part == 1 and next_loc == end) or (part == 2 and map[next_loc] == 1):
                return self.path + [next_loc]

    def add_layer(self) -> Optional[list[tuple[int, int]]]:
        global total_nodes

        if not self.children:
            # haven't attempted to add child nodes at this layer yet
            if shortest_path := self._add_children():
                return shortest_path
        else:
            # already have child nodes; keep traversing the tree
            keeper_children = []
            for child in self.children:
                if shortest_path := child.add_layer():
                    return shortest_path
                if len(child.children) > 0:
                    keeper_children.append(child)

            # if (pruned := len(self.children) - len(keeper_children)) > 0:
            #     print(f'Pruned away {pruned} child nodes')
            # total_nodes -= len(self.children) - len(keeper_children)
            # self.children = keeper_children


def solve_depth_first():

    if part == 1:
        root = Node([start])
        reached_locations.add(start)
    else:
        root = Node([end])
        reached_locations.add(end)

    num_layers = 1
    while True:
        print(f'Adding layer {num_layers + 1} (total nodes: {total_nodes})')
        if best_path := root.add_layer():
            return best_path
        num_layers += 1



def main():
    # path = solve_recursive([start])
    # path = solve_recursive_non_optimial([start])
    path = solve_depth_first()
    print(len(path) - 1)

    # plt.show()
    import IPython; IPython.embed()


if __name__ == "__main__":
    main()


# Some other ideas:
#
# Any valid path must pass through many steps of +1 (at least 25, but possibly
# more since there may be negative steps in the path). Therefore, perhaps it
# would help to identify where all such steps of +1 exist in the map, and then
# try to identify the shortest path between all combinations of steps at
# adjacent elevations.
#
# It may be possible to eliminate many locations as unreachable by any valid
# path. For example, if there is a plateau in the map with sides that are at
# least 2 greater than anything else nearby, this area would be unreachable.
# However, I don't think this would cut down on computation time drastically.
# The recursive algorithm already avoids locations that aren't reachable.
#
# How I would do it in real life: Tie a rope to the start point, and drag it
# along with me traversing the maze to the summit. Then from the summit pull
# the rope taut. As long as I crossed the right boundaries between regions and
# didn't loop around some plateau or mountain this should yield the optimum
# path.
# 1) a) To do the initial rope lay-out maybe a completley random walk would be
#    sufficient, as long as it follows the rules along the way? But it could
#    wrap around plateaus or valleys and the tightening part would not help to
#    fix that.
#    b) Or, perhaps the initial lay-out should move towards the next step up as
#    directly as possible? This doesn't seem guaranteed to find the ideal path
#    between obstacles either.
#    c) Perhaps there's a way to try all possible routes around obstacles. This
#    seems like it would be tricky to formulate.
#    d) Use human input to draw the initial rope.
#    e) Come up with a simplified representation of regions containing
#    obstacles. For each obstacle, can either go around it to the left or to
#    the right. All combinations of left/right over all obstacles might yield
#    all possible initial rope layouts. However defining this rigorously might
#    be difficult.
# 2) The "pulling the rope taut" part might be able to re-use some of the
# rope bridge code from Day 9, but with some extra topology constraints on
# how the rope segments can follow and filling in some of the corners since
# diagonal moves are not allowed in this Day 12 puzzle.
#
#
# Okay, finally broke down and got a hint from AJ who looked at the solution.
# I need to do a breadth-first search, rather than a depth-first search (which
# is what the recursive algorithm is doing). This makes a lot of sense. This
# can be represented with a tree structure, where each branch forking from a
# node represents the direction taken from that node. There should be at most
# three branches from each node since back-tracking is clearly non-optimal.
# Many branches will terminate before reaching the end. By building out the
# tree layer by layer (breadth-first), the shortest path will be found as soon
# as some leaf node in a new layer is at the end point on the map.
#
# Unfortunately even this approach is too slow. The tree becomes massive very
# quickly, such that things grind to a near halt once it has about 20 layers.
# With no filtering out of obviously bad paths, the number of nodes would be
# 4**layer. Even with filtering the number of nodes appears to grow
# exponentially, resulting in a gigantic tree that takes a long time to
# traverse and which eventually consumes all of the system's memory.
#
# I think another strategy worth trying is to keep a dict or similar data
# structure that stores the current shortest path to each location on the map.
# Each time a leaf node is about to be added, the path length is compared to
# that dict. If the new leaf would result in a path longer than the current
# best, it's not created. That way hopefully there should only one active path
# to each location on the map. That ought to drastically reduce the number of
# nodes in the tree. However it's possible that this won't work, because
# perhaps the shortest overall path requires taking a slightly longer path to
# a particular location mid-way through the map.
