#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/16
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
import re
import networkx as nx
import matplotlib.pyplot as plt


def load_graph(filename: str) -> nx.Graph:
    G = nx.Graph()
    with open(filename) as f:
        for line in f:
            m = re.search('Valve ([A-Z]{2}) has flow rate=([0-9]+); tunnels? leads? to valves? ', line)

            this_node = m.group(1)
            flow_rate=int(m.group(2))

            G.add_node(this_node, flow_rate=flow_rate)

            line = line[len(m.group(0)):]
            while m := re.search('([A-Z]{2})', line):
                G.add_edge(this_node, m.group(1))
                line = line[4:]

    return G


def display_graph(G: nx.Graph) -> None:
    fig, ax = plt.subplots()
    nx.draw(
        G,
        with_labels=True,
        pos=nx.fruchterman_reingold_layout(G),
        node_color=[n['flow_rate'] if n['flow_rate'] > 0 else -10 for n in G.nodes.values()]
    )
    plt.show()


def solveit(
        G: nx.Graph,
        unopened_valves: list[str],
        minutes_remaining: int,
        pressure_relief_so_far: int,
        current_valve: str = 'AA',
        sp: str = '',
    ) -> int:
    """Find the maximum pressure relief possible for Part 1.

    This is a depth-first recursive search. It's fairly dumb, but it's at least
    smart enough to stop when time runs out.

    Args:
        G: Graph representing the cave system.
        unopened_valves: List of valves that have not been opened yet.
        minutes_remaining: Minutes remaining until the volcano erupts.
        pressure_relief_so_far: How much pressure will be relieved by all of
            the already opened valves by the end of the 30 minutes.
        current_valve: Current location in the volcano.

    Returns:
        The maximum pressure release.
    """

    # I guess we're going depth first
    pressure_relief_max = pressure_relief_so_far
    for next_valve in unopened_valves:

        # how long it will take to reach the next valve
        minutes_to_next = nx.shortest_path_length(G, current_valve, next_valve)

        # duration from the time the next valve is turned on until the end
        minutes_remaining_next = minutes_remaining - minutes_to_next - 1

        # how much pressure the next valve will release from the time it's opened till the end
        pressure_relief_next = max(minutes_remaining_next, 0) * G.nodes[next_valve]['flow_rate']

        if minutes_remaining_next <= 0 or len(unopened_valves) <= 1:
            pressure_relief_total = pressure_relief_so_far + pressure_relief_next
        else:
            pressure_relief_total = solveit(
                G=G,
                unopened_valves=[valve for valve in unopened_valves if valve != next_valve],
                minutes_remaining=minutes_remaining_next,
                pressure_relief_so_far=pressure_relief_so_far + pressure_relief_next,
                current_valve=next_valve,
                sp=sp + ' '
            )

        pressure_relief_max = max(pressure_relief_max, pressure_relief_total)

    return pressure_relief_max


@dataclass(frozen=True)
class Actor:
    # How long until this actor reaches the destination valve and turns it on, such that they are
    # ready to make the next decision.
    minutes_to_next: int
    # Which valve this actor is headed for currently.
    destination_valve: str


@lru_cache(maxsize=None)
def shortest_path_length(G: nx.Graph, valve_0: str, valve_1: str) -> int:
    return nx.shortest_path_length(G, valve_0, valve_1)


# @profile
def solveit_2(
        G: nx.Graph,
        unopened_valves: list[str],
        minutes_remaining: int,
        actors: list[Actor, Actor],
        pressure_relief_so_far: int,
        sp: str = '',
    ) -> int:
    """Find the maximum pressure relief possible for Part 2.

    This is a depth-first recursive search. It's fairly dumb, and it's quite slow, but it does
    (eventually) produce the correct answer!

    Args:
        G: Graph representing the cave system.
        unopened_valves: List of valves that have not been opened yet.
        minutes_remaining: Minutes remaining until the volcano erupts.
        actors: Current state of myself and the elephant. The first actor in the tuple must always
            be the one that has reached a valve, and so must have `minutes_to_next == 0`.
        pressure_relief_so_far: How much pressure will be relieved by all of
            the already opened valves by the end of the 26 minutes.

    Returns:
        The maximum pressure release.
    """

    # If only one of us is ready to make a decision, do the same as in Part 1.
    # If both are ready to make a decision, need to modify the for loop to iterate over all
    # permutations of length 2, except at the start when order doesn't matter so we just need all
    # combinations.

    assert actors[0].minutes_to_next == 0

    # Both the elephant and I are ready to make a new decision
    if actors[1].minutes_to_next == 0:

        pressure_relief_max = pressure_relief_so_far
        for idx, next_valves in enumerate(combinations(unopened_valves, 2)):

            if actors[0].destination_valve == 'AA':
                print(f'Trying initial combination {idx+1:2d} of 105')

            # how long it will take to reach the next valves and turn them on?

            # assign actors to next valves such that the total path length is minimized
            minutes_to_next_0_a = shortest_path_length(G, actors[0].destination_valve, next_valves[0]) + 1
            minutes_to_next_1_a = shortest_path_length(G, actors[1].destination_valve, next_valves[1]) + 1
            minutes_to_next_0_b = shortest_path_length(G, actors[0].destination_valve, next_valves[1]) + 1
            minutes_to_next_1_b = shortest_path_length(G, actors[1].destination_valve, next_valves[0]) + 1
            if minutes_to_next_0_a + minutes_to_next_1_a < minutes_to_next_0_b + minutes_to_next_1_b:
                next_valve_0 = next_valves[0]
                next_valve_1 = next_valves[1]
                minutes_to_next_0 = minutes_to_next_0_a
                minutes_to_next_1 = minutes_to_next_1_a
            else:
                next_valve_0 = next_valves[1]
                next_valve_1 = next_valves[0]
                minutes_to_next_0 = minutes_to_next_0_b
                minutes_to_next_1 = minutes_to_next_1_b

            # how long until the next decision point
            minutes_to_next_decision = min(minutes_to_next_0, minutes_to_next_1)

            # how many minutes will remain until the volcano erupts at the next decision point
            minutes_remaining_next = minutes_remaining - minutes_to_next_decision

            actors_next = [
                Actor(minutes_to_next_0 - minutes_to_next_decision, next_valve_0),
                Actor(minutes_to_next_1 - minutes_to_next_decision, next_valve_1)
            ]
            actors_next = sorted(actors_next, key=lambda actor: actor.minutes_to_next)

            # how much pressure the next valves will release from the time they are opened till the end
            pressure_relief_next = (
                max(minutes_remaining - minutes_to_next_0, 0) * G.nodes[next_valve_0]['flow_rate'] +
                max(minutes_remaining - minutes_to_next_1, 0) * G.nodes[next_valve_1]['flow_rate']
            )

            if minutes_remaining_next <= 0 or len(unopened_valves) <= 2:
                pressure_relief_total = pressure_relief_so_far + pressure_relief_next
            else:
                pressure_relief_total = solveit_2(
                    G=G,
                    unopened_valves=[valve for valve in unopened_valves if valve not in (next_valve_0, next_valve_1)],
                    minutes_remaining=minutes_remaining_next,
                    actors=actors_next,
                    pressure_relief_so_far=pressure_relief_so_far + pressure_relief_next,
                    sp=sp + ' '
                )

            pressure_relief_max = max(pressure_relief_max, pressure_relief_total)

        return pressure_relief_max


    # Only one of us is ready to make a new decision (always actors[0])
    current_valve = actors[0].destination_valve
    pressure_relief_max = pressure_relief_so_far
    for next_valve in unopened_valves:

        # how long it will take to reach the next valve and turn it on
        minutes_to_next = shortest_path_length(G, current_valve, next_valve) + 1

        # how long until the next decision point
        minutes_to_next_decision = min(minutes_to_next, actors[1].minutes_to_next)

        # how many minutes will remain until the volcano erupts at the next decision point
        minutes_remaining_next = minutes_remaining - minutes_to_next_decision

        actors_next = [
            Actor(minutes_to_next - minutes_to_next_decision, next_valve),
            Actor(actors[1].minutes_to_next - minutes_to_next_decision, actors[1].destination_valve)
        ]
        actors_next = sorted(actors_next, key=lambda actor: actor.minutes_to_next)

        # how much pressure the next valve will release from the time it's opened till the end
        pressure_relief_next = max(minutes_remaining - minutes_to_next, 0) * G.nodes[next_valve]['flow_rate']

        if minutes_remaining_next <= 0 or len(unopened_valves) <= 1:
            pressure_relief_total = pressure_relief_so_far + pressure_relief_next
        else:
            pressure_relief_total = solveit_2(
                G=G,
                unopened_valves=[valve for valve in unopened_valves if valve != next_valve],
                minutes_remaining=minutes_remaining_next,
                actors=actors_next,
                pressure_relief_so_far=pressure_relief_so_far + pressure_relief_next,
                sp=sp + ' '
            )

        pressure_relief_max = max(pressure_relief_max, pressure_relief_total)

    return pressure_relief_max


def main():

    G = load_graph('input')

    # list of names of all valves that have non-zero flow rate
    nonzero_valves = [node for node in G.nodes if G.nodes[node]['flow_rate'] > 0]

    # pressure_relief = solveit(
    #     G=G,
    #     unopened_valves=nonzero_valves,
    #     minutes_remaining=30,
    #     pressure_relief_so_far=0,
    #     current_valve='AA'
    # )

    # pressure_relief = solveit_2(
    #     G=G,
    #     unopened_valves=nonzero_valves,
    #     minutes_remaining=26,
    #     actors=[Actor(0, 'AA'), Actor(0, 'AA')],
    #     pressure_relief_so_far=0,
    # )

    # print(f'Pressure relief: {pressure_relief}')

    import IPython; IPython.embed()


if __name__ == "__main__":
    main()


# Ideas:
#
# Could build up a tree structure like the map path puzzle. In this case I don't
# see an obvious benefit of using breadth-first versus depth-first search,
# because the objective is not to find the shortest path. Furthermore, it may
# not be possible to prune the tree mid-way through building it up because a
# branch that is performing poorly at first could later take the lead and
# surpass others if high flow valves are opened relatively late.
#
# There are 30 minutes in which to perform actions, so a full tree will be 30
# levels deep. Based on the structure of the network, it seems that each node
# in the tree will typically have two or three branches coming out of it. One
# branch is to open the valve at that location (if it hasn't been opened
# already), and there's always one tunnel to another location but often two.
# Some locations have connections to three, four, or five other locations.
#
# This means an exhaustive search of the space will likely have between 2**30
# and 3**30 nodes. That's almost certainly too large to be practical, so I need
# to find a more efficient approach.
#
# Interestingly, a significant number of valves have a flow rate of zero, so
# that reduces the search space fairly significantly. The exact structure of
# the network could also reduce the search space quite a lot. Maybe a tree that
# only includes nodes for valves that have non-zero flow rate and which have
# not been turned on yet on that branch will be small enough to search
# exhaustively.
#
# This feels a bit like a travelling salesman problem with a slightly different
# objective.
#
# Another idea is to list all the valves in flow-rate order. Then work out the
# total pressure release if they are visited in that exact order, taking the
# shortest path between each of them. This isn't guaranteed to be the optimum
# path, though, and it's not entirely clear where to go from here. I suppose
# the order could be fiddled (e.g., swap the first two valves) to see if there
# are orderings that yield a higher overall flow rate, but I don't see a way to
# go about this that is systematic.
#
# Maybe the key is to just try all possible ordering of visiting valves that
# have a non-zero flow rate. Only 15 of the valves in the main puzzle input
# have a non-zero flow rate. The total number of orderings of this subset of
# valves is 15! = 1,307,674,368,000 Alas, this is also impractically large to
# search exhaustively. However, many orderings are probably redundant because
# time will run out before all valves can be turned on in many cases. If I
# build up a tree structure that accounts for this, I can probably cut down
# on this number significantly. It's hard to say how much without trying it...
# It may also be possible to avoid going down some branches of the tree by
# keeping track of the best path so far while building the tree, and use some
# calculation to determine whether a new branch can or can't do better than the
# best so far.
#
# Another idea is to prune out parts of the graph that are obviously paths that
# should never be taken. We really only need the shortest connections between
# all of the valves with non-zero flow rate and valve AA since it's the
# starting point. All other tunnels are useless. Maybe a pruned graph will
# make some form of exhaustive search practical.
#
# When a valve is turned on, there's no reason to visit that location again
# unless it's on the shortest path to some other valve. Thus perhaps the graph
# can be pruned as valves are opened such that the search space is reduced.
#
# There must be some algorithms from graph theory that can help here. Right now,
# all edges have effectively a weight of one since it takes one minute to
# traverse each one. These edge weights could become more meaningful if I re-
# draw the graph to eliminate valves with zero flow rate. I could also draw new
# edges that fully interconnect all of the non-zero valves such that each edge
# weight represents the shortest path length between those nodes. Not sure if
# that would help or not.
#
# Some algorithms and data structures that seem potentially relevant:
# * Heap / priority queue
# * Dijkstra's algorithm
# * Travelling salesman problem
#
# Heap seems promising. Each node could represent a valve to be turned on, and
# the root node would represent the valve that, if turned on next, would yield
# the highest benefit. However, heap is designed to work for nodes with static
# keys. In this problem, the key values would presumably represent the total
# pressure relief provided by that valve, but these values depend on the order
# the valves are turned on and are not fixed.
#
# The cost function for the problem is the following:
# P = sum over k { F_k * (30 - t_k) }
# where P is the total pressure relieved (what we want to maximize), k is the
# valve index, F_k is the flow rate of valve k, and t_k is the time that valve
# k is opened, in minutes. The values t_k are a function of the order the
# valves are turned on.
#
# Useful stuff in networkx:
# https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
# nx.shortest_path(G, 'AA', 'BB')
# nx.shortest_path_length(G, 'AA', 'BB')
