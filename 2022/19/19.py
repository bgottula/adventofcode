#!/usr/bin/env python3

"""
https://adventofcode.com/2022/day/19
"""

from __future__ import annotations
from math import prod
from functools import lru_cache
from typing import NamedTuple
import re
import numpy as np


class Recipe(NamedTuple):
    """Represents the cost to construct each type of robot."""
    ore: np.ndarray
    clay: np.ndarray
    obsidian: np.ndarray
    geode: np.ndarray
    nothing: np.ndarray = np.array([0, 0, 0, 0])


# Load up the recipes into this list
recipes = []
recipe_maximums = []
with open('input') as f:
    for line in f:
        s = re.search(
            'Blueprint [0-9]+: '
            'Each ore robot costs ([0-9]+) ore. '
            'Each clay robot costs ([0-9]+) ore. '
            'Each obsidian robot costs ([0-9]+) ore and ([0-9]+) clay. '
            'Each geode robot costs ([0-9]+) ore and ([0-9]+) obsidian.',
            line
        )
        recipe = Recipe(
            ore=np.array([int(s[1]), 0, 0, 0]),
            clay=np.array([int(s[2]), 0, 0, 0]),
            obsidian=np.array([int(s[3]), int(s[4]), 0, 0]),
            geode=np.array([int(s[5]), 0, int(s[6]), 0]),
        )
        recipes.append(recipe)
        recipe_maximums.append(np.array([
            max(recipe.ore[0], recipe.clay[0], recipe.obsidian[0], recipe.geode[0]),
            recipe.obsidian[1],
            recipe.geode[2],
            10000000,  # can never have enough geode robots
        ]))


# @profile
@lru_cache
def geodes_collected(
        construction_schedule: tuple[int],
        # recipe: Recipe,
    ) -> int:
    """Compute number of geodes collected as a function of construction schedule.

    Args:
        construction_schedule: Array representing the robot construction schedule. Each element
            indicates the type of robot constructed during that minute, where:

            0 - no robot constructed
            1 - ore robot
            2 - clay robot
            3 - obsidian robot
            4 - geode robot

            Values outside [0, 4] are undefined, but will be treated the same as 0.

        recipe: Resources required to construct each type of robot.

    Returns: The number of geodes collected, or -1 if a constraint was violated.
    """

    # It's never possible to construct a robot during the first minute
    if construction_schedule[1] > 0:
        return -1

    # Construction schedule
    # Index 0 == minute 0
    # construction_schedule = np.insert(construction_schedule, 0, 1)
    new_ore_robot = np.array(construction_schedule) == 1
    new_clay_robot = np.array(construction_schedule) == 2
    new_obsidian_robot = np.array(construction_schedule) == 3
    new_geode_robot = np.array(construction_schedule) == 4

    # Robot inventory
    # Index 0 == minute 0
    ore_robots = np.cumsum(new_ore_robot)
    clay_robots = np.cumsum(new_clay_robot)
    obsidian_robots = np.cumsum(new_obsidian_robot)
    geode_robots = np.cumsum(new_geode_robot)


    # Construction costs
    # Index 0 == minute 1
    global recipe
    ore_cost = (
        recipe.ore[0] * new_ore_robot[1:]
        + recipe.clay[0] * new_clay_robot[1:]
        + recipe.obsidian[0] * new_obsidian_robot[1:]
        + recipe.geode[0] * new_geode_robot[1:]
    )
    clay_cost = recipe.obsidian[1] * new_obsidian_robot[1:]
    obsidian_cost = recipe.geode[2] * new_geode_robot[1:]

    # Resources at the end of each minute
    # Index 0 == minute 1
    ore = np.cumsum(ore_robots)[:-1] - np.cumsum(ore_cost)
    clay = np.cumsum(clay_robots)[:-1] - np.cumsum(clay_cost)
    obsidian = np.cumsum(obsidian_robots)[:-1] - np.cumsum(obsidian_cost)
    geodes = np.cumsum(geode_robots)[:-1]

    # print(ore)
    # print(clay)
    # print(obsidian)
    # print(geodes)

    # import IPython; IPython.embed()

    # Make sure there was enough resources at the end of the previous minute for
    # any construction that happened. This does not check construction on minute
    # 1. That is covered by a separate check at the start of this function.
    if (
        np.any(ore_cost[1:] > ore[:-1]) or
        np.any(clay_cost[1:] > clay[:-1]) or
        np.any(obsidian_cost[1:] > obsidian[:-1])
        ):
        return -1

    # total number of geodes collected over the entire time period
    return geodes[-1]


# @profile
def solveit(
        recipe_idx: int,
        resources: np.ndarray = np.array([0, 0, 0, 0]),
        robots: np.ndarray = np.array([1, 0, 0, 0]),
        minutes_left: int = 24,
    ) -> int:
    """Simple depth-first recursive search algorithm.

    Each recursion level represents one minute of time.

    Args:
        recipe_idx: Index into the global `recipes` list.
        resources: Array where each element gives the number of resources available at the start of
            this minute.
        robots: Array where each element gives the number of robots existing at the start of this
            minute.
        minutes_left: Minutes remaining until time runs out.

    Returns:
        The total number of geodes collected using the best strategy.
    """

    # print(f'Resources: {resources}')

    assert minutes_left >= 1
    # not enough time left to make any more robots that will help
    if minutes_left == 1:
        return (resources + robots)[-1]

    geodes_collected = 0

    # try to build each type of robot (can build at most one this minute)
    for idx, robot_cost in enumerate(recipes[recipe_idx]):

        if any(robot_cost > resources):
            # can't afford to make one of these right now
            continue

        robots_new = robots.copy()
        if idx < 4:
            robots_new[idx] += 1
        geodes = solveit(
            resources=resources + robots - robot_cost,
            robots=robots_new,
            minutes_left=minutes_left - 1,
            recipe_idx=recipe_idx,
        )
        geodes_collected = max(geodes_collected, geodes)

    return geodes_collected

# @profile
# def time_to_next_robot(
#         robot_cost: np.ndarray,
#         resources: np.ndarray,
#         robots: np.ndarray,
#     ) -> int:
#     """Calculate time until enough resources are available to build a robot"""
#     s = (robot_cost - resources) / robots
#     t = np.ceil(s)
#     u = max(t)
#     v = int(u)
#     w = v + 1
#     return max(w, 1)


def time_to_next_robot(
        robot_cost: np.ndarray,
        resources: np.ndarray,
        robots: np.ndarray,
    ) -> int:
    """Calculate time until enough resources are available to build a robot"""
    return max(int(max(np.ceil((robot_cost - resources) / robots))) + 1, 1)


n = 0
import time

# @profile
def solveit2(
        recipe_idx: int,
        resources: np.ndarray = np.array([0, 0, 0, 0]),
        robots: np.ndarray = np.array([1, 0, 0, 0]),
        minutes_left: int = 32,
    ) -> int:
    """Simple depth-first recursive search algorithm.

    Each recursion level represents the next robot to build.

    Args:
        recipe_idx: Index into the global `recipes` list.
        resources: Array where each element gives the number of resources available at the start of
            this minute.
        robots: Array where each element gives the number of robots existing at the start of this
            minute.
        minutes_left: Minutes remaining until time runs out.

    Returns:
        The total number of geodes collected using the best strategy.
    """

    # global n
    # print(f'\nCall {n}')
    # n += 1
    # print(f'Resources: {resources}')
    # print(f'Robots: {robots}')
    # print(f'Minutes left: {minutes_left}')
    # time.sleep(0.1)

    assert minutes_left >= 0

    recipe_max = recipe_maximums[recipe_idx]

    # Try building each type of robot to find which yields the most geodes collected
    geodes_collected = 0
    for idx, robot_cost in enumerate(recipes[recipe_idx][:-1]):

        # if robots[idx] >= recipe_max[idx]:
        #     # no need to build more of this robot type
        #     continue

        if minutes_left * robots[idx] + resources[idx] >= minutes_left * recipe_max[idx]:
            # no need to build more of this robot type
            continue

        try:
            minutes_to_build = time_to_next_robot(robot_cost, resources, robots)
        except OverflowError:
            # can't build this robot type yet
            continue

        if minutes_to_build > minutes_left:
            # pointless to build this robot type since not enough time remains
            continue

        robots_new = robots.copy()
        robots_new[idx] += 1
        geodes = solveit2(
            resources=resources + minutes_to_build * robots - robot_cost,
            robots=robots_new,
            minutes_left=minutes_left - minutes_to_build,
            recipe_idx=recipe_idx,
        )
        geodes_collected = max(geodes_collected, geodes)

    # Don't build any more robots case
    geodes = (resources + minutes_left * robots)[-1]
    geodes_collected = max(geodes_collected, geodes)

    return geodes_collected


# geodes = solveit2(
#     recipe_idx=0,
#     resources=np.array([0, 0, 0, 0]),
#     robots=np.array([1, 0, 0, 0]),
#     minutes_left=32,
# )


# ---- Part 1 ----

# Use multiprocessing to solve multiple recipes in parallel
# import multiprocessing
# with multiprocessing.Pool(processes=22) as pool:
#     geodes = pool.map(solveit2, range(len(recipes)))

# print(geodes)
# print(sum(geodes * (np.arange(len(geodes)) + 1)))


# ---- Part 2 ----

# Use multiprocessing to solve multiple recipes in parallel
import multiprocessing
with multiprocessing.Pool(processes=22) as pool:
    geodes = pool.map(solveit2, range(3))

print(geodes)
print(prod(geodes))


# Results solveit()   solveit2() (prior to final optimization)
# 19 - 1              0m1.629s
# 20 - 2              0m4.038s
# 21 - 3  3m1.283s    0m11.817s
# 22 - 5  11m31.362s  0m37.438s
# 23 - 7              2m4.474s
# 24 - 9              7m3.928s



# solveit2() with optimizations
# 24 - 9    0m2.265s
# 26 - 15   0m11.112s
# 28 - 26   0m54.065s
# 30 - 39   4m8.413s
# 32



# import IPython; IPython.embed()


# Code used to investigate elapsed time vs minutes remaining for different recipes
# import time
# results = np.zeros((5, len(recipes)))
# for recipe_idx in range(len(recipes)):
#     print(f'Recipe index {recipe_idx}:')
#     for start_idx, start_min in enumerate(range(13, 18)):
#         start_time = time.perf_counter()
#         solveit(recipe_idx, minutes_left=start_min)
#         elapsed = time.perf_counter() - start_time
#         results[start_idx, recipe_idx] = elapsed
#         print(f'\tWith {start_min} mins remaining at start took {elapsed} s')
# np.savetxt('/tmp/day19trend.csv', results, delimiter=',')


# ================================= Notes and Strategy ===========================================
#
# Given a recipe:
# * Robots will collects resources / crack geodes automatically
# * Only decisions to be made each minute are:
#   1) Whether to construct a new robot, and if yes
#   2) Which robot to construct
#
# It seems like another case where a recursive search strategy is possible. Unfortunately, this
# seems to take too long, even with reasonably optimized code. Not super surprising.
#
# 13 mins: 0.095 s
# 14 mins: 0.12 s
# 15 mins: 0.22 s
# 16 mins: 0.47 s
# 17 mins: 1.3 s
# 18 mins: 4.2 s
# 19 mins: 14.1 s
# 20 mins: 50.5 s
#
# If the trend continues, the 24 minute starting point will take around 5000-6000 seconds, which
# is 1.4 to 1.7 hours. If true, this isn't super infeasible to run. May as well give it a shot.
# The main downside is that if there are bugs in my code, which is likely, then the turn-around
# time to find and fix them is quite large. I'm most worried about off-by-1 errors in the logic.
#
# To run against all of the recipes will be a larger challenge. Fortunately this is an embarassingly
# parallel problem and I have 24 CPU cores at my disposal, so running the full set of 30 recipes
# would probably take something like 3-4 hours to run. Not great, but not totally impractical.
#
# -- optimizing the recursive strategy --
# This may be futile, but there are probably ways to optimize the recursive strategy.
# * Rewrite as an iterative algorithm rather than recursive. Unclear if the function call overhead
#   is really hurting much here though.
# * Figure out when it will be impossible to make any more geode-cracking robots. From that point,
#   it should be trivial to calculate how many more geodes will be cracked in the time remaining.
#   This is an early-stopping technique. Perhaps it yields just enough speed-up to be worth it.
# * Figure out the minimum amount of resources that must be accumulated before a new robot of any
#   type could be produced, to skip unnecessary recursion levels. Unclear if this would make much
#   difference.
# * Make each step in the search be "which robot to build next" rather than minutes.
#
# Another idea is to come up with the equation for number of geodes on minute 24 as a function of
# the types of robots manufactured on each minute, then run an optimizer to maximize that function.
# scipy.optimize seems to be designed for continuous rather than discrete optimization, so I'm not
# sure if it can be used. Another idea is to use simulated annealing. I could probably write such
# a thing myself, or I could use a package like https://github.com/perrygeo/simanneal. This
# approach is not guaranteed to find the global optimum solution, which makes me think it's
# probably the wrong approach.
#
# ------------------------------ A Long and Unproductive Detour -----------------------------------
#
# This may be a case where working backwards it the best strategy. Starting with a geode-cracking
# robot, work backwards to figure out which robots and in which order need to be produced in order
# to manufacture the geode-cracking robot. Obviously if there's only a single geode-cracking robot
# the best strategy is to manufacture it as soon as feasible. If there are eventually two geode-
# cracking robots, it's less clear if the first one should be created as soon as possible or if
# it's advantagous to delay construction of the first robot a bit such that the second one can be
# created sooner.
#
# The problem space could be reduced by first figuring out the soonest that the first geode-
# cracking robot could possibly be constructed. Using example blueprint 1, this can't happen until
# there are 2 ore and 7 obsidian. So at least one obsidian robot must have been constructed, but
# possibly more, and potentially additional ore robots were constructed.
#
# Let's solve a smaller problem first. Here are some options for simplification:
# 1) Reduce the number of types of robots.
# 2) Lift the constraint that only one robot can be built per minute.
# 3) Simplify the blueprints such that each requires just one type of resource.
#
# Taking all three of these together is the easiest problem I can think of, where only geode
# collecting robots can be constructed and the blueprint for that robot requires no resources.
# In this trivial case, one geode collecting robot can be built on each of the 24 minutes. The
# total number of geodes collected is then n * (n - 1) / 2 where n = 24, which is 276 geodes.
#
# Now introduce one resource requirement. Let's say the blueprint for a geode-collecting robot
# requires 2 ore, but we still don't get to build new ore collecting robots we are only allowed
# to use the one we have at the start. This means it takes two minutes to accumulate two ore, so
# the fastest we can build geode collecting robots is once every two minutes starting on minute 3
# (the ore must be available at the start of the minute when construction happens). Geode
# collecting robots are constructed on minute 3, 5, 7, ..., and 23 and gather a total of 121
# geodes. Still very straightforward.
#
# Now let's introduce ore collecting robot construction back into the mix. Let's say an ore-
# collecting robot also requires 2 ore. What's the best strategy now? Each time two ore are
# available, we have a choice between constructing an ore collecting robot and a geode collecting
# robot. In this case, the best strategy is to construct an ore collecting robot as soon as
# possible (minute 3), then construct geode collecting robots on every minute starting on minute
# 5. Since the rate of ore collection allows geode robot construction once per minute there's no
# benefit from constructing additional ore-collecting robots; the additional ore would just pile up
# unused. We can determine the maximum number of ore collecting robots that would ever be needed
# directly from the blueprint: Two ore per minute is the maximum rate of ore production we could
# ever need, since this is the amount consumed when one geode-collecting robot is constructed per
# minute. With the optimum strategy 190 geodes are collected.
#
# Now let's say ore collecting robots require 5 ore. It's less obvious that the time spent waiting
# for 5 ore to accumulate in order to increase the production rate of geode collecting robots is
# worth it. The second ore collecting robot can't be constructed until minute 6, and the first
# geode collecting robot can't be constructed until minute 8. That seems a long time to wait to
# get the first geodes, but it is in fact better than only building geode robots: total geodes
# collected is 136, versus 121 by only building geode robots. How much would an ore robot need to
# cost before the construction cost is no longer worth it? When we build the second ore robot, the
# total number of geodes collected is n * (n - 1) / 2, where n = 22 - C and C is the cost of the
# ore robot. Plugging in values for C we find that for C = 5 we collect 136 geodes and for C = 6
# we collect 120 geodes. So it only makes sense to construct the second ore robot if the cost is
# 5 ore or less. In this simplified scenario we can directly calculate this up front without any
# trial-and-error sort of optimization.
#
# A pattern emerging in these simplified scenarios is that it never makes sense to delay
# construction of ore robots, if we build them at all. There's a clear order of building ore robots
# first and once those are constructed it only makes sense to then make geode robots as fast as
# possible. It isn't clear yet if this generalizes to more complex blueprints and scenarios but I
# suspect some weaker form of it may still hold.
