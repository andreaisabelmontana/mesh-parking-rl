"""The default parking scenario used by training, tests, and the demo.

A car enters an 8x8 lot from the top-left and must park (facing South) in a
slot on the bottom row, flanked on both sides by parked cars. Reaching the slot
requires driving across the lot, turning, and easing into a one-cell gap with
the correct heading — exactly the kind of sequencing that makes parking hard.
"""

from __future__ import annotations

from .env import ParkingEnv

WIDTH = 8
HEIGHT = 8
START = (0, 0, 1)      # top-left, facing East
GOAL = (4, 7, 2)       # bottom-row slot, must face South (aligned)

# Parked cars flanking the goal slot, plus a couple of pillars to route around.
OBSTACLES = [
    (3, 7), (5, 7),          # cars on either side of the goal slot
    (3, 6), (5, 6),          # their front bumpers (narrow the approach)
    (2, 3), (5, 3), (6, 3),  # a row of pillars mid-lot
]


def make_env(max_steps: int = 120, seed: int | None = None) -> ParkingEnv:
    return ParkingEnv(
        width=WIDTH,
        height=HEIGHT,
        start=START,
        goal=GOAL,
        obstacles=OBSTACLES,
        max_steps=max_steps,
        seed=seed,
    )
