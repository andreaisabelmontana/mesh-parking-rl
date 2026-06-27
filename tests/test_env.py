"""Tests for the parking environment mechanics."""

import numpy as np
import pytest

from parking.env import (
    ParkingEnv,
    FORWARD,
    REVERSE,
    LEFT,
    RIGHT,
    HEADINGS,
)
from parking.scenario import make_env


# ----------------------------------------------------------- discretisation
def test_state_index_is_bijective():
    """Every (col,row,heading) maps to a unique index in [0, n_states)."""
    env = ParkingEnv(width=5, height=4, start=(0, 0, 0), goal=(4, 3, 0))
    seen = set()
    for r in range(env.height):
        for c in range(env.width):
            for h in range(4):
                idx = env.state_index((c, r, h))
                assert 0 <= idx < env.n_states
                seen.add(idx)
    assert len(seen) == env.n_states == 5 * 4 * 4


def test_state_index_is_consistent():
    """Same state -> same index, every time."""
    env = make_env()
    s = (3, 4, 2)
    assert env.state_index(s) == env.state_index(s)


def test_state_index_rejects_out_of_bounds():
    env = ParkingEnv(width=4, height=4, start=(0, 0, 0), goal=(3, 3, 0))
    with pytest.raises(ValueError):
        env.state_index((4, 0, 0))
    with pytest.raises(ValueError):
        env.state_index((0, 0, 9))


# --------------------------------------------------------------- boundaries
def test_in_bounds():
    env = ParkingEnv(width=3, height=3, start=(0, 0, 0), goal=(2, 2, 0))
    assert env.in_bounds(0, 0)
    assert env.in_bounds(2, 2)
    assert not env.in_bounds(3, 0)
    assert not env.in_bounds(-1, 0)
    assert not env.in_bounds(0, 3)


def test_boundary_collision_keeps_car_in_place():
    """Driving into the lot edge bounces back and is penalised, not moved."""
    env = ParkingEnv(width=4, height=4, start=(0, 0, 3), goal=(3, 3, 0))  # facing West
    env.reset()
    s, r, done, info = env.step(FORWARD)  # forward (West) leaves the lot
    assert info["collision"] is True
    assert s == (0, 0, 3)  # unchanged
    assert not done


# -------------------------------------------------------- collision / walls
def test_is_blocked_for_obstacle_and_boundary():
    env = ParkingEnv(width=4, height=4, obstacles=[(2, 2)], goal=(3, 3, 0))
    assert env.is_blocked(2, 2)       # obstacle
    assert env.is_blocked(4, 4)       # off-grid
    assert not env.is_blocked(1, 1)   # free


def test_driving_into_obstacle_is_penalised_and_blocked():
    # Car at (1,1) facing East; obstacle directly East at (2,1).
    env = ParkingEnv(width=4, height=4, start=(1, 1, 1), goal=(3, 3, 0), obstacles=[(2, 1)])
    env.reset()
    s, r, done, info = env.step(FORWARD)
    assert info["collision"] is True
    assert s == (1, 1, 1)                       # did not enter the obstacle
    assert r < 0                                # net negative reward
    assert r <= -env.COLLISION_PENALTY          # at least the collision penalty


def test_turn_never_collides():
    """Turning in place is always legal even when boxed in by obstacles."""
    env = ParkingEnv(
        width=3, height=3, start=(1, 1, 0), goal=(2, 2, 0),
        obstacles=[(0, 1), (2, 1), (1, 0)],
    )
    env.reset()
    s, r, done, info = env.step(LEFT)
    assert info["collision"] is False
    assert s == (1, 1, 3)  # rotated, same cell


# --------------------------------------------------------------- kinematics
def test_forward_moves_along_heading():
    env = ParkingEnv(width=5, height=5, start=(2, 2, 1), goal=(4, 4, 0))  # facing East
    env.reset()
    s, _, _, _ = env.step(FORWARD)
    assert s == (3, 2, 1)


def test_reverse_moves_opposite_heading():
    env = ParkingEnv(width=5, height=5, start=(2, 2, 0), goal=(4, 4, 0))  # facing North
    env.reset()
    s, _, _, _ = env.step(REVERSE)  # reverse while facing North -> moves South
    assert s == (2, 3, 0)


def test_steer_left_right_cycle_headings():
    env = ParkingEnv(width=5, height=5, start=(2, 2, 0), goal=(4, 4, 0))
    env.reset()
    s, _, _, _ = env.step(RIGHT)
    assert s == (2, 2, 1)
    s, _, _, _ = env.step(LEFT)
    assert s == (2, 2, 0)
    s, _, _, _ = env.step(LEFT)
    assert s == (2, 2, 3)  # wraps around


def test_headings_table_is_unit_vectors():
    for dc, dr in HEADINGS:
        assert abs(dc) + abs(dr) == 1


# ------------------------------------------------------------------- goal
def test_goal_requires_correct_heading_alignment():
    env = ParkingEnv(width=4, height=4, start=(0, 0, 1), goal=(2, 2, 2))
    # On the goal cell but wrong heading -> not a goal.
    assert not env.is_goal((2, 2, 0))
    assert not env.is_goal((2, 2, 1))
    # Correct cell AND heading -> goal.
    assert env.is_goal((2, 2, 2))


def test_reaching_goal_gives_positive_reward_and_done():
    # Start just North of the goal, facing South; one forward step parks it.
    env = ParkingEnv(width=4, height=4, start=(1, 0, 2), goal=(1, 1, 2))
    env.reset()
    s, r, done, info = env.step(FORWARD)
    assert info["goal"] is True
    assert done is True
    assert s == (1, 1, 2)
    assert r >= env.GOAL_REWARD - env.STEP_COST  # large positive


def test_wrong_heading_on_goal_cell_is_not_done():
    # Arrive on the goal cell with the wrong heading -> episode continues.
    env = ParkingEnv(width=4, height=4, start=(1, 0, 2), goal=(1, 1, 0))
    env.reset()
    s, r, done, info = env.step(FORWARD)
    assert s == (1, 1, 2)
    assert info["goal"] is False
    assert not done


# ------------------------------------------------------- reward signs / cost
def test_step_cost_is_negative_when_no_progress():
    """A neutral turn that doesn't change distance costs reward (net negative)."""
    env = make_env()
    env.reset()
    # A turn doesn't change cell distance; with step cost it must be < 0 here
    # because heading misalignment shaping is symmetric for one turn either way.
    s, r, done, info = env.step(LEFT)
    assert not info["goal"]
    assert r < 0


def test_timeout_terminates_episode():
    env = ParkingEnv(width=4, height=4, start=(0, 0, 0), goal=(3, 3, 0), max_steps=3)
    env.reset()
    done = False
    info = {}
    steps = 0
    while not done:
        # spin in place so we never reach the goal
        s, r, done, info = env.step(LEFT)
        steps += 1
    assert info["timeout"] is True
    assert steps == 3


def test_reset_returns_start_pose():
    env = make_env()
    env.step(FORWARD)
    s = env.reset()
    assert s == env.start
    assert env.steps == 0


# ----------------------------------------------------- construction guards
def test_invalid_layouts_raise():
    with pytest.raises(ValueError):
        ParkingEnv(width=4, height=4, goal=(2, 2, 0), obstacles=[(2, 2)])  # goal on obstacle
    with pytest.raises(ValueError):
        ParkingEnv(width=4, height=4, goal=(9, 9, 0))  # goal off grid
