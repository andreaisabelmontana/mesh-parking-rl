"""Mesh-discretised parking-lot environment.

The lot is a W x H grid of cells. The car occupies a single cell and carries a
heading discretised into 4 compass directions. Movement is non-holonomic: the
car drives forward / reverse along its heading, or turns in place left / right.

State  : (col, row, heading)  -> a cell on the mesh plus one of 4 headings.
Actions: 0 forward, 1 reverse, 2 steer-left, 3 steer-right.
Goal   : reach the target cell with the target heading (aligned).

Reward shaping (see README for the WHY):
    +GOAL_REWARD            on reaching the goal cell with correct heading
    -COLLISION_PENALTY      on hitting a wall, obstacle, or the lot boundary
    -STEP_COST              every step (encourages short manoeuvres)
    potential-based shaping  drives the agent toward the goal pose without
                             changing the optimal policy (Ng et al., 1999).

The implementation uses only the standard library + numpy.
"""

from __future__ import annotations

import numpy as np

# Heading encoding: index -> (d_col, d_row). Row increases downward.
#   0 = North (up), 1 = East (right), 2 = South (down), 3 = West (left)
HEADINGS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int64)
HEADING_NAMES = ("N", "E", "S", "W")

# Actions
FORWARD, REVERSE, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("forward", "reverse", "steer-left", "steer-right")
N_ACTIONS = 4

# Cell types in the static lot grid
FREE, WALL = 0, 1


class ParkingEnv:
    """A discretised parking lot for tabular RL.

    Parameters
    ----------
    width, height : int
        Mesh dimensions in cells.
    start : (col, row, heading)
        Initial pose of the car.
    goal : (col, row, heading)
        Target parked pose (cell + required heading for "aligned").
    obstacles : iterable of (col, row)
        Occupied cells (other parked cars / walls) the car must avoid.
    max_steps : int
        Episode horizon.
    """

    # Reward constants
    GOAL_REWARD = 100.0
    COLLISION_PENALTY = 20.0
    STEP_COST = 1.0
    SHAPING_SCALE = 1.0  # weight on potential-based shaping

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        start=(0, 0, 1),
        goal=(6, 6, 2),
        obstacles=None,
        max_steps: int = 120,
        seed: int | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.start = tuple(int(x) for x in start)
        self.goal = tuple(int(x) for x in goal)
        self.max_steps = int(max_steps)
        self.n_actions = N_ACTIONS
        self.rng = np.random.default_rng(seed)

        # Static occupancy grid (walls / parked cars). Indexed [row, col].
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.obstacles = set()
        if obstacles:
            for (c, r) in obstacles:
                self._mark_obstacle(int(c), int(r))

        self._validate_layout()
        self.state = self.start
        self.steps = 0

    # ------------------------------------------------------------------ setup
    def _mark_obstacle(self, col: int, row: int) -> None:
        if not self.in_bounds(col, row):
            raise ValueError(f"obstacle ({col},{row}) out of bounds")
        self.grid[row, col] = WALL
        self.obstacles.add((col, row))

    def _validate_layout(self) -> None:
        gc, gr, _ = self.goal
        sc, sr, _ = self.start
        if not self.in_bounds(gc, gr):
            raise ValueError("goal out of bounds")
        if not self.in_bounds(sc, sr):
            raise ValueError("start out of bounds")
        if self.grid[gr, gc] == WALL:
            raise ValueError("goal cell is an obstacle")
        if self.grid[sr, sc] == WALL:
            raise ValueError("start cell is an obstacle")

    # -------------------------------------------------------------- mechanics
    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def is_blocked(self, col: int, row: int) -> bool:
        """A cell is blocked if out of bounds or occupied by an obstacle."""
        if not self.in_bounds(col, row):
            return True
        return self.grid[row, col] == WALL

    def is_goal(self, state) -> bool:
        """Goal requires the correct cell AND correct heading (alignment)."""
        col, row, heading = state
        gc, gr, gh = self.goal
        return col == gc and row == gr and heading == gh

    @property
    def n_states(self) -> int:
        return self.width * self.height * 4

    def state_index(self, state) -> int:
        """Flatten (col, row, heading) -> a single integer index.

        Consistent, bijective mapping used by the tabular agent.
        """
        col, row, heading = state
        if not self.in_bounds(col, row):
            raise ValueError(f"state ({col},{row}) out of bounds")
        if not (0 <= heading < 4):
            raise ValueError(f"heading {heading} out of range")
        return (row * self.width + col) * 4 + heading

    # ----------------------------------------------------------- transitions
    def _apply_action(self, state, action):
        """Return the geometric next pose for an action (no reward / clipping)."""
        col, row, heading = state
        if action == FORWARD:
            dc, dr = HEADINGS[heading]
            return (col + dc, row + dr, heading)
        if action == REVERSE:
            dc, dr = HEADINGS[heading]
            return (col - dc, row - dr, heading)
        if action == LEFT:
            return (col, row, (heading - 1) % 4)
        if action == RIGHT:
            return (col, row, (heading + 1) % 4)
        raise ValueError(f"invalid action {action}")

    def _potential(self, state) -> float:
        """Shaping potential: higher = closer to the goal pose.

        Combines Manhattan distance to the goal cell with a small heading-
        alignment bonus. Used in a potential-based shaping term so it never
        changes the set of optimal policies.
        """
        col, row, heading = state
        gc, gr, gh = self.goal
        dist = abs(col - gc) + abs(row - gr)
        # circular heading difference in {0,1,2} (0 best, 2 opposite)
        hdiff = min((heading - gh) % 4, (gh - heading) % 4)
        return -(float(dist) + 0.5 * float(hdiff))

    # ------------------------------------------------------------------- API
    def reset(self, state=None):
        """Reset to the start pose (or a supplied pose). Returns the state."""
        self.state = tuple(int(x) for x in state) if state is not None else self.start
        self.steps = 0
        return self.state

    def step(self, action):
        """Advance one step.

        Returns (next_state, reward, done, info).
        """
        action = int(action)
        self.steps += 1
        prev = self.state
        nxt = self._apply_action(prev, action)
        ncol, nrow, nheading = nxt

        info = {"collision": False, "goal": False, "timeout": False}
        reward = -self.STEP_COST

        # Turns never move the car, so only forward/reverse can collide.
        moved = action in (FORWARD, REVERSE)
        if moved and self.is_blocked(ncol, nrow):
            # Stay in place; pay a collision penalty.
            reward -= self.COLLISION_PENALTY
            info["collision"] = True
            nxt = prev  # bounce back
        else:
            self.state = nxt

        # Potential-based shaping: F = gamma * phi(s') - phi(s). We fold gamma
        # in at the agent; here we use gamma ~ 1 for the shaping term, which is
        # a standard and policy-invariant choice.
        shaping = self.SHAPING_SCALE * (self._potential(self.state) - self._potential(prev))
        reward += shaping

        done = False
        if self.is_goal(self.state):
            reward += self.GOAL_REWARD
            info["goal"] = True
            done = True
        elif self.steps >= self.max_steps:
            info["timeout"] = True
            done = True

        return self.state, reward, done, info

    # ---------------------------------------------------------------- render
    def render_ascii(self, state=None) -> str:
        """Return an ASCII picture of the lot with the car drawn in."""
        col, row, heading = state if state is not None else self.state
        gc, gr, _ = self.goal
        arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
        rows = []
        for r in range(self.height):
            cells = []
            for c in range(self.width):
                if (c, r) == (col, row):
                    cells.append(arrows[heading])
                elif (c, r) == (gc, gr):
                    cells.append("G")
                elif self.grid[r, c] == WALL:
                    cells.append("#")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)
