"""Tabular Q-learning agent over the mesh-discretised parking state.

State is flattened to a single integer index by the environment, so the Q-table
is a dense (n_states x n_actions) numpy array. Exploration is epsilon-greedy
with exponential decay.
"""

from __future__ import annotations

import numpy as np

from .env import N_ACTIONS


class QLearningAgent:
    """Off-policy tabular Q-learning.

    Parameters
    ----------
    n_states : int
        Number of discrete states (env.n_states).
    n_actions : int
        Number of actions.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon, epsilon_min, epsilon_decay : float
        Epsilon-greedy schedule. epsilon is multiplied by epsilon_decay each
        episode (call decay()), clamped at epsilon_min.
    seed : int | None
        RNG seed for reproducible exploration / tie-breaking.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = N_ACTIONS,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    # --------------------------------------------------------------- policy
    def act(self, state_idx: int, greedy: bool = False) -> int:
        """Pick an action. epsilon-greedy unless greedy=True."""
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state_idx)

    def greedy_action(self, state_idx: int) -> int:
        """Argmax with random tie-breaking (avoids a bias toward action 0)."""
        row = self.q[state_idx]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    # --------------------------------------------------------------- update
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        """One Q-learning backup."""
        target = r if done else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay(self) -> None:
        """Decay epsilon one episode's worth."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
