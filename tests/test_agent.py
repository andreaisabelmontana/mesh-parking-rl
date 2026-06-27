"""Tests for the Q-learning agent and the learning result."""

import numpy as np
import pytest

from parking.agent import QLearningAgent
from parking.scenario import make_env
from train import train, evaluate


# --------------------------------------------------------------- mechanics
def test_qtable_shape():
    agent = QLearningAgent(n_states=20, n_actions=4)
    assert agent.q.shape == (20, 4)
    assert np.all(agent.q == 0.0)


def test_greedy_action_picks_argmax():
    agent = QLearningAgent(n_states=3, n_actions=4, seed=0)
    agent.q[1] = np.array([0.0, 5.0, 0.0, 0.0])
    assert agent.greedy_action(1) == 1


def test_update_moves_value_toward_target():
    agent = QLearningAgent(n_states=3, n_actions=2, alpha=0.5, gamma=0.0)
    before = agent.q[0, 0]
    agent.update(0, 0, r=10.0, s_next=1, done=True)
    # done => target is just r=10; with alpha 0.5 the value moves halfway.
    assert agent.q[0, 0] == pytest.approx(before + 0.5 * (10.0 - before))
    assert agent.q[0, 0] == pytest.approx(5.0)


def test_update_uses_bootstrap_when_not_done():
    agent = QLearningAgent(n_states=3, n_actions=2, alpha=1.0, gamma=0.9)
    agent.q[1] = np.array([0.0, 4.0])  # max next value = 4
    agent.update(0, 0, r=1.0, s_next=1, done=False)
    # target = 1 + 0.9*4 = 4.6; alpha 1.0 => value becomes target exactly.
    assert agent.q[0, 0] == pytest.approx(4.6)


def test_epsilon_decays_and_clamps():
    agent = QLearningAgent(
        n_states=2, n_actions=2, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5
    )
    agent.decay()
    assert agent.epsilon == pytest.approx(0.5)
    for _ in range(20):
        agent.decay()
    assert agent.epsilon == pytest.approx(0.1)  # clamped at min


def test_greedy_flag_avoids_exploration():
    # epsilon = 1 means act() would normally explore; greedy=True must not.
    agent = QLearningAgent(n_states=2, n_actions=4, epsilon=1.0, seed=0)
    agent.q[0] = np.array([0.0, 0.0, 9.0, 0.0])
    for _ in range(20):
        assert agent.act(0, greedy=True) == 2


# ----------------------------------------------------- the learning result
@pytest.fixture(scope="module")
def trained():
    """Train once and reuse across the slow assertions."""
    env = make_env(seed=0)
    agent = QLearningAgent(n_states=env.n_states, seed=0)
    _, successes = train(env, agent, episodes=3000, quiet=True)
    return env, agent, successes


def test_trained_agent_beats_random_by_a_wide_margin(trained):
    env, agent, _ = trained
    greedy = lambda si: agent.greedy_action(si)  # noqa: E731
    rng = np.random.default_rng(123)
    random_policy = lambda si: int(rng.integers(env.n_actions))  # noqa: E731

    g_rate, g_steps = evaluate(env, greedy, episodes=200)
    r_rate, _ = evaluate(env, random_policy, episodes=200)

    # The trained agent should park essentially always...
    assert g_rate >= 0.95
    # ...and a random policy should almost never park in this lot.
    assert r_rate <= 0.3
    # The gap is the headline result.
    assert g_rate - r_rate >= 0.5
    # And it should park in a sensible number of steps (not by luck-flailing).
    assert g_steps < env.max_steps


def test_training_converges(trained):
    _, _, successes = trained
    # By the end of training the agent reaches the goal nearly every episode
    # (under a small residual epsilon).
    assert np.mean(successes[-300:]) >= 0.8


def test_greedy_rollout_reaches_goal(trained):
    env, agent, _ = trained
    from train import rollout

    _, _, reached, steps = rollout(env, lambda si: agent.greedy_action(si))
    assert reached is True
    assert steps < env.max_steps
