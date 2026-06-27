"""Train the Q-learning agent to park, then save figures + results.json.

Usage:
    python train.py [--episodes N] [--seed S] [--quiet]

Outputs (under figures/ and ./):
    figures/learning_curve.png   success rate + return over training
    figures/trajectory.png       the trained agent's parking manoeuvre
    figures/trajectory.txt       ASCII frames of that manoeuvre
    results.json                 real success rate, mean steps, vs random
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# matplotlib without a display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from parking.agent import QLearningAgent  # noqa: E402
from parking.env import ACTION_NAMES, HEADING_NAMES  # noqa: E402
from parking.scenario import make_env  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# --------------------------------------------------------------------- train
def train(env, agent, episodes: int, log_every: int = 200, quiet: bool = False):
    """Run Q-learning. Returns per-episode returns and a rolling success rate."""
    returns = []
    successes = []  # 1.0 if the episode reached the goal
    for ep in range(episodes):
        s = env.reset()
        si = env.state_index(s)
        total = 0.0
        done = False
        reached = False
        while not done:
            a = agent.act(si)
            s2, r, done, info = env.step(a)
            si2 = env.state_index(s2)
            agent.update(si, a, r, si2, done)
            si = si2
            total += r
            if info["goal"]:
                reached = True
        agent.decay()
        returns.append(total)
        successes.append(1.0 if reached else 0.0)
        if (not quiet) and (ep + 1) % log_every == 0:
            window = successes[-log_every:]
            print(
                f"ep {ep + 1:5d} | eps {agent.epsilon:.3f} | "
                f"return {total:8.1f} | success@{log_every} {np.mean(window):.2f}"
            )
    return np.array(returns), np.array(successes)


# --------------------------------------------------------------- evaluation
def rollout(env, policy, max_steps=None):
    """Run one greedy episode under `policy(state_idx) -> action`.

    Returns (states, actions, reached_goal, n_steps).
    """
    s = env.reset()
    states = [s]
    actions = []
    reached = False
    steps = 0
    horizon = max_steps if max_steps is not None else env.max_steps
    for _ in range(horizon):
        a = policy(env.state_index(s))
        s, _, done, info = env.step(a)
        states.append(s)
        actions.append(a)
        steps += 1
        if info["goal"]:
            reached = True
        if done:
            break
    return states, actions, reached, steps


def evaluate(env, policy, episodes: int = 200):
    """Success rate + mean steps on successful episodes."""
    wins = 0
    steps_on_win = []
    for _ in range(episodes):
        _, _, reached, steps = rollout(env, policy)
        if reached:
            wins += 1
            steps_on_win.append(steps)
    rate = wins / episodes
    mean_steps = float(np.mean(steps_on_win)) if steps_on_win else float("nan")
    return rate, mean_steps


# ------------------------------------------------------------------ figures
def rolling(x, w):
    if len(x) < w:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def save_learning_curve(returns, successes, path):
    w = max(1, len(successes) // 50)
    sr = rolling(successes, w)
    rr = rolling(returns, w)
    x = np.arange(len(sr)) + w

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    ax1.plot(x, sr, color="#2563eb", lw=2)
    ax1.set_ylabel("success rate")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(f"Learning curve (rolling window = {w} episodes)")
    ax1.grid(alpha=0.25)

    ax2.plot(np.arange(len(rr)) + w, rr, color="#10182b", lw=1.5)
    ax2.set_ylabel("episode return")
    ax2.set_xlabel("episode")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_trajectory(env, states, path):
    """Draw the lot + the trained agent's path as a static figure."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    W, H = env.width, env.height

    # obstacles
    for (c, r) in env.obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#9aa6bf"))
    # goal cell
    gc, gr, gh = env.goal
    ax.add_patch(plt.Rectangle((gc, gr), 1, 1, color="#e8f0fe", ec="#2563eb", lw=2))

    # path of cell centres
    xs = [s[0] + 0.5 for s in states]
    ys = [s[1] + 0.5 for s in states]
    ax.plot(xs, ys, "-o", color="#2563eb", ms=4, lw=1.8, alpha=0.85)

    # heading arrows along the path
    head_vec = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    for s in states[:: max(1, len(states) // 14)]:
        dx, dy = head_vec[s[2]]
        ax.arrow(
            s[0] + 0.5, s[1] + 0.5, dx * 0.32, dy * 0.32,
            head_width=0.16, head_length=0.16, fc="#10182b", ec="#10182b",
        )

    # start / end markers
    ax.plot(xs[0], ys[0], "s", color="#16a34a", ms=11, label="start")
    ax.plot(xs[-1], ys[-1], "*", color="#dc2626", ms=18, label="parked")

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.grid(True, color="#dbe4f3")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at top, matching the env
    ax.set_title(f"Trained parking manoeuvre  (goal heading: {HEADING_NAMES[gh]})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_trajectory_ascii(env, states, actions, path):
    lines = []
    for i, s in enumerate(states):
        act = ACTION_NAMES[actions[i - 1]] if i > 0 else "start"
        lines.append(f"step {i:2d}  ({act})  pose=({s[0]},{s[1]},{HEADING_NAMES[s[2]]})")
        lines.append(env.render_ascii(s))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Train the mesh parking agent.")
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    env = make_env(seed=args.seed)
    agent = QLearningAgent(n_states=env.n_states, seed=args.seed)

    if not args.quiet:
        print("Lot layout (G = goal, # = obstacle, > = car):")
        print(env.render_ascii(env.start))
        print(f"\nStates: {env.n_states}  |  training for {args.episodes} episodes\n")

    returns, successes = train(env, agent, args.episodes, quiet=args.quiet)

    # Greedy + random policies for evaluation
    greedy = lambda si: agent.greedy_action(si)  # noqa: E731
    rng = np.random.default_rng(args.seed + 1)
    random_policy = lambda si: int(rng.integers(env.n_actions))  # noqa: E731

    g_rate, g_steps = evaluate(env, greedy, args.eval_episodes)
    r_rate, r_steps = evaluate(env, random_policy, args.eval_episodes)

    # One clean greedy rollout for the trajectory figures
    states, acts, reached, n = rollout(env, greedy)

    save_learning_curve(returns, successes, os.path.join(FIG_DIR, "learning_curve.png"))
    save_trajectory(env, states, os.path.join(FIG_DIR, "trajectory.png"))
    save_trajectory_ascii(env, states, acts, os.path.join(FIG_DIR, "trajectory.txt"))

    results = {
        "episodes": args.episodes,
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "n_states": int(env.n_states),
        "trained_agent": {
            "success_rate": round(g_rate, 4),
            "mean_steps_on_success": round(g_steps, 2) if g_steps == g_steps else None,
        },
        "random_policy": {
            "success_rate": round(r_rate, 4),
            "mean_steps_on_success": round(r_steps, 2) if r_steps == r_steps else None,
        },
        "demo_rollout": {
            "reached_goal": bool(reached),
            "steps": int(n),
            "actions": [ACTION_NAMES[a] for a in acts],
        },
        "final_train_success_rate_last_500": round(float(np.mean(successes[-500:])), 4),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if not args.quiet:
        print("\n=== RESULTS ===")
        print(f"trained : success {g_rate:.1%}  mean steps {g_steps:.1f}")
        print(f"random  : success {r_rate:.1%}")
        print(f"demo rollout reached goal in {n} steps: {reached}")
        print(f"\nwrote {out}")
        print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
