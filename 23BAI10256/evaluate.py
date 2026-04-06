"""
Evaluate trained Q-Learning agent and compare against baselines.

Runs three strategies side by side:
  1. Trained Q-Learning agent
  2. Fixed-timer controller (traditional approach)
  3. Longest-queue-first heuristic

Generates comparison plots.
"""

import numpy as np
import config as cfg
from environment import TrafficIntersection
from agent import QLearningAgent, FixedTimerController, LongestQueueController
from utils import ensure_dirs, plot_comparison


def evaluate_strategy(strategy_name, env, choose_fn, num_episodes=50, step_fn=None):
    """
    Run a strategy for several episodes and collect metrics.

    Parameters
    ----------
    strategy_name : str
    env : TrafficIntersection
    choose_fn : callable(state, queues) → action
    num_episodes : int
    step_fn : callable or None
        Called each step (used by FixedTimerController to advance its internal timer).

    Returns
    -------
    dict
        Averaged metrics.
    """
    all_rewards = []
    all_avg_waits = []
    all_throughputs = []
    all_max_queues = []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        max_queue_seen = 0

        for step in range(cfg.EPISODE_LENGTH):
            action = choose_fn(state, env.get_queue_counts())
            next_state, reward, done, info = env.step(action)

            if step_fn:
                step_fn()

            state = next_state
            total_reward += reward
            max_queue_seen = max(max_queue_seen, max(info["queues"].values()))

            if done:
                break

        total_wait = sum(env.wait_times.values())
        total_arrivals = max(sum(env.total_arrivals.values()), 1)
        total_throughput = sum(env.throughput.values())

        all_rewards.append(total_reward)
        all_avg_waits.append(total_wait / total_arrivals)
        all_throughputs.append(total_throughput)
        all_max_queues.append(max_queue_seen)

    return {
        "avg_reward": np.mean(all_rewards),
        "avg_wait_time": np.mean(all_avg_waits),
        "avg_throughput": np.mean(all_throughputs),
        "avg_max_queue": np.mean(all_max_queues),
    }


def evaluate():
    """Run evaluation and comparison."""
    ensure_dirs()
    env = TrafficIntersection()

    print("=" * 60)
    print("[Traffic Signal] Smart Traffic Signal - Evaluation & Comparison")
    print("=" * 60)

    results = {}

    # ── 1. Q-Learning Agent ────────────────────────────
    print("\n  [*] Evaluating Q-Learning Agent...")
    agent = QLearningAgent(load_path=cfg.BEST_MODEL_PATH)
    agent.epsilon = 0.0  # Fully greedy

    results["Q-Learning"] = evaluate_strategy(
        "Q-Learning", env,
        choose_fn=lambda s, q: agent.choose_action(s, greedy=True),
    )

    # ── 2. Fixed Timer ─────────────────────────────────
    print("  [*] Evaluating Fixed Timer Controller...")
    fixed = FixedTimerController(green_duration=8)

    def fixed_choose(state, queues):
        return fixed.choose_action()

    results["Fixed Timer"] = evaluate_strategy(
        "Fixed Timer", env,
        choose_fn=fixed_choose,
        step_fn=fixed.step,
    )
    fixed.reset()

    # ── 3. Longest Queue ──────────────────────────────
    print("  [*] Evaluating Longest Queue Heuristic...")
    lq = LongestQueueController()

    results["Longest Queue"] = evaluate_strategy(
        "Longest Queue", env,
        choose_fn=lambda s, q: lq.choose_action(queues=q),
    )

    # ── Print results ─────────────────────────────────
    print("\n" + "=" * 60)
    print("[RESULTS] Results Summary")
    print("=" * 60)
    header = f"  {'Strategy':<20s} {'Avg Reward':>12s} {'Avg Wait':>10s} {'Throughput':>12s} {'Max Queue':>10s}"
    print(header)
    print("  " + "─" * 66)
    for name, m in results.items():
        print(f"  {name:<20s} {m['avg_reward']:>12.1f} {m['avg_wait_time']:>10.2f} "
              f"{m['avg_throughput']:>12.1f} {m['avg_max_queue']:>10.1f}")
    print("=" * 60)

    # ── Plot comparison ───────────────────────────────
    comparison_metrics = {}
    for name, m in results.items():
        comparison_metrics[name] = {
            "avg_wait_time": m["avg_wait_time"],
            "avg_throughput": m["avg_throughput"],
            "avg_max_queue": m["avg_max_queue"],
        }
    plot_comparison(comparison_metrics)

    # ── Improvement percentage ────────────────────────
    ql_wait = results["Q-Learning"]["avg_wait_time"]
    ft_wait = results["Fixed Timer"]["avg_wait_time"]
    if ft_wait > 0:
        improvement = ((ft_wait - ql_wait) / ft_wait) * 100
        print(f"\n  [>>] Q-Learning reduces wait time by {improvement:.1f}% vs Fixed Timer")

    return results


if __name__ == "__main__":
    evaluate()
