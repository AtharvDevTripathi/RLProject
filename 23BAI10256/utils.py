"""
Utility functions for state discretisation, encoding, plotting, and logging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config as cfg


# ─────────────────────────────────────────────
# State Helpers
# ─────────────────────────────────────────────

def discretize_queue(count: int) -> int:
    """
    Convert a raw vehicle count into a discrete level.
    Bins defined in config.QUEUE_BINS = [0, 1, 4, 8]
      Level 0: count == 0
      Level 1: 1 <= count <= 3
      Level 2: 4 <= count <= 7
      Level 3: count >= 8
    """
    for i in range(len(cfg.QUEUE_BINS) - 1, -1, -1):
        if count >= cfg.QUEUE_BINS[i]:
            return i
    return 0


def encode_state(queue_counts: dict, current_green: int = 0) -> int:
    """
    Encode discretized queue counts + current green phase into a state index.

    State = queue_encoding * NUM_DIRECTIONS + current_green
    This lets the agent know which direction is currently green,
    which is critical for reasoning about switching costs.

    Parameters
    ----------
    queue_counts : dict
        Raw vehicle counts keyed by direction name.
    current_green : int
        Index of the direction currently having green (0-3).

    Returns
    -------
    int
        State index in [0, NUM_STATES).
    """
    queue_state = 0
    for i, direction in enumerate(cfg.DIRECTIONS):
        level = discretize_queue(queue_counts[direction])
        queue_state += level * (cfg.NUM_QUEUE_LEVELS ** i)
    return queue_state * cfg.NUM_DIRECTIONS + current_green


def decode_state(state: int) -> dict:
    """Decode a state index back into per-direction discrete levels + current green."""
    current_green = state % cfg.NUM_DIRECTIONS
    queue_state = state // cfg.NUM_DIRECTIONS
    levels = {}
    for direction in cfg.DIRECTIONS:
        levels[direction] = queue_state % cfg.NUM_QUEUE_LEVELS
        queue_state //= cfg.NUM_QUEUE_LEVELS
    levels["current_green"] = current_green
    return levels


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_training_curves(rewards, avg_waits, filename="training_curves.png"):
    """Plot episode rewards and average wait times over training."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q-Learning Training Progress", fontsize=15, fontweight="bold")

    # Rewards
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color="#89b4fa", linewidth=0.8)
    # Smoothed
    window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, color="#cba6f7", linewidth=2,
                label=f"Smoothed (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Average wait times
    ax = axes[1]
    ax.plot(avg_waits, alpha=0.3, color="#f38ba8", linewidth=0.8)
    if window > 1:
        smoothed = np.convolve(avg_waits, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(avg_waits)), smoothed, color="#a6e3a1", linewidth=2,
                label=f"Smoothed (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Wait Time (steps)")
    ax.set_title("Average Vehicle Wait Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg.RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Training curves saved to {path}")


def plot_comparison(metrics: dict, filename="comparison.png"):
    """
    Plot bar chart comparing strategies.

    Parameters
    ----------
    metrics : dict
        { strategy_name: { metric_name: value, ... }, ... }
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    strategies = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8"]
    fig.suptitle("Strategy Comparison", fontsize=15, fontweight="bold")

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics[s][metric] for s in strategies]
        bars = ax.bar(strategies, values, color=colors[:len(strategies)], edgecolor="white", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(cfg.RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Comparison chart saved to {path}")


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
