"""
Training pipeline for the Q-Learning traffic signal agent.

Runs the agent through multiple episodes, tracks rewards and wait times,
decays epsilon, and saves the best model.
"""

import time
import numpy as np
import config as cfg
from environment import TrafficIntersection
from agent import QLearningAgent
from utils import ensure_dirs, plot_training_curves


def train():
    """Main training loop."""
    ensure_dirs()
    env = TrafficIntersection()
    agent = QLearningAgent()

    # Tracking
    episode_rewards = []
    episode_avg_waits = []
    best_reward = -float("inf")

    print("=" * 60)
    print("[Traffic Signal] Smart Traffic Signal - Q-Learning Training")
    print("=" * 60)
    print(f"  States:   {cfg.NUM_STATES}")
    print(f"  Actions:  {cfg.NUM_ACTIONS}")
    print(f"  Episodes: {cfg.NUM_EPISODES}")
    print(f"  ε start:  {cfg.EPSILON_START} → {cfg.EPSILON_END}")
    print(f"  α = {cfg.LEARNING_RATE}, γ = {cfg.DISCOUNT_FACTOR}")
    print("=" * 60)

    start_time = time.time()

    for episode in range(1, cfg.NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(cfg.EPISODE_LENGTH):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        # End of episode
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        # Average wait time = total wait / total arrivals
        total_wait = sum(env.wait_times.values())
        total_arrivals = max(sum(env.total_arrivals.values()), 1)
        avg_wait = total_wait / total_arrivals
        episode_avg_waits.append(avg_wait)

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(cfg.BEST_MODEL_PATH)

        # Periodic checkpoint
        if episode % cfg.SAVE_INTERVAL == 0:
            checkpoint_path = f"{cfg.MODEL_DIR}/q_table_ep{episode}.npy"
            agent.save(checkpoint_path)

        # Logging
        if episode % cfg.LOG_INTERVAL == 0:
            recent_rewards = episode_rewards[-cfg.LOG_INTERVAL:]
            recent_waits = episode_avg_waits[-cfg.LOG_INTERVAL:]
            elapsed = time.time() - start_time
            print(f"  Episode {episode:>4d}/{cfg.NUM_EPISODES} │ "
                  f"ε={agent.epsilon:.4f} │ "
                  f"Avg Reward={np.mean(recent_rewards):>8.1f} │ "
                  f"Avg Wait={np.mean(recent_waits):>5.1f} │ "
                  f"Best={best_reward:>8.1f} │ "
                  f"Time={elapsed:.1f}s")

    # ── Training complete ──────────────────────────────
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"[DONE] Training complete in {total_time:.1f}s")
    print(f"   Best reward: {best_reward:.1f}")
    print(f"   Final ε:     {agent.epsilon:.4f}")
    print(f"   Model saved: {cfg.BEST_MODEL_PATH}")
    print("=" * 60)

    # Plot training curves
    plot_training_curves(episode_rewards, episode_avg_waits)

    return agent, episode_rewards, episode_avg_waits


if __name__ == "__main__":
    train()
