"""
Tabular Q-Learning Agent for traffic signal control.

Maintains a Q-table of shape (NUM_STATES, NUM_ACTIONS) and updates it
using the standard Q-learning rule with epsilon-greedy exploration.
"""

import os
import numpy as np
import config as cfg


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
    """

    def __init__(self, load_path=None):
        self.num_states = cfg.NUM_STATES
        self.num_actions = cfg.NUM_ACTIONS
        self.lr = cfg.LEARNING_RATE
        self.gamma = cfg.DISCOUNT_FACTOR
        self.epsilon = cfg.EPSILON_START
        self.epsilon_min = cfg.EPSILON_END
        self.epsilon_decay = cfg.EPSILON_DECAY

        # Initialize Q-table with small random values (breaks symmetry)
        self.q_table = np.random.uniform(low=-0.01, high=0.01,
                                          size=(self.num_states, self.num_actions))

        if load_path and os.path.exists(load_path):
            self.load(load_path)
            print(f"  [OK] Loaded Q-table from {load_path}")

    def choose_action(self, state: int, greedy: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : int
            Current state index.
        greedy : bool
            If True, always pick the best action (no exploration).

        Returns
        -------
        int
            Chosen action index.
        """
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Update the Q-table using the Q-learning rule.

        Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
        """
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path=None):
        """Save Q-table to disk."""
        path = path or cfg.BEST_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path=None):
        """Load Q-table from disk."""
        path = path or cfg.BEST_MODEL_PATH
        self.q_table = np.load(path)

    def get_q_values(self, state: int) -> np.ndarray:
        """Return Q-values for a given state (useful for visualisation)."""
        return self.q_table[state].copy()


class FixedTimerController:
    """
    Baseline: Fixed-cycle traffic controller.
    Cycles through directions with a fixed green duration.
    """

    def __init__(self, green_duration=8):
        self.green_duration = green_duration
        self.current_phase = 0
        self.timer = 0

    def choose_action(self, state: int = None, greedy: bool = True) -> int:
        """Return current phase direction."""
        return self.current_phase

    def step(self):
        """Advance one step — switch phase when timer expires."""
        self.timer += 1
        if self.timer >= self.green_duration:
            self.timer = 0
            self.current_phase = (self.current_phase + 1) % cfg.NUM_ACTIONS

    def reset(self):
        self.current_phase = 0
        self.timer = 0


class LongestQueueController:
    """
    Baseline: Always give green to the direction with the longest queue.
    """

    def choose_action(self, state: int = None, greedy: bool = True, queues: dict = None) -> int:
        """Return direction with the most vehicles."""
        if queues is None:
            return 0
        max_queue = -1
        best_dir = 0
        for i, d in enumerate(cfg.DIRECTIONS):
            if queues[d] > max_queue:
                max_queue = queues[d]
                best_dir = i
        return best_dir
