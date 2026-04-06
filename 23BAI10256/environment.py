"""
Custom Traffic Intersection Simulation Environment.

Models a single 4-way intersection with vehicle arrivals (Poisson),
queuing, signal phases, and realistic yellow transitions.
Provides a gym-like interface: reset(), step(action) → (state, reward, done, info).
"""

import numpy as np
import config as cfg
from utils import encode_state


class TrafficIntersection:
    """
    Simulates a 4-way intersection with traffic signals.

    State:  Discretized vehicle queue counts per direction (encoded as int).
    Action: Index of direction to set green (0=N, 1=S, 2=E, 3=W).
    Reward: Negative total waiting time across all directions (encourages
            the agent to reduce cumulative delays).
    """

    def __init__(self, arrival_rates=None, episode_length=None):
        self.arrival_rates = arrival_rates or cfg.ARRIVAL_RATES
        self.episode_length = episode_length or cfg.EPISODE_LENGTH
        self.rng = np.random.default_rng()
        self.reset()

    # ── Gym-like interface ─────────────────────────────

    def reset(self):
        """Reset the environment to the initial state."""
        # Queues: number of waiting vehicles per direction
        self.queues = {d: 0 for d in cfg.DIRECTIONS}

        # Cumulative wait time per direction (sum of queue × steps)
        self.wait_times = {d: 0.0 for d in cfg.DIRECTIONS}

        # Total vehicles that have passed through
        self.throughput = {d: 0 for d in cfg.DIRECTIONS}

        # Total vehicles that arrived
        self.total_arrivals = {d: 0 for d in cfg.DIRECTIONS}

        # Signal state
        self.current_green = 0  # Index into DIRECTIONS
        self.green_timer = 0    # How many steps current green has been active
        self.yellow_active = False
        self.yellow_timer = 0
        self.pending_green = None  # Direction waiting after yellow

        # Step counter
        self.step_count = 0
        self.done = False

        return self._get_state()

    def step(self, action: int):
        """
        Execute one simulation step.

        Parameters
        ----------
        action : int
            Index of direction to set green (0–3).

        Returns
        -------
        state : int
            Encoded state index.
        reward : float
            Reward for this step.
        done : bool
            Whether the episode has ended.
        info : dict
            Additional info (queues, throughput, etc.).
        """
        assert 0 <= action < cfg.NUM_ACTIONS, f"Invalid action {action}"

        # ── 1. Handle signal transitions ────────────────
        switching = self._handle_signal(action)

        # ── 2. Arrive new vehicles ──────────────────────
        self._arrive_vehicles()

        # ── 3. Process green direction (vehicles depart) ─
        departed = self._depart_vehicles()

        # ── 4. Accumulate waiting time ──────────────────
        total_wait = 0
        for d in cfg.DIRECTIONS:
            self.wait_times[d] += self.queues[d]
            total_wait += self.queues[d]

        # ── 5. Compute reward ───────────────────────────
        # Core: reward throughput, penalize total queue length
        reward = 0.0

        # Positive reward for each vehicle that passes through
        reward += departed * 4.0

        # Penalty proportional to total queue length
        reward -= total_wait * 0.5

        # Penalty for switching (yellow = wasted time where nobody moves)
        if switching:
            reward -= 3.0

        # ── 6. Advance step ─────────────────────────────
        self.step_count += 1
        self.done = self.step_count >= self.episode_length

        state = self._get_state()
        info = {
            "queues": dict(self.queues),
            "throughput": dict(self.throughput),
            "wait_times": dict(self.wait_times),
            "current_green": self.current_green,
            "yellow_active": self.yellow_active,
            "step": self.step_count,
        }

        return state, reward, self.done, info

    # ── Internal methods ───────────────────────────────

    def _handle_signal(self, action: int) -> bool:
        """Manage signal phase transitions with yellow and minimum green.
        Returns True if a new switch was initiated this step."""
        if self.yellow_active:
            # Currently in yellow transition -- count down
            self.yellow_timer += 1
            if self.yellow_timer >= cfg.YELLOW_DURATION:
                # Yellow over -> switch to pending green
                self.current_green = self.pending_green
                self.green_timer = 0
                self.yellow_active = False
                self.yellow_timer = 0
                self.pending_green = None
            return False  # Can't change action during yellow

        # Green is active
        self.green_timer += 1

        if action != self.current_green and self.green_timer >= cfg.MIN_GREEN_DURATION:
            # Agent wants to switch AND minimum green duration met -> start yellow
            self.yellow_active = True
            self.yellow_timer = 0
            self.pending_green = action
            return True
        return False

    def _arrive_vehicles(self):
        """Add vehicles according to Poisson arrival rates."""
        for d in cfg.DIRECTIONS:
            arrivals = self.rng.poisson(self.arrival_rates[d])
            self.queues[d] = min(self.queues[d] + arrivals, cfg.MAX_QUEUE)
            self.total_arrivals[d] += arrivals

    def _depart_vehicles(self) -> int:
        """Let vehicles through on the green direction (unless yellow).
        Returns the number of vehicles that departed."""
        if self.yellow_active:
            return 0  # No one moves during yellow

        green_dir = cfg.DIRECTIONS[self.current_green]
        departed = min(self.queues[green_dir], cfg.GREEN_THROUGHPUT)
        self.queues[green_dir] -= departed
        self.throughput[green_dir] += departed
        return departed

    def _get_state(self) -> int:
        """Return the encoded state from current queue counts + current green phase."""
        return encode_state(self.queues, self.current_green)

    # ── Utility ────────────────────────────────────────

    def get_queue_counts(self) -> dict:
        """Return raw queue counts (for visualisation / logging)."""
        return dict(self.queues)

    def get_signal_state(self) -> dict:
        """Return signal state for visualisation."""
        signals = {}
        for i, d in enumerate(cfg.DIRECTIONS):
            if self.yellow_active:
                if i == self.current_green:
                    signals[d] = "yellow"  # Old green → yellow
                elif i == self.pending_green:
                    signals[d] = "red"     # Pending is still red
                else:
                    signals[d] = "red"
            else:
                signals[d] = "green" if i == self.current_green else "red"
        return signals
