"""
Central configuration for the Smart Traffic Signal RL project.
All hyperparameters, environment settings, and discretization parameters live here.
"""

# ─────────────────────────────────────────────
# Environment Settings
# ─────────────────────────────────────────────

# Number of simulation steps per episode
EPISODE_LENGTH = 500

# Vehicle arrival rates (Poisson λ) per direction per step
# Moderate traffic: system utilisation ~70% so queues ebb and flow
# Total arrivals ~2.8/step vs throughput of 4/step
ARRIVAL_RATES = {
    "north": 0.8,
    "south": 0.8,
    "east":  0.6,
    "west":  0.6,
}

# Maximum number of vehicles that can queue in one direction
MAX_QUEUE = 20

# Vehicles that pass through per step when their light is green
GREEN_THROUGHPUT = 4

# Minimum number of steps a green phase must last before switching
MIN_GREEN_DURATION = 4

# Number of yellow transition steps between phase changes
YELLOW_DURATION = 2

# ─────────────────────────────────────────────
# State Discretisation
# ─────────────────────────────────────────────

# Bin edges for discretizing vehicle counts into levels
# Level 0: 0 vehicles
# Level 1: 1-3 vehicles
# Level 2: 4-7 vehicles
# Level 3: 8+ vehicles
QUEUE_BINS = [0, 1, 4, 8]
NUM_QUEUE_LEVELS = len(QUEUE_BINS)  # 4

# Directions (order matters — keep consistent everywhere)
DIRECTIONS = ["north", "south", "east", "west"]
NUM_DIRECTIONS = len(DIRECTIONS)

# Total state space size (queue levels per direction * current green phase)
# 4^4 * 4 = 1024 states — still tiny for tabular Q-learning
NUM_STATES = (NUM_QUEUE_LEVELS ** NUM_DIRECTIONS) * NUM_DIRECTIONS  # 1024

# ─────────────────────────────────────────────
# Action Space
# ─────────────────────────────────────────────

# Actions = index of direction to set green
# 0 → north, 1 → south, 2 → east, 3 → west
NUM_ACTIONS = NUM_DIRECTIONS  # 4

# ─────────────────────────────────────────────
# Q-Learning Hyperparameters
# ─────────────────────────────────────────────

LEARNING_RATE = 0.1          # α
DISCOUNT_FACTOR = 0.95       # γ
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_END = 0.01           # Minimum exploration rate
EPSILON_DECAY = 0.9990       # Multiplicative decay per episode (slower for more episodes)

# ─────────────────────────────────────────────
# Training Settings
# ─────────────────────────────────────────────

NUM_EPISODES = 2000          # Total training episodes
LOG_INTERVAL = 50            # Print progress every N episodes
SAVE_INTERVAL = 100          # Save model checkpoint every N episodes

# ─────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────

MODEL_DIR = "models"
RESULTS_DIR = "results"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_q_table.npy"

# ─────────────────────────────────────────────
# Pygame Visualisation Settings
# ─────────────────────────────────────────────

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 750
FPS = 10  # Frames/steps per second during visualisation

# Colors (RGB)
COLOR_BG = (30, 30, 46)
COLOR_ROAD = (69, 71, 90)
COLOR_ROAD_MARKING = (205, 214, 244)
COLOR_GRASS = (64, 160, 90)
COLOR_RED = (243, 70, 70)
COLOR_YELLOW = (249, 226, 100)
COLOR_GREEN = (100, 230, 120)
COLOR_CAR_COLORS = [
    (137, 180, 250),   # Blue
    (245, 194, 231),   # Pink
    (166, 227, 161),   # Green
    (250, 179, 135),   # Peach
    (203, 166, 247),   # Mauve
    (249, 226, 175),   # Yellow
    (148, 226, 213),   # Teal
]
COLOR_TEXT = (205, 214, 244)
COLOR_PANEL_BG = (24, 24, 37)
