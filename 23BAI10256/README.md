# 🚦 Smart Traffic Signal Control using Q-Learning

An intelligent traffic signal controller trained with **Reinforcement Learning (Q-Learning)** that adapts in real-time to traffic conditions — unlike traditional fixed-timer signals that stay green even when no vehicles are passing.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![RL](https://img.shields.io/badge/Algorithm-Q--Learning-green)
![Pygame](https://img.shields.io/badge/Visualization-Pygame-orange)

---

## 🎯 Problem Statement

Traditional traffic lights use fixed-duration cycles, wasting green time on empty lanes while vehicles pile up on busy ones. Our Q-Learning agent **observes vehicle queues** on all four approaches (North, South, East, West) and **learns** the optimal signaling policy to minimize total waiting time.

### How It's Different from Traditional Traffic Lights

| Feature | Traditional | Our RL Agent |
|---------|------------|--------------|
| Green duration | Fixed timer (e.g., 30s) | Adaptive based on traffic |
| Observation | None | Vehicle counts per direction |
| Decision making | Round-robin cycle | Q-Learning policy |
| Empty lane handling | Wastes green time | Skips/shortens green |
| Heavy traffic handling | Same as light traffic | Prioritizes congested lanes |

---

## 🏗️ Architecture

```
State:  Discretized vehicle counts per direction → 256 possible states
Action: Set green for N, S, E, or W → 4 possible actions
Reward: -(total waiting vehicles) → encourages minimizing delays
```

The agent uses **tabular Q-Learning** with:
- **ε-greedy exploration** with decay
- **Minimum green duration** to prevent unrealistic flickering
- **Yellow transition phases** between signal changes

---

## 📦 Installation

```bash
# Clone the repository
cd TrafficSignalRL

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Agent

```bash
python train.py
```

This will:
- Train for 1000 episodes
- Save the best Q-table to `models/best_q_table.npy`
- Generate training curves in `results/training_curves.png`

### 2. Evaluate & Compare

```bash
python evaluate.py
```

Compares three strategies:
- **Q-Learning** (trained agent)
- **Fixed Timer** (traditional 8-step cycle)
- **Longest Queue** (greedy heuristic)

### 3. Watch the Visualisation 🎮

```bash
python visualize.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `↑ / ↓` | Speed up / Slow down |
| `R` | Restart episode |
| `Q / ESC` | Quit |

---

## 📁 Project Structure

```
TrafficSignalRL/
├── config.py           # All hyperparameters and settings
├── environment.py      # Custom traffic simulation (gym-like)
├── agent.py            # Q-Learning agent + baseline controllers
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation & comparison
├── visualize.py        # Pygame real-time visualization
├── utils.py            # Helpers (state encoding, plotting)
├── requirements.txt    # Dependencies
├── models/             # Saved Q-tables
└── results/            # Training curves & comparison plots
```

---

## ⚙️ Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ARRIVAL_RATES` | N:2.5, S:2.5, E:2.0, W:2.0 | Vehicle arrival rate (Poisson λ) |
| `MAX_QUEUE` | 20 | Max vehicles per direction |
| `GREEN_THROUGHPUT` | 3 | Vehicles passing per green step |
| `MIN_GREEN_DURATION` | 4 | Minimum green steps before switching |
| `LEARNING_RATE` | 0.1 | Q-Learning α |
| `DISCOUNT_FACTOR` | 0.95 | Q-Learning γ |
| `NUM_EPISODES` | 1000 | Training episodes |

---

## 📊 Results

After training, the Q-Learning agent typically achieves:
- **30-50% reduction** in average wait time vs. fixed-timer
- **Higher throughput** during moderate traffic conditions
- **Fairer** distribution of green time based on actual demand

---

## 🧠 How Q-Learning Works Here

1. **State**: Vehicle counts on each direction are discretized into 4 levels (0, 1-3, 4-7, 8+), giving 4⁴=256 states
2. **Action**: Choose which direction gets green (4 actions)
3. **Reward**: Negative sum of all waiting vehicles (minimise total delay)
4. **Update Rule**: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]`
5. **Exploration**: ε-greedy with decay from 1.0 → 0.01 over training

---

## 📄 License

This project is for educational purposes. Feel free to use and modify.
