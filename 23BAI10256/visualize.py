"""
Pygame Real-Time Visualisation for the Smart Traffic Signal RL.

Renders a 4-way intersection with:
  - Roads, lane markings, and grass
  - Vehicles as colored rectangles queuing and departing
  - Traffic signal lights (red/yellow/green circles)
  - Info panel with live stats, Q-values, and controls
  - Speed controls: SPACE=pause, UP/DOWN=speed, R=restart
"""

import sys
import os
import math
import random
import pygame
import numpy as np
import config as cfg
from environment import TrafficIntersection
from agent import QLearningAgent


# ─────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────

ROAD_WIDTH = 140
LANE_WIDTH = ROAD_WIDTH // 2
INTERSECTION_SIZE = ROAD_WIDTH
CAR_W, CAR_H = 28, 50  # Width, Height of a car rectangle (vertical orientation)
CAR_GAP = 6             # Gap between queued cars
SIGNAL_RADIUS = 14
PANEL_WIDTH = 280

# Center of the intersection
CX = (cfg.WINDOW_WIDTH - PANEL_WIDTH) // 2
CY = cfg.WINDOW_HEIGHT // 2


def run_visualisation(model_path=None):
    """Launch the Pygame visualisation window."""
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    pygame.display.set_caption("Smart Traffic Signal - Q-Learning RL")
    clock = pygame.time.Clock()

    # Load fonts
    try:
        font_path = None
        font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
        font_main = pygame.font.SysFont("Segoe UI", 16)
        font_small = pygame.font.SysFont("Segoe UI", 13)
        font_big = pygame.font.SysFont("Segoe UI", 28, bold=True)
    except Exception:
        font_title = pygame.font.Font(None, 26)
        font_main = pygame.font.Font(None, 20)
        font_small = pygame.font.Font(None, 16)
        font_big = pygame.font.Font(None, 32)

    # Setup environment and agent
    env = TrafficIntersection()
    path = model_path or cfg.BEST_MODEL_PATH

    if os.path.exists(path):
        agent = QLearningAgent(load_path=path)
        agent.epsilon = 0.0
        print(f"  [OK] Loaded trained model from {path}")
    else:
        print(f"  [WARN] No model found at {path} - using random agent")
        agent = QLearningAgent()
        agent.epsilon = 1.0  # Fully random

    state = env.reset()

    # Assign persistent colors to car slots
    car_colors = {}  # (direction, slot) → color
    color_counter = 0

    # Animation state
    paused = False
    speed = cfg.FPS
    total_reward = 0.0
    step_num = 0
    episode_num = 1

    running = True
    while running:
        # ── Event handling ─────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = min(60, speed + 2)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 2)
                elif event.key == pygame.K_r:
                    # Restart episode
                    state = env.reset()
                    total_reward = 0.0
                    step_num = 0
                    episode_num += 1
                    car_colors.clear()

        if not paused:
            # ── Agent step ─────────────────────────
            action = agent.choose_action(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            step_num += 1

            if done:
                state = env.reset()
                total_reward = 0.0
                step_num = 0
                episode_num += 1
                car_colors.clear()

        # ── Drawing ────────────────────────────────
        screen.fill(cfg.COLOR_BG)

        draw_grass(screen)
        draw_roads(screen)
        draw_lane_markings(screen)
        draw_crosswalks(screen)
        draw_signals(screen, env)
        draw_vehicles(screen, env, car_colors, color_counter)
        draw_info_panel(screen, env, agent, state, total_reward, step_num,
                        episode_num, speed, paused,
                        font_title, font_main, font_small, font_big)

        pygame.display.flip()
        clock.tick(speed)

    pygame.quit()


# ─────────────────────────────────────────────────
# Drawing Functions
# ─────────────────────────────────────────────────

def draw_grass(screen):
    """Draw grass areas in the four quadrants."""
    hw = ROAD_WIDTH // 2
    # Top-left
    pygame.draw.rect(screen, cfg.COLOR_GRASS, (0, 0, CX - hw, CY - hw))
    # Top-right
    pygame.draw.rect(screen, cfg.COLOR_GRASS, (CX + hw, 0, cfg.WINDOW_WIDTH - PANEL_WIDTH - CX - hw, CY - hw))
    # Bottom-left
    pygame.draw.rect(screen, cfg.COLOR_GRASS, (0, CY + hw, CX - hw, cfg.WINDOW_HEIGHT - CY - hw))
    # Bottom-right
    pygame.draw.rect(screen, cfg.COLOR_GRASS, (CX + hw, CY + hw, cfg.WINDOW_WIDTH - PANEL_WIDTH - CX - hw, cfg.WINDOW_HEIGHT - CY - hw))


def draw_roads(screen):
    """Draw the horizontal and vertical roads."""
    hw = ROAD_WIDTH // 2
    road_area_w = cfg.WINDOW_WIDTH - PANEL_WIDTH

    # Vertical road
    pygame.draw.rect(screen, cfg.COLOR_ROAD, (CX - hw, 0, ROAD_WIDTH, cfg.WINDOW_HEIGHT))
    # Horizontal road
    pygame.draw.rect(screen, cfg.COLOR_ROAD, (0, CY - hw, road_area_w, ROAD_WIDTH))


def draw_lane_markings(screen):
    """Draw dashed center line markings on the roads."""
    hw = ROAD_WIDTH // 2
    road_area_w = cfg.WINDOW_WIDTH - PANEL_WIDTH
    dash_len = 20
    gap_len = 15
    color = (cfg.COLOR_ROAD_MARKING[0], cfg.COLOR_ROAD_MARKING[1], cfg.COLOR_ROAD_MARKING[2], 100)

    # Vertical center line (above intersection)
    y = 0
    while y < CY - hw:
        pygame.draw.line(screen, cfg.COLOR_ROAD_MARKING, (CX, y), (CX, min(y + dash_len, CY - hw)), 2)
        y += dash_len + gap_len

    # Vertical center line (below intersection)
    y = CY + hw
    while y < cfg.WINDOW_HEIGHT:
        pygame.draw.line(screen, cfg.COLOR_ROAD_MARKING, (CX, y), (CX, min(y + dash_len, cfg.WINDOW_HEIGHT)), 2)
        y += dash_len + gap_len

    # Horizontal center line (left of intersection)
    x = 0
    while x < CX - hw:
        pygame.draw.line(screen, cfg.COLOR_ROAD_MARKING, (x, CY), (min(x + dash_len, CX - hw), CY), 2)
        x += dash_len + gap_len

    # Horizontal center line (right of intersection)
    x = CX + hw
    while x < road_area_w:
        pygame.draw.line(screen, cfg.COLOR_ROAD_MARKING, (x, CY), (min(x + dash_len, road_area_w), CY), 2)
        x += dash_len + gap_len


def draw_crosswalks(screen):
    """Draw crosswalk stripes at the intersection edges."""
    hw = ROAD_WIDTH // 2
    stripe_w = 8
    stripe_gap = 6
    stripe_color = (205, 214, 244, 80)

    # North crosswalk (above intersection)
    for i in range(ROAD_WIDTH // (stripe_w + stripe_gap)):
        x = CX - hw + i * (stripe_w + stripe_gap)
        pygame.draw.rect(screen, cfg.COLOR_ROAD_MARKING,
                         (x, CY - hw - 12, stripe_w, 10))

    # South crosswalk
    for i in range(ROAD_WIDTH // (stripe_w + stripe_gap)):
        x = CX - hw + i * (stripe_w + stripe_gap)
        pygame.draw.rect(screen, cfg.COLOR_ROAD_MARKING,
                         (x, CY + hw + 2, stripe_w, 10))

    # West crosswalk
    for i in range(ROAD_WIDTH // (stripe_w + stripe_gap)):
        y = CY - hw + i * (stripe_w + stripe_gap)
        pygame.draw.rect(screen, cfg.COLOR_ROAD_MARKING,
                         (CX - hw - 12, y, 10, stripe_w))

    # East crosswalk
    for i in range(ROAD_WIDTH // (stripe_w + stripe_gap)):
        y = CY - hw + i * (stripe_w + stripe_gap)
        pygame.draw.rect(screen, cfg.COLOR_ROAD_MARKING,
                         (CX + hw + 2, y, 10, stripe_w))


def draw_signals(screen, env):
    """Draw traffic signal lights at each intersection corner."""
    signals = env.get_signal_state()
    hw = ROAD_WIDTH // 2
    offset = 20  # Distance from intersection edge

    positions = {
        "north": (CX + hw + offset, CY - hw - offset),    # Top-right corner area
        "south": (CX - hw - offset, CY + hw + offset),    # Bottom-left corner area
        "east":  (CX + hw + offset, CY + hw + offset),    # Bottom-right corner area
        "west":  (CX - hw - offset, CY - hw - offset),    # Top-left corner area
    }

    for direction, state in signals.items():
        x, y = positions[direction]

        # Signal housing (dark rectangle)
        housing_w, housing_h = 22, 58
        pygame.draw.rect(screen, (20, 20, 30),
                         (x - housing_w // 2, y - housing_h // 2, housing_w, housing_h),
                         border_radius=5)
        pygame.draw.rect(screen, (60, 60, 80),
                         (x - housing_w // 2, y - housing_h // 2, housing_w, housing_h),
                         width=2, border_radius=5)

        # Three circles: red, yellow, green
        r_center = (x, y - 18)
        y_center = (x, y)
        g_center = (x, y + 18)

        # Dim versions
        dim_red = (80, 25, 25)
        dim_yellow = (80, 72, 30)
        dim_green = (25, 70, 35)

        if state == "red":
            pygame.draw.circle(screen, cfg.COLOR_RED, r_center, SIGNAL_RADIUS)
            pygame.draw.circle(screen, dim_yellow, y_center, SIGNAL_RADIUS)
            pygame.draw.circle(screen, dim_green, g_center, SIGNAL_RADIUS)
            # Glow effect
            glow_surf = pygame.Surface((SIGNAL_RADIUS * 4, SIGNAL_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*cfg.COLOR_RED, 40),
                               (SIGNAL_RADIUS * 2, SIGNAL_RADIUS * 2), SIGNAL_RADIUS * 2)
            screen.blit(glow_surf, (r_center[0] - SIGNAL_RADIUS * 2, r_center[1] - SIGNAL_RADIUS * 2))
        elif state == "yellow":
            pygame.draw.circle(screen, dim_red, r_center, SIGNAL_RADIUS)
            pygame.draw.circle(screen, cfg.COLOR_YELLOW, y_center, SIGNAL_RADIUS)
            pygame.draw.circle(screen, dim_green, g_center, SIGNAL_RADIUS)
            glow_surf = pygame.Surface((SIGNAL_RADIUS * 4, SIGNAL_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*cfg.COLOR_YELLOW, 40),
                               (SIGNAL_RADIUS * 2, SIGNAL_RADIUS * 2), SIGNAL_RADIUS * 2)
            screen.blit(glow_surf, (y_center[0] - SIGNAL_RADIUS * 2, y_center[1] - SIGNAL_RADIUS * 2))
        else:  # green
            pygame.draw.circle(screen, dim_red, r_center, SIGNAL_RADIUS)
            pygame.draw.circle(screen, dim_yellow, y_center, SIGNAL_RADIUS)
            pygame.draw.circle(screen, cfg.COLOR_GREEN, g_center, SIGNAL_RADIUS)
            glow_surf = pygame.Surface((SIGNAL_RADIUS * 4, SIGNAL_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*cfg.COLOR_GREEN, 40),
                               (SIGNAL_RADIUS * 2, SIGNAL_RADIUS * 2), SIGNAL_RADIUS * 2)
            screen.blit(glow_surf, (g_center[0] - SIGNAL_RADIUS * 2, g_center[1] - SIGNAL_RADIUS * 2))


def draw_vehicles(screen, env, car_colors, color_counter):
    """Draw queued vehicles as colored rounded rectangles."""
    queues = env.get_queue_counts()
    hw = ROAD_WIDTH // 2
    lane_offset = LANE_WIDTH // 2  # Cars drive on the left lane (incoming)

    for i, direction in enumerate(cfg.DIRECTIONS):
        count = queues[direction]
        for slot in range(count):
            # Assign a persistent color for this slot
            key = (direction, slot)
            if key not in car_colors:
                car_colors[key] = random.choice(cfg.COLOR_CAR_COLORS)
            color = car_colors[key]

            # Position based on direction
            if direction == "north":
                # Cars come from top, queue above intersection in RIGHT lane (going down)
                cx = CX + lane_offset
                cy = CY - hw - 20 - slot * (CAR_H + CAR_GAP)
                w, h = CAR_W, CAR_H
            elif direction == "south":
                # Cars come from bottom, queue below intersection in LEFT lane (going up)
                cx = CX - lane_offset
                cy = CY + hw + 20 + slot * (CAR_H + CAR_GAP)
                w, h = CAR_W, CAR_H
            elif direction == "east":
                # Cars come from right, queue to the right of intersection
                cx = CX + hw + 20 + slot * (CAR_H + CAR_GAP)
                cy = CY + lane_offset
                w, h = CAR_H, CAR_W  # Horizontal car
            elif direction == "west":
                # Cars come from left, queue to the left of intersection
                cx = CX - hw - 20 - slot * (CAR_H + CAR_GAP)
                cy = CY - lane_offset
                w, h = CAR_H, CAR_W  # Horizontal car

            # Draw car body
            rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
            pygame.draw.rect(screen, color, rect, border_radius=6)

            # Car window (darker strip)
            if direction in ("north", "south"):
                win_rect = pygame.Rect(cx - w // 2 + 4, cy - 6, w - 8, 12)
            else:
                win_rect = pygame.Rect(cx - 6, cy - h // 2 + 4, 12, h - 8)
            darker = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
            pygame.draw.rect(screen, darker, win_rect, border_radius=3)


def draw_info_panel(screen, env, agent, state, total_reward, step_num,
                    episode_num, speed, paused,
                    font_title, font_main, font_small, font_big):
    """Draw the right-side information panel."""
    panel_x = cfg.WINDOW_WIDTH - PANEL_WIDTH
    panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, cfg.WINDOW_HEIGHT)
    pygame.draw.rect(screen, cfg.COLOR_PANEL_BG, panel_rect)

    # Divider line
    pygame.draw.line(screen, (60, 60, 80), (panel_x, 0), (panel_x, cfg.WINDOW_HEIGHT), 2)

    x = panel_x + 18
    y = 20

    # Title
    title = font_big.render("Traffic RL", True, cfg.COLOR_TEXT)
    screen.blit(title, (x, y))
    y += 45

    # Separator
    pygame.draw.line(screen, (60, 60, 80), (x, y), (panel_x + PANEL_WIDTH - 18, y))
    y += 15

    # Episode / Step
    labels = [
        ("Episode", str(episode_num)),
        ("Step", f"{step_num}/{cfg.EPISODE_LENGTH}"),
        ("Speed", f"{speed} FPS"),
        ("Status", "⏸ PAUSED" if paused else "▶ Running"),
    ]

    for label, value in labels:
        text = font_main.render(f"{label}:", True, (150, 150, 170))
        val = font_main.render(value, True, cfg.COLOR_TEXT)
        screen.blit(text, (x, y))
        screen.blit(val, (x + 100, y))
        y += 24

    y += 10
    pygame.draw.line(screen, (60, 60, 80), (x, y), (panel_x + PANEL_WIDTH - 18, y))
    y += 15

    # Queue counts
    header = font_title.render("Queue Counts", True, cfg.COLOR_TEXT)
    screen.blit(header, (x, y))
    y += 30

    queues = env.get_queue_counts()
    signals = env.get_signal_state()
    bar_max_w = 140

    for direction in cfg.DIRECTIONS:
        count = queues[direction]
        sig = signals[direction]

        # Direction label
        label = font_main.render(f"{direction.capitalize()[:1]}:", True, (150, 150, 170))
        screen.blit(label, (x, y + 2))

        # Signal indicator dot
        sig_color = {"red": cfg.COLOR_RED, "yellow": cfg.COLOR_YELLOW, "green": cfg.COLOR_GREEN}[sig]
        pygame.draw.circle(screen, sig_color, (x + 25, y + 11), 6)

        # Queue bar
        bar_x = x + 40
        bar_y = y + 3
        bar_h = 16
        bar_bg = pygame.Rect(bar_x, bar_y, bar_max_w, bar_h)
        pygame.draw.rect(screen, (50, 50, 65), bar_bg, border_radius=3)

        fill_w = int((count / cfg.MAX_QUEUE) * bar_max_w)
        if fill_w > 0:
            # Color gradient: green → yellow → red based on fill
            ratio = count / cfg.MAX_QUEUE
            if ratio < 0.4:
                bar_color = cfg.COLOR_GREEN
            elif ratio < 0.7:
                bar_color = cfg.COLOR_YELLOW
            else:
                bar_color = cfg.COLOR_RED
            fill_rect = pygame.Rect(bar_x, bar_y, fill_w, bar_h)
            pygame.draw.rect(screen, bar_color, fill_rect, border_radius=3)

        # Count text
        ct = font_small.render(str(count), True, cfg.COLOR_TEXT)
        screen.blit(ct, (bar_x + bar_max_w + 8, y + 2))

        y += 26

    y += 10
    pygame.draw.line(screen, (60, 60, 80), (x, y), (panel_x + PANEL_WIDTH - 18, y))
    y += 15

    # Reward
    reward_header = font_title.render("Performance", True, cfg.COLOR_TEXT)
    screen.blit(reward_header, (x, y))
    y += 28

    rew_text = font_main.render(f"Total Reward:", True, (150, 150, 170))
    rew_val = font_main.render(f"{total_reward:.0f}", True,
                                cfg.COLOR_GREEN if total_reward > -500 else cfg.COLOR_RED)
    screen.blit(rew_text, (x, y))
    screen.blit(rew_val, (x + 120, y))
    y += 24

    # Total throughput
    tp = sum(env.throughput.values())
    tp_text = font_main.render(f"Throughput:", True, (150, 150, 170))
    tp_val = font_main.render(f"{tp} vehicles", True, cfg.COLOR_TEXT)
    screen.blit(tp_text, (x, y))
    screen.blit(tp_val, (x + 120, y))
    y += 24

    # Total wait
    tw = sum(env.wait_times.values())
    tw_text = font_main.render(f"Total Wait:", True, (150, 150, 170))
    tw_val = font_main.render(f"{tw:.0f}", True, cfg.COLOR_TEXT)
    screen.blit(tw_text, (x, y))
    screen.blit(tw_val, (x + 120, y))
    y += 30

    # Q-Values
    pygame.draw.line(screen, (60, 60, 80), (x, y), (panel_x + PANEL_WIDTH - 18, y))
    y += 15
    qv_header = font_title.render("Q-Values", True, cfg.COLOR_TEXT)
    screen.blit(qv_header, (x, y))
    y += 28

    q_vals = agent.get_q_values(state)
    max_q = np.max(q_vals)
    for i, direction in enumerate(cfg.DIRECTIONS):
        is_best = (q_vals[i] == max_q)
        color = cfg.COLOR_GREEN if is_best else (150, 150, 170)
        marker = "★" if is_best else " "
        text = font_main.render(
            f"{marker} {direction.capitalize()[:5]:>5s}: {q_vals[i]:>8.2f}", True, color)
        screen.blit(text, (x, y))
        y += 22

    y += 20
    pygame.draw.line(screen, (60, 60, 80), (x, y), (panel_x + PANEL_WIDTH - 18, y))
    y += 15

    # Controls
    ctrl_header = font_title.render("Controls", True, cfg.COLOR_TEXT)
    screen.blit(ctrl_header, (x, y))
    y += 26

    controls = [
        "SPACE  — Pause / Resume",
        "↑ / ↓  — Speed up / down",
        "R      — Restart episode",
        "Q/ESC  — Quit",
    ]
    for ctrl in controls:
        text = font_small.render(ctrl, True, (120, 120, 140))
        screen.blit(text, (x, y))
        y += 20


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    run_visualisation(model)
