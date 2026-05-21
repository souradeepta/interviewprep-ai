#!/usr/bin/env python3
"""
Daily Load Pattern Animation
Shows 24-hour traffic pattern with capacity visualization.

Usage:
    python3 02-daily-load-animation.py --output output.gif --peak-qps 3500 --capacity 5000
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse


def create_load_over_time_animation(
    output_path: str,
    title: str = "Daily Request Load",
    peak_qps: int = 3500,
    capacity_qps: int = 5000,
    peak_hour: int = 13,
    figsize: tuple = (12, 8)
):
    """
    Animate realistic request load pattern over 24 hours.

    Args:
        output_path: Output GIF file path
        title: Animation title
        peak_qps: Peak requests per minute
        capacity_qps: System capacity in QPS
        peak_hour: Hour of peak load (0-23)
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Simulate realistic load pattern
    hours = np.linspace(0, 24, 240)
    # Peak during specified hour
    base_load = 200 + peak_qps * np.exp(-((hours - peak_hour)**2) / 8)
    noise = np.random.normal(0, 50, len(hours))
    requests_per_minute = np.maximum(base_load + noise, 50)

    # Concurrent users (roughly proportional to request rate)
    concurrent_users = requests_per_minute / 5 + np.random.normal(0, 5, len(hours))
    concurrent_users = np.maximum(concurrent_users, 10)

    # Capacity lines
    capacity_concurrent = 10000

    # Plot 1: Requests over time
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, capacity_qps + 500)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Requests/min')
    ax1.set_title(title)
    ax1.axhline(y=capacity_qps/60, color='r', linestyle='--', label=f'Capacity ({capacity_qps} QPS)', linewidth=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Concurrent users
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, capacity_concurrent + 1000)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Concurrent Users')
    ax2.set_title('Concurrent Connection Load')
    ax2.axhline(y=capacity_concurrent, color='r', linestyle='--', label=f'Capacity ({capacity_concurrent} users)', linewidth=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Animation elements
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Current Load')
    fill1 = None
    line2, = ax2.plot([], [], 'g-', linewidth=2)
    fill2 = None

    ax1.legend(loc='upper left')

    def animate(frame):
        nonlocal fill1, fill2
        end_idx = int(frame * len(hours) / 120)

        if end_idx > 0:
            line1.set_data(hours[:end_idx], requests_per_minute[:end_idx])
            line2.set_data(hours[:end_idx], concurrent_users[:end_idx])

            # Remove old fills and create new ones
            if fill1:
                fill1.remove()
            if fill2:
                fill2.remove()

            fill1 = ax1.fill_between(hours[:end_idx], requests_per_minute[:end_idx], alpha=0.3)
            fill2 = ax2.fill_between(hours[:end_idx], concurrent_users[:end_idx], alpha=0.3)

        return line1, line2

    anim = animation.FuncAnimation(fig, animate, frames=120, interval=50, blit=False, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate daily load pattern animation')
    parser.add_argument('--output', default='02-daily-load.gif', help='Output GIF path')
    parser.add_argument('--title', default='Daily Request Load', help='Animation title')
    parser.add_argument('--peak-qps', type=int, default=3500, help='Peak QPS')
    parser.add_argument('--capacity-qps', type=int, default=5000, help='System capacity QPS')
    parser.add_argument('--peak-hour', type=int, default=13, help='Hour of peak load (0-23)')

    args = parser.parse_args()

    create_load_over_time_animation(args.output, args.title, args.peak_qps, args.capacity_qps, args.peak_hour)
