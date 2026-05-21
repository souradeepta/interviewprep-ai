#!/usr/bin/env python3
"""
Cost Timeline Animation
Shows daily API cost accumulation based on token usage.

Usage:
    python3 04-cost-timeline-animation.py --output output.gif --cost-per-1k 0.001 --tokens-per-request 4000
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse


def create_cost_over_time_animation(
    output_path: str,
    title: str = "Daily API Cost Estimation",
    peak_qps: int = 3500,
    peak_hour: int = 13,
    tokens_per_request: int = 4000,
    cost_per_1k_tokens: float = 0.001,
    figsize: tuple = (14, 5)
):
    """
    Animate cost accumulation over 24 hours.

    Args:
        output_path: Output GIF file path
        title: Animation title
        peak_qps: Peak requests per minute
        peak_hour: Hour of peak load (0-23)
        tokens_per_request: Average tokens per request
        cost_per_1k_tokens: Cost per 1000 tokens
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Simulate token usage based on request load
    hours = np.linspace(0, 24, 240)
    requests = 200 + peak_qps * np.exp(-((hours - peak_hour)**2) / 8) + np.random.normal(0, 50, len(hours))
    requests = np.maximum(requests, 50)

    # Token usage and cost
    total_tokens = requests * tokens_per_request
    hourly_cost = total_tokens * cost_per_1k_tokens / 60  # Convert to per-minute cost
    cumulative_cost = np.cumsum(hourly_cost) / 60  # Cumulative hourly cost

    # Left plot: Hourly costs
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, max(hourly_cost) * 1.2)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Hourly Cost')
    ax1.grid(True, alpha=0.3)

    # Right plot: Cumulative cost
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, max(cumulative_cost) * 1.1)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Cumulative Cost ($)')
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3)

    # Animation elements
    line, = ax2.plot([], [], 'g-', linewidth=2)
    fill = None

    cost_text = ax2.text(12, max(cumulative_cost) * 0.95, '', fontsize=12, ha='center',
                         weight='bold', color='green')

    def animate(frame):
        nonlocal fill
        end_idx = int(frame * len(hours) / 100)

        if end_idx > 0:
            # Hourly costs (bin the data)
            hour_indices = (hours[:end_idx] * 60).astype(int) // 60
            hour_bins = np.arange(0, 25)
            hourly_costs = np.zeros(24)
            for i, h in enumerate(hour_indices):
                if h < 24:
                    hourly_costs[h] += hourly_cost[i]

            # Clear and redraw bars
            ax1.clear()
            ax1.set_xlim(0, 24)
            ax1.set_ylim(0, max(hourly_cost) * 1.2)
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Cost ($)')
            ax1.set_title('Hourly Cost')
            ax1.grid(True, alpha=0.3)
            ax1.bar(range(24), hourly_costs, color='#FF6B6B', alpha=0.7, edgecolor='black')

            # Update cumulative line
            line.set_data(hours[:end_idx], cumulative_cost[:end_idx])

            # Remove old fill and create new one
            if fill:
                fill.remove()
            fill = ax2.fill_between(hours[:end_idx], cumulative_cost[:end_idx], alpha=0.3, color='green')

            # Update cost text
            if end_idx > 0:
                cost_text.set_text(f'Daily Cost: ${cumulative_cost[end_idx-1]:.2f}')

        return line, cost_text

    anim = animation.FuncAnimation(fig, animate, frames=101, interval=30, blit=False, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cost timeline animation')
    parser.add_argument('--output', default='04-cost-timeline.gif', help='Output GIF path')
    parser.add_argument('--title', default='Daily API Cost Estimation', help='Animation title')
    parser.add_argument('--peak-qps', type=int, default=3500, help='Peak QPS')
    parser.add_argument('--peak-hour', type=int, default=13, help='Hour of peak (0-23)')
    parser.add_argument('--tokens-per-request', type=int, default=4000, help='Tokens per request')
    parser.add_argument('--cost-per-1k', type=float, default=0.001, help='Cost per 1K tokens')

    args = parser.parse_args()

    create_cost_over_time_animation(args.output, args.title, args.peak_qps, args.peak_hour,
                                   args.tokens_per_request, args.cost_per_1k)
