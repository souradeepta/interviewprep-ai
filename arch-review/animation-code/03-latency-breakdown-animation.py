#!/usr/bin/env python3
"""
Latency Component Breakdown Animation
Shows best/average/worst case latency breakdown across system components.

Usage:
    python3 03-latency-breakdown-animation.py --output output.gif --title "My System"
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse


def create_latency_components_animation(
    output_path: str,
    title: str = "Component Latency Breakdown",
    components: list = None,
    latency_best: list = None,
    latency_avg: list = None,
    latency_worst: list = None,
    figsize: tuple = (12, 6)
):
    """
    Animate stacked bar chart showing latency contribution by component.

    Args:
        output_path: Output GIF file path
        title: Animation title
        components: List of component names
        latency_best: Best case latencies (ms)
        latency_avg: Average case latencies (ms)
        latency_worst: Worst case latencies (ms)
        figsize: Figure size
    """

    # Default values (customer service platform)
    if components is None:
        components = ['Intent\nClassify', 'Vector\nSearch', 'Retrieval\nRank', 'LLM\nGeneration', 'Format &\nRespond']
    if latency_best is None:
        latency_best = [100, 80, 30, 800, 100]
    if latency_avg is None:
        latency_avg = [200, 150, 50, 1400, 200]
    if latency_worst is None:
        latency_worst = [400, 300, 150, 2000, 300]

    scenarios = ['Best Case\n(Simple Query)', 'Average Case\n(Typical Query)', 'Worst Case\n(Complex Query)']
    all_latencies = [latency_best, latency_avg, latency_worst]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    # Extend colors if needed
    while len(colors) < len(components):
        colors.append('#' + '%06x' % np.random.randint(0, 0xFFFFFF))

    x = np.arange(len(scenarios))
    width = 0.6

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, sum(latency_worst) + 200)
    ax.grid(axis='y', alpha=0.3)

    # Storage for animated bars
    bars_by_component = [[] for _ in range(len(components))]

    def animate(frame):
        # Clear previous bars
        for bar_list in bars_by_component:
            for bar in bar_list:
                bar.remove()
        bars_by_component[:] = [[] for _ in range(len(components))]

        # Animate to a certain completion level
        progress = min(1.0, frame / 100)

        bottom = np.zeros(len(scenarios))
        for component_idx, component in enumerate(components):
            heights = []
            for scenario_idx, latencies in enumerate(all_latencies):
                height = latencies[component_idx] * progress
                heights.append(height)

            bars = ax.bar(x, heights, width, label=component if component_idx < len(components) else '',
                         bottom=bottom, color=colors[component_idx], alpha=0.8, edgecolor='black', linewidth=0.5)
            bars_by_component[component_idx] = list(bars)

            # Add labels
            for i, (bar, height) in enumerate(zip(bars, heights)):
                if height > 50:
                    ax.text(bar.get_x() + bar.get_width()/2, bottom[i] + height/2,
                           f'{int(height)}ms', ha='center', va='center',
                           fontsize=8, weight='bold', color='white')

            bottom += heights

        # Add legend on final frame
        if frame == 100:
            ax.legend(loc='upper left', ncol=len(components), fontsize=9)

        return [bar for bars in bars_by_component for bar in bars]

    anim = animation.FuncAnimation(fig, animate, frames=101, interval=30, blit=True, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate latency breakdown animation')
    parser.add_argument('--output', default='03-latency-breakdown.gif', help='Output GIF path')
    parser.add_argument('--title', default='Component Latency Breakdown', help='Animation title')
    parser.add_argument('--components', nargs='+',
                       default=['Intent\nClassify', 'Vector\nSearch', 'Retrieval\nRank', 'LLM\nGeneration', 'Format &\nRespond'],
                       help='Component names')
    parser.add_argument('--best', nargs='+', type=int, default=[100, 80, 30, 800, 100],
                       help='Best case latencies')
    parser.add_argument('--avg', nargs='+', type=int, default=[200, 150, 50, 1400, 200],
                       help='Average case latencies')
    parser.add_argument('--worst', nargs='+', type=int, default=[400, 300, 150, 2000, 300],
                       help='Worst case latencies')

    args = parser.parse_args()

    create_latency_components_animation(args.output, args.title, args.components, args.best, args.avg, args.worst)
