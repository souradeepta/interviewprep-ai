#!/usr/bin/env python3
"""
Request Flow Animation
Shows a request flowing through a multi-stage pipeline with latency breakdown.

Usage:
    python3 01-request-flow-animation.py --output output.gif --title "My Pipeline"

Customization:
    Modify 'stages' and 'latencies' lists below to match your system architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse


def create_request_flow_animation(
    output_path: str,
    title: str = "Request Flow Through Pipeline",
    stages: list = None,
    latencies: list = None,
    figsize: tuple = (14, 5)
):
    """
    Animate a request flowing through pipeline stages.

    Args:
        output_path: Output GIF file path
        title: Animation title
        stages: List of stage names (e.g., ['Ingestion', 'Classification', ...])
        latencies: List of latencies in ms for each stage
        figsize: Figure size (width, height)
    """

    # Default stages and latencies (customer service platform example)
    if stages is None:
        stages = ['Ingestion', 'Intent\nClassifier', 'Vector\nSearch', 'LLM\nGeneration', 'Formatting']
    if latencies is None:
        latencies = [50, 200, 150, 1400, 200]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Request animation
    ax1.set_xlim(-0.5, len(stages) - 0.5)
    ax1.set_ylim(-0.5, 2)
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, fontsize=9)
    ax1.set_ylabel('Processing Stage')
    ax1.set_title(title)
    ax1.set_yticks([])
    ax1.grid(axis='x', alpha=0.3)

    # Right plot: Cumulative latency
    ax2.set_xlim(-0.5, len(stages) - 0.5)
    ax2.set_ylim(0, max(latencies) * 1.3)
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylabel('Cumulative Latency (ms)')
    ax2.set_title('Latency Breakdown')
    ax2.grid(axis='y', alpha=0.3)

    # Create bars for latency chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    # Extend colors if more stages than default
    while len(colors) < len(stages):
        colors.append('#' + '%06x' % np.random.randint(0, 0xFFFFFF))

    cumulative = np.cumsum([0] + latencies)
    bars = ax2.bar(range(len(stages)), latencies, color=colors[:len(stages)], alpha=0.7, edgecolor='black')

    # Add latency labels
    for i, (bar, lat) in enumerate(zip(bars, latencies)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{lat}ms', ha='center', va='center', fontweight='bold', fontsize=10)

    # Animation elements
    request_dot, = ax1.plot([], [], 'ro', markersize=15, label='Request')
    progress_line, = ax2.plot([], [], 'g-', linewidth=2, label='Total Time')
    info_text = ax1.text(0.5, 1.5, '', fontsize=11, ha='center', weight='bold')
    total_text = ax2.text(0.5, max(cumulative)*0.95, '', fontsize=11, ha='center',
                          weight='bold', color='green')

    ax1.legend(loc='upper right')

    def animate(frame):
        stage_pos = frame / 10  # 0 to len(stages) over 110 frames

        if stage_pos < len(stages):
            request_dot.set_data([stage_pos], [1])
            current_stage = int(stage_pos)
            info_text.set_text(f'Stage: {stages[current_stage]}\nLatency: {latencies[current_stage]}ms')

            # Update progress line
            up_to = min(current_stage + 1, len(stages))
            progress_line.set_data(range(up_to), cumulative[:up_to])
            total_text.set_text(f'Total: {cumulative[up_to]:.0f}ms')
        else:
            # Final state
            request_dot.set_data([len(stages)-0.5], [1])
            info_text.set_text(f'Complete!\nTotal Time: {sum(latencies)}ms')
            progress_line.set_data(range(len(stages)), cumulative[1:])
            total_text.set_text(f'Total: {sum(latencies)}ms')

        return request_dot, progress_line, info_text, total_text

    anim = animation.FuncAnimation(fig, animate, frames=120, interval=50, blit=True, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate request flow animation')
    parser.add_argument('--output', default='01-request-flow.gif', help='Output GIF path')
    parser.add_argument('--title', default='Request Flow Through Pipeline', help='Animation title')
    parser.add_argument('--stages', nargs='+',
                       default=['Ingestion', 'Intent\nClassifier', 'Vector\nSearch', 'LLM\nGeneration', 'Formatting'],
                       help='Stage names')
    parser.add_argument('--latencies', nargs='+', type=int, default=[50, 200, 150, 1400, 200],
                       help='Latencies in ms')

    args = parser.parse_args()

    create_request_flow_animation(args.output, args.title, args.stages, args.latencies)
