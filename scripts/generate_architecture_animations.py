#!/usr/bin/env python3
"""
Generate animated visualizations for architecture review systems.
Creates matplotlib-based GIFs showing system dynamics, data flows, and performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from pathlib import Path


def create_request_flow_animation(output_path: str, title: str = "Request Flow Through Pipeline"):
    """
    Animate a request flowing through a multi-stage pipeline.
    Shows latency at each stage and total processing time.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pipeline stages with latencies (ms)
    stages = ['Ingestion', 'Intent\nClassifier', 'Vector\nSearch', 'LLM\nGeneration', 'Formatting']
    latencies = [50, 200, 150, 1400, 200]

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
    ax2.set_ylim(0, 2500)
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylabel('Cumulative Latency (ms)')
    ax2.set_title('Latency Breakdown')
    ax2.grid(axis='y', alpha=0.3)

    # Create bars for latency chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    cumulative = np.cumsum([0] + latencies)
    bars = ax2.bar(range(len(stages)), latencies, color=colors, alpha=0.7, edgecolor='black')

    # Add latency labels
    for i, (bar, lat) in enumerate(zip(bars, latencies)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{lat}ms', ha='center', va='center', fontweight='bold', fontsize=10)

    # Animation elements
    request_dot, = ax1.plot([], [], 'ro', markersize=15, label='Request')
    progress_line, = ax2.plot([], [], 'g-', linewidth=2, label='Total Time')
    info_text = ax1.text(0.5, 1.5, '', fontsize=11, ha='center', weight='bold')
    total_text = ax2.text(0.5, 2300, '', fontsize=11, ha='center', weight='bold', color='green')

    ax1.legend(loc='upper right')

    def animate(frame):
        # Animate request moving through pipeline
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
    return output_path


def create_load_over_time_animation(output_path: str, title: str = "Daily Request Load"):
    """
    Animate realistic request load pattern over 24 hours.
    Shows peak hours, concurrent users, and service capacity.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Simulate realistic load pattern
    hours = np.linspace(0, 24, 240)
    # Peak during business hours (9-17), low at night
    base_load = 200 + 2000 * np.exp(-((hours - 13)**2) / 8)  # Peak at 1pm
    noise = np.random.normal(0, 50, len(hours))
    requests_per_minute = np.maximum(base_load + noise, 50)

    # Concurrent users (roughly proportional to request rate)
    concurrent_users = requests_per_minute / 5 + np.random.normal(0, 5, len(hours))
    concurrent_users = np.maximum(concurrent_users, 10)

    # Capacity lines
    capacity_qps = 3500
    capacity_concurrent = 10000

    # Plot 1: Requests over time
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 4000)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Requests/min')
    ax1.set_title(title)
    ax1.axhline(y=capacity_qps/60, color='r', linestyle='--', label='Capacity', linewidth=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Concurrent users
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 12000)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Concurrent Users')
    ax2.set_title('Concurrent Connection Load')
    ax2.axhline(y=capacity_concurrent, color='r', linestyle='--', label='Capacity (10K)', linewidth=2)
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
    return output_path


def create_latency_components_animation(output_path: str, title: str = "Component Latency Breakdown"):
    """
    Animate stacked bar chart showing contribution of each component to total latency.
    Shows best case, average case, and worst case scenarios.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Components with latencies: (best, avg, worst)
    components = ['Intent\nClassify', 'Vector\nSearch', 'Retrieval\nRank', 'LLM\nGeneration', 'Format &\nRespond']
    latency_best = [100, 80, 30, 800, 100]
    latency_avg = [200, 150, 50, 1400, 200]
    latency_worst = [400, 300, 150, 2000, 300]

    scenarios = ['Best Case\n(Simple Query)', 'Average Case\n(Typical Query)', 'Worst Case\n(Complex Query)']
    all_latencies = [latency_best, latency_avg, latency_worst]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    x = np.arange(len(scenarios))
    width = 0.6

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 3500)
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
        for component_idx, (component, colors_list) in enumerate(zip(components, colors)):
            heights = []
            for scenario_idx, latencies in enumerate(all_latencies):
                height = latencies[component_idx] * progress
                heights.append(height)

            bars = ax.bar(x, heights, width, label=component if component_idx < len(components) else '',
                         bottom=bottom, color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
            bars_by_component[component_idx] = list(bars)

            # Add labels
            for i, (bar, height) in enumerate(zip(bars, heights)):
                if height > 50:  # Only show label if bar is large enough
                    ax.text(bar.get_x() + bar.get_width()/2, bottom[i] + height/2,
                           f'{int(height)}ms', ha='center', va='center',
                           fontsize=8, weight='bold', color='white')

            bottom += heights

        # Add legend
        if frame == 100:
            ax.legend(loc='upper left', ncol=len(components), fontsize=9)

        return [bar for bars in bars_by_component for bar in bars]

    anim = animation.FuncAnimation(fig, animate, frames=101, interval=30, blit=True, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    return output_path


def create_cost_over_time_animation(output_path: str, title: str = "Daily API Cost Estimation"):
    """
    Animate cost accumulation over 24 hours based on token usage.
    Shows per-request cost, cumulative cost, and cost by component.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Simulate token usage based on request load
    hours = np.linspace(0, 24, 240)
    requests = 200 + 2000 * np.exp(-((hours - 13)**2) / 8) + np.random.normal(0, 50, len(hours))
    requests = np.maximum(requests, 50)

    # Token usage per request: ~4000 tokens avg (input + output)
    tokens_per_request = 4000
    total_tokens = requests * tokens_per_request

    # Cost: $0.001 per 1K tokens (gpt-4-turbo estimate)
    cost_per_1k_tokens = 0.001
    hourly_cost = total_tokens * cost_per_1k_tokens / 60  # Convert to per-minute cost
    cumulative_cost = np.cumsum(hourly_cost) / 60  # Cumulative hourly cost

    # Left plot: Hourly costs
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 150)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Hourly Cost')
    ax1.grid(True, alpha=0.3)

    # Right plot: Cumulative cost
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 250)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Cumulative Cost ($)')
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3)

    # Animation elements
    bars = ax1.bar([], [], color='#FF6B6B', alpha=0.7, edgecolor='black')
    line, = ax2.plot([], [], 'g-', linewidth=2)
    fill = None

    cost_text = ax2.text(12, 230, '', fontsize=12, ha='center', weight='bold', color='green')

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
            ax1.set_ylim(0, 150)
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
    return output_path


def create_sentiment_escalation_animation(output_path: str, title: str = "Sentiment vs Escalation Rate"):
    """
    Animate scatter plot showing relationship between user sentiment and escalation rate.
    Shows real-time data points accumulating over time.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    np.random.seed(42)
    # Generate realistic sentiment-escalation correlation
    n_samples = 500
    sentiment_scores = np.random.normal(0.3, 0.35, n_samples)
    sentiment_scores = np.clip(sentiment_scores, -1, 1)

    # Escalation probability inversely correlated with sentiment
    escalation_base = 0.8 - (sentiment_scores + 1) / 2 * 0.7
    escalation_rates = np.clip(escalation_base + np.random.normal(0, 0.1, n_samples), 0, 1)

    colors_array = escalation_rates

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('User Sentiment Score', fontsize=12)
    ax.set_ylabel('Escalation Rate', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    # Add decision boundary line
    x_line = np.linspace(-1, 1, 100)
    y_line = 0.8 - (x_line + 1) / 2 * 0.7
    ax.plot(x_line, y_line, 'r--', linewidth=2, label='Decision Boundary')

    scatter = ax.scatter([], [], c=[], cmap='RdYlGn_r', s=60, alpha=0.6,
                        vmin=0, vmax=1, edgecolors='black', linewidth=0.5)

    # Add regions
    ax.axvspan(-1.2, -0.5, alpha=0.1, color='red', label='High Escalation Risk')
    ax.axvspan(-0.5, 1.2, alpha=0.1, color='green', label='Low Escalation Risk')

    stats_text = ax.text(-1.0, 0.95, '', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.legend(loc='upper left')

    def animate(frame):
        end_idx = int(frame * len(sentiment_scores) / 100)

        if end_idx > 0:
            scatter.set_offsets(np.c_[sentiment_scores[:end_idx], escalation_rates[:end_idx]])
            scatter.set_array(colors_array[:end_idx])

            # Calculate statistics
            avg_sentiment = np.mean(sentiment_scores[:end_idx])
            avg_escalation = np.mean(escalation_rates[:end_idx])
            stats_text.set_text(
                f'Samples: {end_idx}\n'
                f'Avg Sentiment: {avg_sentiment:.2f}\n'
                f'Avg Escalation: {avg_escalation:.1%}'
            )

        return scatter, stats_text

    anim = animation.FuncAnimation(fig, animate, frames=101, interval=30, blit=True, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    return output_path


def generate_all_animations(output_dir: str = 'arch-review/animations'):
    """Generate all animation types for architecture documentation."""
    os.makedirs(output_dir, exist_ok=True)

    animations = [
        ('01-request-flow.gif', create_request_flow_animation, 'Request Flow Through Pipeline'),
        ('02-daily-load.gif', create_load_over_time_animation, 'Daily Request Load Pattern'),
        ('03-latency-breakdown.gif', create_latency_components_animation, 'Latency Component Breakdown'),
        ('04-cost-timeline.gif', create_cost_over_time_animation, 'Daily Cost Accumulation'),
        ('05-sentiment-escalation.gif', create_sentiment_escalation_animation, 'Sentiment vs Escalation Correlation'),
    ]

    print(f"Generating animations in {output_dir}/\n")
    created = []

    for filename, func, title in animations:
        output_path = os.path.join(output_dir, filename)
        try:
            func(output_path, title)
            size_kb = os.path.getsize(output_path) / 1024
            print(f"✅ {filename} ({size_kb:.1f}KB)")
            created.append(filename)
        except Exception as e:
            print(f"❌ {filename}: {e}")

    print(f"\n✅ Created {len(created)}/5 animations")
    return created


if __name__ == '__main__':
    generate_all_animations()
