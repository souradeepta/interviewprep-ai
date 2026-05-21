#!/usr/bin/env python3
"""
Sentiment vs Escalation Animation
Shows scatter plot relationship between user sentiment and escalation rate.

Usage:
    python3 05-sentiment-escalation-animation.py --output output.gif --num-samples 500
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse


def create_sentiment_escalation_animation(
    output_path: str,
    title: str = "Sentiment vs Escalation Rate",
    num_samples: int = 500,
    sentiment_mean: float = 0.3,
    sentiment_std: float = 0.35,
    figsize: tuple = (12, 7)
):
    """
    Animate scatter plot showing sentiment-escalation correlation.

    Args:
        output_path: Output GIF file path
        title: Animation title
        num_samples: Number of data points to generate
        sentiment_mean: Mean of sentiment distribution
        sentiment_std: Std dev of sentiment distribution
        figsize: Figure size
    """
    np.random.seed(42)

    # Generate realistic sentiment-escalation correlation
    sentiment_scores = np.random.normal(sentiment_mean, sentiment_std, num_samples)
    sentiment_scores = np.clip(sentiment_scores, -1, 1)

    # Escalation probability inversely correlated with sentiment
    escalation_base = 0.8 - (sentiment_scores + 1) / 2 * 0.7
    escalation_rates = np.clip(escalation_base + np.random.normal(0, 0.1, num_samples), 0, 1)

    colors_array = escalation_rates

    fig, ax = plt.subplots(figsize=figsize)

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

    stats_text = ax.text(-1.0, 0.95, '', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.legend(loc='upper left')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Escalation Rate', rotation=270, labelpad=15)

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
    print(f"✅ Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sentiment-escalation animation')
    parser.add_argument('--output', default='05-sentiment-escalation.gif', help='Output GIF path')
    parser.add_argument('--title', default='Sentiment vs Escalation Rate', help='Animation title')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of data points')
    parser.add_argument('--sentiment-mean', type=float, default=0.3, help='Sentiment mean')
    parser.add_argument('--sentiment-std', type=float, default=0.35, help='Sentiment std dev')

    args = parser.parse_args()

    create_sentiment_escalation_animation(args.output, args.title, args.num_samples,
                                         args.sentiment_mean, args.sentiment_std)
