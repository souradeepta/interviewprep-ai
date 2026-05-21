#!/usr/bin/env python3
"""
Generate animated architecture diagrams showing:
1. Component deployment/initialization
2. Request flow through the pipeline
3. Data flow and interactions
4. Load distribution and scaling

Creates GIF animations for system, application, and process flow architectures.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
from pathlib import Path


def create_system_architecture_animation(output_path: str, title: str = "System Architecture"):
    """Animate system components appearing and connecting."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Component positions
    components = [
        {'pos': (1, 8), 'size': (1.5, 0.8), 'label': 'Load\nBalancer', 'color': '#FF6B6B'},
        {'pos': (4, 8), 'size': (1.5, 0.8), 'label': 'API\nGateway', 'color': '#4ECDC4'},
        {'pos': (7, 8), 'size': (1.5, 0.8), 'label': 'Service\nMesh', 'color': '#45B7D1'},

        {'pos': (1, 5), 'size': (1.2, 0.7), 'label': 'Service 1', 'color': '#FFA07A'},
        {'pos': (3.5, 5), 'size': (1.2, 0.7), 'label': 'Service 2', 'color': '#FFA07A'},
        {'pos': (6, 5), 'size': (1.2, 0.7), 'label': 'Service 3', 'color': '#FFA07A'},
        {'pos': (8.5, 5), 'size': (1.2, 0.7), 'label': 'Service 4', 'color': '#FFA07A'},

        {'pos': (1, 2), 'size': (1.5, 0.7), 'label': 'Database', 'color': '#98D8C8'},
        {'pos': (4, 2), 'size': (1.5, 0.7), 'label': 'Cache', 'color': '#98D8C8'},
        {'pos': (7, 2), 'size': (1.5, 0.7), 'label': 'Search\nIndex', 'color': '#98D8C8'},
    ]

    boxes = []
    texts = []

    def animate(frame):
        # Animate components appearing
        num_components = min(frame // 3 + 1, len(components))

        # Clear and redraw
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Draw boxes
        for i in range(num_components):
            comp = components[i]
            x, y = comp['pos']
            w, h = comp['size']

            # Animated appearance
            scale = min(1.0, (frame - i * 3) / 8)
            center_x, center_y = x + w/2, y + h/2
            scaled_w = w * scale
            scaled_h = h * scale
            box_x = center_x - scaled_w/2
            box_y = center_y - scaled_h/2

            box = FancyBboxPatch(
                (box_x, box_y), scaled_w, scaled_h,
                boxstyle="round,pad=0.1",
                edgecolor='black',
                facecolor=comp['color'],
                alpha=0.7,
                linewidth=2
            )
            ax.add_patch(box)

            # Add label
            ax.text(
                x + w/2, y + h/2, comp['label'],
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white'
            )

        # Draw connections between layers
        if num_components > 3:  # After top layer appears
            for i in range(3):
                x1, y1 = components[i]['pos']
                w1, h1 = components[i]['size']
                for j in range(i*2, min(i*2+3, 4)):
                    if j < num_components - 3:
                        x2, y2 = components[j + 3]['pos']
                        arrow = FancyArrowPatch(
                            (x1 + w1/2, y1 - 0.1),
                            (x2 + components[j+3]['size'][0]/2, y2 + components[j+3]['size'][1] + 0.1),
                            arrowstyle='->', mutation_scale=20,
                            color='gray', alpha=0.5, linewidth=1
                        )
                        ax.add_patch(arrow)

        # Status text
        status = f"Components: {num_components}/{len(components)}"
        ax.text(5, 0.3, status, ha='center', fontsize=10, style='italic')

        return ax,

    anim = animation.FuncAnimation(fig, animate, frames=120, interval=30, blit=False, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Created: {output_path}")


def create_request_flow_animation(output_path: str, title: str = "Request Flow Animation"):
    """Animate a request flowing through the application architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Pipeline stages
    stages = [
        {'x': 1, 'label': 'Ingress', 'color': '#FF6B6B'},
        {'x': 3, 'label': 'Auth', 'color': '#4ECDC4'},
        {'x': 5, 'label': 'Process', 'color': '#45B7D1'},
        {'x': 7, 'label': 'Cache', 'color': '#FFA07A'},
        {'x': 9, 'label': 'Response', 'color': '#98D8C8'},
    ]

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Draw pipeline stages
        for stage in stages:
            box = FancyBboxPatch(
                (stage['x'] - 0.4, 4.5), 0.8, 2,
                boxstyle="round,pad=0.1",
                edgecolor='black',
                facecolor=stage['color'],
                alpha=0.7,
                linewidth=2
            )
            ax.add_patch(box)
            ax.text(stage['x'], 3.8, stage['label'], ha='center', fontsize=9, fontweight='bold')

        # Draw arrows between stages
        for i in range(len(stages) - 1):
            arrow = FancyArrowPatch(
                (stages[i]['x'] + 0.4, 5.5),
                (stages[i+1]['x'] - 0.4, 5.5),
                arrowstyle='->', mutation_scale=20,
                color='gray', alpha=0.5, linewidth=2
            )
            ax.add_patch(arrow)

        # Animate request ball moving through pipeline
        request_progress = (frame % 120) / 120
        request_idx = int(request_progress * (len(stages) - 1))
        next_idx = min(request_idx + 1, len(stages) - 1)

        progress_in_stage = (request_progress * (len(stages) - 1)) - request_idx

        current_stage = stages[request_idx]
        next_stage = stages[next_idx]

        request_x = current_stage['x'] + (next_stage['x'] - current_stage['x']) * progress_in_stage
        request_y = 5.5

        # Draw request
        circle = plt.Circle((request_x, request_y), 0.2, color='red', alpha=0.8, zorder=10)
        ax.add_patch(circle)

        # Draw timeline
        elapsed = request_idx * 20  # 20ms per stage
        ax.text(5, 1, f"Elapsed: {elapsed}ms | Stage: {request_idx}/{len(stages)-1}",
                ha='center', fontsize=11, fontweight='bold')

        return ax,

    anim = animation.FuncAnimation(fig, animate, frames=240, interval=20, blit=False, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Created: {output_path}")


def create_data_flow_animation(output_path: str, title: str = "Data Flow Animation"):
    """Animate data flowing through system components."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Components in layers
    inputs = [
        {'pos': (1, 8), 'label': 'Client 1'},
        {'pos': (3, 8), 'label': 'Client 2'},
        {'pos': (5, 8), 'label': 'Client 3'},
    ]

    processing = [
        {'pos': (2, 5.5), 'label': 'Processor'},
        {'pos': (6, 5.5), 'label': 'ML Model'},
    ]

    outputs = [
        {'pos': (2, 3), 'label': 'Cache'},
        {'pos': (4, 3), 'label': 'DB'},
        {'pos': (6, 3), 'label': 'Queue'},
    ]

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Draw components
        for inp in inputs:
            box = FancyBboxPatch((inp['pos'][0]-0.4, inp['pos'][1]-0.3), 0.8, 0.6,
                                boxstyle="round,pad=0.05", edgecolor='black',
                                facecolor='#FF6B6B', alpha=0.7, linewidth=2)
            ax.add_patch(box)
            ax.text(inp['pos'][0], inp['pos'][1], inp['label'], ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')

        for proc in processing:
            box = FancyBboxPatch((proc['pos'][0]-0.5, proc['pos'][1]-0.35), 1, 0.7,
                                boxstyle="round,pad=0.05", edgecolor='black',
                                facecolor='#4ECDC4', alpha=0.7, linewidth=2)
            ax.add_patch(box)
            ax.text(proc['pos'][0], proc['pos'][1], proc['label'], ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')

        for out in outputs:
            box = FancyBboxPatch((out['pos'][0]-0.4, out['pos'][1]-0.3), 0.8, 0.6,
                                boxstyle="round,pad=0.05", edgecolor='black',
                                facecolor='#98D8C8', alpha=0.7, linewidth=2)
            ax.add_patch(box)
            ax.text(out['pos'][0], out['pos'][1], out['label'], ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')

        # Animate data packets flowing
        packet_progress = (frame % 100) / 100

        # From inputs to processing
        for i, inp in enumerate(inputs):
            proc = processing[i % len(processing)]
            packet_x = inp['pos'][0] + (proc['pos'][0] - inp['pos'][0]) * packet_progress
            packet_y = inp['pos'][1] + (proc['pos'][1] - inp['pos'][1]) * packet_progress

            color = plt.cm.Spectral(i / len(inputs))
            circle = plt.Circle((packet_x, packet_y), 0.15, color=color, alpha=0.8, zorder=10)
            ax.add_patch(circle)

        # From processing to outputs
        for i, proc in enumerate(processing):
            out = outputs[i % len(outputs)]
            packet_x = proc['pos'][0] + (out['pos'][0] - proc['pos'][0]) * (packet_progress + 0.3) % 1
            packet_y = proc['pos'][1] + (out['pos'][1] - proc['pos'][1]) * (packet_progress + 0.3) % 1

            color = plt.cm.viridis(i / len(processing))
            circle = plt.Circle((packet_x, packet_y), 0.12, color=color, alpha=0.7, zorder=10)
            ax.add_patch(circle)

        # Status
        packets = int(packet_progress * 100)
        ax.text(5, 0.5, f"Data Packets in Flight: {packets}", ha='center', fontsize=10, fontweight='bold')

        return ax,

    anim = animation.FuncAnimation(fig, animate, frames=120, interval=30, blit=False, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Created: {output_path}")


def create_scaling_animation(output_path: str, title: str = "Auto-Scaling Animation"):
    """Animate system scaling up and down based on load."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Pod count over time
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 20)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Number of Pods')
    ax1.set_title('Pod Auto-Scaling')
    ax1.grid(True, alpha=0.3)

    # Right plot: Load distribution
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title('Load Distribution Across Pods')
    ax2.axis('off')

    def animate(frame):
        # Simulate load pattern
        time = frame
        load = 5 + 8 * np.sin(time / 40) + np.random.normal(0, 0.5)
        load = max(1, min(18, load))
        pods = max(3, int(load + 2))  # With headroom

        # Left plot: Load and pod count
        ax1.clear()
        ax1.set_xlim(0, 120)
        ax1.set_ylim(0, 20)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Count')
        ax1.set_title('Pod Auto-Scaling')
        ax1.grid(True, alpha=0.3)

        # Generate history
        times = np.linspace(0, time, max(2, int(time)))
        loads = 5 + 8 * np.sin(times / 40)
        pod_counts = np.maximum(3, np.ceil(loads + 2))

        ax1.plot(times, loads, 'b-', label='Load (requests/sec)', linewidth=2)
        ax1.step(times, pod_counts, 'r-', label='Pods Running', linewidth=2, where='post')
        ax1.legend(loc='upper left')
        ax1.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Max Capacity')

        # Right plot: Pod distribution
        ax2.clear()
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_title(f'Pod Distribution ({pods} Pods, Load: {load:.1f})')
        ax2.axis('off')

        # Draw pods in a circle
        for i in range(pods):
            angle = 2 * np.pi * i / max(pods, 3)
            x = 0.8 * np.cos(angle)
            y = 0.8 * np.sin(angle)

            # Pod load (varies)
            pod_load = min(1.0, (load / pods) / 10)
            color = plt.cm.RdYlGn_r(pod_load)

            circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, edgecolor='black', linewidth=2)
            ax2.add_patch(circle)
            ax2.text(x, y, f'P{i+1}', ha='center', va='center', fontweight='bold', fontsize=9)

            # Load bar
            ax2.barh(y - 0.5, pod_load * 0.8, height=0.1, left=x-0.4, color=color, alpha=0.6)

        return ax1, ax2

    anim = animation.FuncAnimation(fig, animate, frames=120, interval=40, blit=False, repeat=True)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
    print(f"✅ Created: {output_path}")


def generate_all_animated_diagrams():
    """Generate all animated architecture diagrams."""
    output_dir = Path('arch-review/animated-diagrams')
    output_dir.mkdir(parents=True, exist_ok=True)

    animations = [
        ('01-system-architecture-deployment.gif', create_system_architecture_animation,
         'System Architecture: Component Deployment'),
        ('02-request-flow-pipeline.gif', create_request_flow_animation,
         'Application Architecture: Request Flow Through Pipeline'),
        ('03-data-flow-movement.gif', create_data_flow_animation,
         'Data Flow: Packets Moving Through System'),
        ('04-auto-scaling-load.gif', create_scaling_animation,
         'Infrastructure: Auto-Scaling Response to Load'),
    ]

    print("Generating animated architecture diagrams...\n")

    for filename, func, title in animations:
        output_path = str(output_dir / filename)
        func(output_path, title)

    print(f"\n✅ Created {len(animations)} animated diagrams in {output_dir}/")

    # Create index
    index = """# Animated Architecture Diagrams

Dynamic visualizations showing how systems work in real-time.

## Diagrams

### 1. System Architecture: Component Deployment
**File**: `01-system-architecture-deployment.gif`

Shows infrastructure components appearing and connecting:
- Load balancers
- API gateways and service mesh
- Microservices (4 services)
- Data layer (database, cache, search index)
- Connection establishment between layers

**Use case**: Understanding system scale, infrastructure planning

### 2. Request Flow Pipeline
**File**: `02-request-flow-pipeline.gif`

Animates a single request flowing through 5 stages:
- Ingress → Authentication → Processing → Cache → Response
- Shows latency accumulation at each stage
- Total elapsed time visualization

**Use case**: Understanding request lifecycle, identifying bottlenecks

### 3. Data Flow Movement
**File**: `03-data-flow-movement.gif`

Shows data packets flowing through the system:
- Multiple clients sending data simultaneously
- Data through processors and ML models
- Distribution to storage (cache, DB, queue)
- Concurrent traffic visualization

**Use case**: Understanding data movement, identifying I/O bottlenecks

### 4. Auto-Scaling Response to Load
**File**: `04-auto-scaling-load.gif`

Demonstrates dynamic scaling:
- Load pattern oscillating over time
- Pod count scaling up/down in response
- Load distribution across pods
- Capacity headroom management

**Use case**: Understanding scaling behavior, capacity planning

## Usage

View animations directly in GitHub markdown:
```markdown
![System Architecture](animated-diagrams/01-system-architecture-deployment.gif)
```

## Benefits

✅ **Visual Learning**: See how components interact dynamically
✅ **Interview Prep**: Explain system behavior with visual aid
✅ **Design Validation**: Verify scaling and flow assumptions
✅ **Documentation**: More engaging than static diagrams
✅ **Troubleshooting**: Identify where problems likely occur

## Integration with Architecture Files

Add to each system's markdown:

```markdown
## Dynamic Architecture Visualization

![System Deployment](../animated-diagrams/01-system-architecture-deployment.gif)

Our system deploys with X components across Y nodes, automatically scaling
from Z to N pods based on demand.
```

All animations loop continuously (suitable for documentation and presentations).
"""

    (output_dir / 'README.md').write_text(index)
    print(f"✅ Created: {output_dir / 'README.md'}")


if __name__ == '__main__':
    generate_all_animated_diagrams()
