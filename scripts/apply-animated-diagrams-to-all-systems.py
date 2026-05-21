#!/usr/bin/env python3
"""
Apply animated architecture diagrams to all 30 systems.
Inserts animated diagram references after the "Dynamic System Visualization" section.
"""

import os
import glob
from pathlib import Path

# Animated diagram content to insert
ANIMATED_DIAGRAMS_SECTION = """
## Animated Architecture Visualization

See the system in action with dynamic visualizations:

### System Deployment Animation
![System Deployment](../animated-diagrams/01-system-architecture-deployment.gif)

Infrastructure components appearing and connecting in real-time, showing load balancers, API gateways, microservices, and data layer setup.

### Request Flow Animation
![Request Flow](../animated-diagrams/02-request-flow-pipeline.gif)

A single request flowing through the complete pipeline with latency accumulation at each stage, demonstrating the critical path and timing constraints.

### Data Flow Animation
![Data Flow](../animated-diagrams/03-data-flow-movement.gif)

Concurrent data packets flowing through processors and ML models to storage systems, showing simultaneous traffic and I/O patterns.

### Auto-Scaling Animation
![Auto-Scaling](../animated-diagrams/04-auto-scaling-load.gif)

Dynamic scaling response to traffic load, showing pod count adjusting up and down with capacity headroom management over time.

"""

def apply_animated_diagrams():
    """Apply animated diagram references to all system files."""
    system_dir = Path("arch-review/systems")
    system_files = sorted(glob.glob(str(system_dir / "*.md")))

    if not system_files:
        print("❌ No system files found in arch-review/systems/")
        return

    applied_count = 0
    skipped_count = 0

    for system_file in system_files:
        try:
            with open(system_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if animated diagrams section already exists
            if "## Animated Architecture Visualization" in content:
                print(f"⏭️  Skipped (already has animated diagrams): {Path(system_file).name}")
                skipped_count += 1
                continue

            # Find insertion point: before "## Related Systems", "## Interview", or end of file
            # These are the final sections that should come after architecture diagrams
            final_section_markers = [
                "\n## Related Systems",
                "\n## Interview Quick-Reference",
                "\n## Interview Q&A",
                "\n---"  # Separator before footer
            ]

            insertion_pos = -1
            for marker in final_section_markers:
                pos = content.find(marker)
                if pos != -1:
                    insertion_pos = pos
                    break

            if insertion_pos == -1:
                # No marker found, insert at end of file
                insertion_pos = len(content)

            # Insert the animated diagrams section
            new_content = content[:insertion_pos] + "\n" + ANIMATED_DIAGRAMS_SECTION + content[insertion_pos:]

            with open(system_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Applied: {Path(system_file).name}")
            applied_count += 1

        except Exception as e:
            print(f"❌ Error processing {Path(system_file).name}: {e}")

    print(f"\n{'='*60}")
    print(f"Applied animated diagrams to {applied_count} systems")
    print(f"Skipped {skipped_count} systems (already had diagrams)")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Applying animated architecture diagrams to all systems...\n")
    apply_animated_diagrams()
