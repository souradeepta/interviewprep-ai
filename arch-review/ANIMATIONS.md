# Architecture Animation Library

Matplotlib-based animated GIFs showing system dynamics, performance characteristics, and operational metrics for AI/ML architecture reviews.

## Overview

This library provides 5 reusable animation templates that can be customized per-system:

| Animation | Purpose | Use Case |
|-----------|---------|----------|
| **Request Flow** | Shows request path through pipeline stages | Demonstrate latency budget allocation |
| **Daily Load** | 24-hour traffic pattern with capacity | Show peak hours, scaling needs |
| **Latency Breakdown** | Best/avg/worst case component breakdown | Identify bottlenecks, optimization targets |
| **Cost Timeline** | Cumulative cost and hourly spend | Visualize daily operational cost |
| **Sentiment vs Escalation** | Correlation between metrics | Show decision boundaries, risk regions |

---

## Generated Animations

All animations are in `arch-review/animations/`:

### 1. Request Flow Through Pipeline (`01-request-flow.gif`)
**Size:** 173.8 KB | **Duration:** ~5 seconds (loop)

Shows a single request flowing through stages, with:
- Real-time latency at each stage
- Cumulative latency accumulation
- Total processing time

**Best for:** 
- Explaining architecture with sequential components
- Latency budget discussions
- Component contribution to SLA

**Example systems:**
- Customer service platform (Ingestion → Intent → RAG → LLM → Format)
- Code review agent (Parse → Analyze → Critique → Format)
- Translation service (Detect → Translate → Quality Check → Output)

---

### 2. Daily Load Pattern (`02-daily-load.gif`)
**Size:** 1.5 MB | **Duration:** ~5 seconds (loop)

Shows realistic 24-hour traffic with:
- Requests per minute over time
- Concurrent user connections
- Capacity headroom visualization
- Peak hour identification

**Best for:**
- Capacity planning discussions
- Scaling requirements explanation
- Resource allocation justification

**Example systems:**
- All systems: Peak hours are 9am-5pm business hours (3500 QPS peak typical)

---

### 3. Latency Component Breakdown (`03-latency-breakdown.gif`)
**Size:** 1.3 MB | **Duration:** ~5 seconds (loop)

Stacked bar chart showing:
- Best case (simple queries): 1.1 seconds
- Average case (typical): 2.0 seconds  
- Worst case (complex): 3.1 seconds

**Best for:**
- SLA discussions
- Component optimization priorities
- Performance regression detection

**Example systems:**
- Customer service: [Classify 200ms, Search 150ms, Rank 50ms, Generate 1400ms, Format 200ms]
- RAG systems: Similar breakdown

---

### 4. Daily Cost Accumulation (`04-cost-timeline.gif`)
**Size:** 563.9 KB | **Duration:** ~5 seconds (loop)

Shows:
- Hourly cost (bar chart)
- Cumulative daily cost (line chart)
- Final day total (~$200 for typical platform)

**Best for:**
- Business case discussions
- Cost optimization opportunities
- Budget forecasting

**Example formula:**
```
Daily cost = Total tokens × Cost per 1K tokens
50K requests × 4K tokens × $0.001/1K = ~$200/day
```

---

### 5. Sentiment vs Escalation Scatter (`05-sentiment-escalation.gif`)
**Size:** 2.2 MB | **Duration:** ~5 seconds (loop)

Shows:
- 500 data points: (sentiment score, escalation rate)
- Decision boundary line
- Risk regions (red: high escalation, green: low)

**Best for:**
- Classification threshold discussions
- Risk management visualization
- Model decision boundary explanation

---

## How to Customize Animations

### For Your System

Each animation in `scripts/generate_architecture_animations.py` can be customized:

```python
# Example: Customize request flow for your system
def create_request_flow_animation(output_path: str, title: str = "Custom Title"):
    stages = ['Custom Stage 1', 'Custom Stage 2', ...]
    latencies = [50, 200, ...]  # milliseconds
    # Rest of function generates animation
```

### Adding to Architecture Documents

Embed GIF in markdown:

```markdown
## Performance Metrics

### Request Latency Breakdown

The system processes requests through 5 stages with these typical latencies:

![Request Flow Animation](../animations/01-request-flow.gif)

- **Ingestion**: 50ms - Message validation and queuing
- **Intent Classification**: 200ms - Multi-label intent model inference
- **Vector Search**: 150ms - Semantic similarity retrieval from 10K documents
- **LLM Generation**: 1400ms - Multi-turn response generation (GPT-4)
- **Formatting**: 200ms - Response post-processing and channel routing

**Total P99 latency: 2.0 seconds** (within 2s SLA target)
```

---

## Technical Details

### Generation Method

- **Library:** Matplotlib `animation` module
- **Format:** Pillow-based GIF (compatible with GitHub, web browsers)
- **Frame rate:** 20 fps
- **File size:** 150KB-2.2MB (optimized for web viewing)
- **Rendering time:** ~5-10 seconds per animation

### Parameters by Animation Type

| Type | Key Params | Customization |
|------|-----------|---------------|
| Request Flow | `stages`, `latencies` | Component names, latency budget |
| Daily Load | `capacity_qps`, `peak_hour` | Scale targets, traffic pattern |
| Latency | `components`, `latencies_best/avg/worst` | System-specific breakdown |
| Cost | `requests`, `cost_per_1k` | Pricing model, request rate |
| Sentiment | `sentiment_scores`, `escalation_rates` | Threshold, correlation strength |

---

## Usage Examples

### Example 1: Customer Service Platform

```markdown
## Architecture

![Daily Load Pattern](../animations/02-daily-load.gif)

The system handles 50K inquiries/day with peak load at 1pm (3500 QPS).
```

### Example 2: RAG System

```markdown
## Latency Budget Allocation

![Latency Breakdown](../animations/03-latency-breakdown.gif)

Critical path is LLM generation (1400ms of 2000ms budget).
Optimization focus: Enable streaming responses, cache retrievals.
```

### Example 3: Cost Transparency

```markdown
## Operational Cost

![Cost Timeline](../animations/04-cost-timeline.gif)

Daily cost: ~$200. Annual: ~$73K at current scale.
Cost drivers: Token usage (90%), vector search (5%), classification (5%).
```

---

## Extending the Library

### Adding New Animation Types

1. Create a new function in `scripts/generate_architecture_animations.py`:

```python
def create_custom_animation(output_path: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    # ... setup
    
    def animate(frame):
        # Update elements based on frame number
        return [updated_elements]
    
    anim = animation.FuncAnimation(fig, animate, frames=101, ...)
    anim.save(output_path, writer='pillow', fps=20)
    plt.close(fig)
```

2. Register in `generate_all_animations()`:

```python
animations.append((
    'XX-custom-name.gif',
    create_custom_animation,
    'Custom Animation Title'
))
```

### Ideas for Additional Animations

- **Multi-model comparison**: Show accuracy/latency trade-off across models
- **Error rate by component**: Which stages fail most often?
- **Queue depth over time**: Buildup during peaks, drain during valleys
- **Cache hit rate progression**: Warming cache over time
- **Concurrent request handling**: How many parallel requests active?
- **Geographical load distribution**: Traffic by region/timezone

---

## Performance Notes

- Animations loop continuously (suitable for documentation)
- Each GIF is self-contained (~160KB-2.2MB)
- No external dependencies for viewing (standard `.gif` format)
- GitHub renders animations directly in `.md` files
- Web browsers support playback on all platforms

---

## Related Files

- Generation script: `scripts/generate_architecture_animations.py`
- Animations directory: `arch-review/animations/`
- System architectures: `arch-review/systems/*.md`

