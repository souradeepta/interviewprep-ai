# Architecture Animation Code

Standalone Python scripts for generating customizable matplotlib animations. Each script generates a specific animation type with command-line arguments for customization.

## Quick Start

### Generate all animations
```bash
cd arch-review/animation-code
python3 01-request-flow-animation.py --output ../animations/01-custom.gif
python3 02-daily-load-animation.py --output ../animations/02-custom.gif
python3 03-latency-breakdown-animation.py --output ../animations/03-custom.gif
python3 04-cost-timeline-animation.py --output ../animations/04-custom.gif
python3 05-sentiment-escalation-animation.py --output ../animations/05-custom.gif
```

### Customize for your system
```bash
# Request flow with your system's pipeline
python3 01-request-flow-animation.py \
  --output myanimation.gif \
  --title "My System Request Flow" \
  --stages "Parse" "Validate" "Process" "Format" "Send" \
  --latencies 50 100 200 150 100

# Daily load with your peak hours
python3 02-daily-load-animation.py \
  --output myload.gif \
  --peak-qps 5000 \
  --capacity-qps 8000 \
  --peak-hour 14
```

## Animation Scripts

### 1. Request Flow Animation
**File:** `01-request-flow-animation.py`

Shows a request flowing through pipeline stages with latency at each stage.

**Parameters:**
```bash
--output FILE                    Output GIF path (default: 01-request-flow.gif)
--title TEXT                     Animation title
--stages STAGE1 STAGE2 ...       Stage names (space-separated)
--latencies MS1 MS2 ...          Latencies in milliseconds (space-separated, ints)
```

**Example:**
```bash
python3 01-request-flow-animation.py \
  --title "Code Review Agent Pipeline" \
  --stages "Repository Parse" "AST Analysis" "Style Check" "Generate Comments" \
  --latencies 100 300 200 800
```

**Output:** Shows request flowing through stages → latency breakdown chart

---

### 2. Daily Load Pattern Animation
**File:** `02-daily-load-animation.py`

Shows 24-hour traffic pattern with peak hours and capacity visualization.

**Parameters:**
```bash
--output FILE           Output GIF path (default: 02-daily-load.gif)
--title TEXT            Animation title
--peak-qps INT          Peak requests per minute (default: 3500)
--capacity-qps INT      System capacity QPS (default: 5000)
--peak-hour INT         Hour of peak (0-23, default: 13)
```

**Example:**
```bash
# System with 5000 QPS peak, capacity 8000, peak at 2pm
python3 02-daily-load-animation.py \
  --title "High-Scale RAG Service" \
  --peak-qps 5000 \
  --capacity-qps 8000 \
  --peak-hour 14
```

**Output:** 
- Top chart: requests/min over 24 hours with capacity line
- Bottom chart: concurrent users with capacity headroom

---

### 3. Latency Component Breakdown Animation
**File:** `03-latency-breakdown-animation.py`

Shows best/average/worst case latency breakdown by component.

**Parameters:**
```bash
--output FILE           Output GIF path (default: 03-latency-breakdown.gif)
--title TEXT            Animation title
--components COMP1 ...  Component names (space-separated)
--best MS1 ...          Best case latencies (space-separated, ints)
--avg MS1 ...           Average case latencies (space-separated, ints)
--worst MS1 ...         Worst case latencies (space-separated, ints)
```

**Example:**
```bash
python3 03-latency-breakdown-animation.py \
  --title "Translation Service Latency" \
  --components "Detect Language" "Translate" "Quality Check" \
  --best 50 200 30 \
  --avg 100 500 50 \
  --worst 200 1200 100
```

**Output:** Stacked bar chart showing component contribution to total latency

---

### 4. Cost Timeline Animation
**File:** `04-cost-timeline-animation.py`

Shows daily API cost accumulation based on token usage.

**Parameters:**
```bash
--output FILE                  Output GIF path (default: 04-cost-timeline.gif)
--title TEXT                   Animation title
--peak-qps INT                 Peak QPS (default: 3500)
--peak-hour INT                Hour of peak (0-23, default: 13)
--tokens-per-request INT       Avg tokens/request (default: 4000)
--cost-per-1k FLOAT            Cost per 1000 tokens (default: 0.001)
```

**Example:**
```bash
# System with 2000 tokens/request at $0.002/1K tokens
python3 04-cost-timeline-animation.py \
  --title "FineTuning Platform Daily Cost" \
  --peak-qps 2000 \
  --tokens-per-request 2000 \
  --cost-per-1k 0.002
```

**Output:**
- Left chart: hourly cost (bar chart)
- Right chart: cumulative daily cost accumulation

---

### 5. Sentiment vs Escalation Animation
**File:** `05-sentiment-escalation-animation.py`

Shows scatter plot of sentiment score vs escalation rate correlation.

**Parameters:**
```bash
--output FILE            Output GIF path (default: 05-sentiment-escalation.gif)
--title TEXT             Animation title
--num-samples INT        Number of data points (default: 500)
--sentiment-mean FLOAT   Sentiment distribution mean (default: 0.3)
--sentiment-std FLOAT    Sentiment distribution std dev (default: 0.35)
```

**Example:**
```bash
python3 05-sentiment-escalation-animation.py \
  --title "Support Bot Escalation Correlation" \
  --num-samples 1000
```

**Output:** 
- Scatter plot with decision boundary
- Color gradient shows escalation rate
- Risk regions highlighted (red: high, green: low)

---

## Integration with Architecture Documents

### Step 1: Generate Custom Animation
```bash
python3 03-latency-breakdown-animation.py \
  --output ../animations/my-system-latency.gif \
  --title "My System Latency" \
  --components "Component A" "Component B" "Component C" \
  --best 50 100 75 \
  --avg 100 200 150 \
  --worst 300 500 400
```

### Step 2: Add to Markdown
```markdown
## Performance Analysis

### Latency Breakdown

![Latency Animation](../animations/my-system-latency.gif)

- **Component A**: 100ms average (best: 50ms, worst: 300ms)
- **Component B**: 200ms average (best: 100ms, worst: 500ms)
- **Component C**: 150ms average (best: 75ms, worst: 400ms)

**Total: 450ms** — Optimization focus on Component B
```

---

## Advanced Customization

### Modify Animation Duration
Edit the animation parameters in the function:
```python
# Change animation duration (frames=120 means 6 seconds at 20fps)
anim = animation.FuncAnimation(fig, animate, frames=120, interval=50, ...)
# frames=200 = 10 seconds, frames=60 = 3 seconds
```

### Change Colors
Modify the `colors` list at the top of each script:
```python
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
# Use any hex color: https://www.color-hex.com/
```

### Adjust Figure Size
```bash
python3 01-request-flow-animation.py --output out.gif
# Then edit the script:
# figsize=(14, 5) → figsize=(16, 6) for larger output
```

---

## Performance Tips

- **Fast generation**: Animations take 5-10 seconds to generate
- **File size**: GIFs are 150KB-2.2MB (optimized for web)
- **Rendering**: Uses Pillow writer (no FFmpeg needed)
- **Quality**: 20 fps default (good balance between quality and size)

### Reduce file size
```python
# In script, change fps from 20 to 10:
anim.save(output_path, writer='pillow', fps=10)
```

---

## Examples

### Customer Service Platform
```bash
python3 01-request-flow-animation.py \
  --output cust-service-flow.gif \
  --title "Customer Service Platform Pipeline" \
  --stages "Ingest" "Intent" "RAG" "LLM" "Format" \
  --latencies 50 200 150 1400 200
```

### RAG Document QA
```bash
python3 02-daily-load-animation.py \
  --output rag-load.gif \
  --title "RAG Service Daily Load" \
  --peak-qps 2000 \
  --capacity-qps 3500 \
  --peak-hour 14
```

### Code Analysis Agent
```bash
python3 03-latency-breakdown-animation.py \
  --output code-analysis-latency.gif \
  --title "Code Analysis Agent" \
  --components "Parse" "Analyze" "Generate" \
  --best 100 200 300 \
  --avg 200 500 800 \
  --worst 400 1000 1500
```

---

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named matplotlib"
```bash
pip install matplotlib numpy pillow
```

**Issue**: GIF has low quality
```bash
# Increase fps and frame count in script
anim = animation.FuncAnimation(fig, animate, frames=200, interval=30, fps=30)
```

**Issue**: Animation looks different than expected
```bash
# Check that number of stages matches number of latencies
--stages "A" "B" "C"      # 3 stages
--latencies 100 200 300   # 3 latencies ✓
```

---

## Extending the Library

### Create New Animation Type
1. Create `XX-myanimation.py` 
2. Implement `create_myanimation_animation()` function
3. Use matplotlib animation framework
4. Test with: `python3 XX-myanimation.py --output test.gif`

### Register in Generator Script
Add to `scripts/generate_architecture_animations.py`:
```python
animations.append((
    'XX-myanimation.gif',
    create_myanimation_animation,
    'My Animation Title'
))
```

---

## Related Files

- Main generator: `scripts/generate_architecture_animations.py`
- Animation guide: `ANIMATIONS.md`
- Generated GIFs: `animations/`
- Architecture systems: `systems/*.md`
