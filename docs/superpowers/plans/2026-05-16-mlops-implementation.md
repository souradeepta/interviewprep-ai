# ML Ops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 32 ML Ops concepts with 33 Jupyter notebooks, 150+ FAANG interview questions, 8+ case studies, and complete learning paths.

**Architecture:** Four implementation phases (Foundation → Production → Scale → Advanced) building concepts sequentially. Each phase creates markdown theory files, Jupyter notebooks with multi-tool implementations, interview question sets, and case studies. Tests validate structure and content. Final output: comprehensive ML Ops learning module with interview prep materials.

**Tech Stack:** Python 3.8+, Jupyter, Markdown, MLflow, Airflow, Feast, FastAPI, Kubernetes, pytest

---

## File Structure

### Directory Layout

```
mlops/
├── concepts/              # 32 markdown files
│   ├── 01-data-pipelines.md
│   ├── 02-feature-stores.md
│   ├── ... (03-32)
├── notebooks/             # 33 Jupyter notebooks
│   ├── 00-concept-map.ipynb
│   ├── 01-data-pipelines.ipynb
│   ├── ... (02-32)
├── interview-questions/   # Interview question files
│   ├── questions.json     # All 150+ questions
│   ├── by-concept.json    # Organized by concept
│   └── answers.md         # Sample answers
├── case-studies/          # 8+ full case studies
│   ├── netflix-feature-store.md
│   ├── uber-model-monitoring.md
│   ├── google-mlops-at-scale.md
│   └── ...
├── README.md
├── MLOPS_ROADMAP.md
└── CONTRIBUTING.md

tests/
├── test_mlops_concepts.py      # Concept structure validation
├── test_mlops_notebooks.py     # Notebook structure validation
└── test_mlops_interviews.py    # Interview questions validation
```

### Concept Naming Convention

```
01-data-pipelines.md / 01-data-pipelines.ipynb
02-feature-stores.md / 02-feature-stores.ipynb
...
32-production-best-practices.md / 32-production-best-practices.ipynb
```

---

## Phase 1: Foundation & Data (Concepts 1-8)

### Task 1: Create Project Structure & Templates

**Files:**
- Create: `mlops/`
- Create: `mlops/concepts/`
- Create: `mlops/notebooks/`
- Create: `mlops/interview-questions/`
- Create: `mlops/case-studies/`
- Create: `mlops/README.md`
- Create: `mlops/CONTRIBUTING.md`

- [ ] **Step 1: Create directories**

```bash
mkdir -p /home/sbisw/github/interviewprep-ml/mlops/{concepts,notebooks,interview-questions,case-studies}
```

- [ ] **Step 2: Create mlops/README.md**

```markdown
# ML Ops for Interview Prep & Learning

Comprehensive ML Ops learning materials for FAANG interview preparation.

## What's Inside

- **32 Core ML Ops Concepts** — Data pipelines, feature stores, monitoring, deployment, scaling, governance
- **33 Jupyter Notebooks** — Production code examples with multi-tool approaches
- **150+ FAANG Interview Questions** — Real questions from Google, Meta, Netflix, Uber, Amazon
- **8+ Case Studies** — Full interview walkthroughs with strong/weak answer patterns
- **Interview Prep Learning Paths** — 1-3 week focused learning for different roles

## Organization

Concepts are organized by ML lifecycle stage:
1. Data & Feature Engineering (1-4)
2. Model Development (5-8)
3. Model Evaluation & Testing (9-12)
4. Deployment & Serving (13-16)
5. Monitoring & Observability (17-20)
6. Infrastructure & Scaling (21-24)
7. Governance & Operations (25-28)
8. Advanced Topics (29-32)

## Quick Start

See [MLOPS_ROADMAP.md](MLOPS_ROADMAP.md) for interview prep learning paths.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add concepts or questions.
```

- [ ] **Step 3: Create mlops/CONTRIBUTING.md**

```markdown
# Contributing to ML Ops

## Concept Template

Each concept has:

1. **Markdown file** (`concepts/{NN}-{name}.md`)
   - Comprehensive overview (4-5 paragraphs)
   - How it works
   - Tool comparisons
   - Interview Q&A
   - Best practices
   - Real-world case studies
   - Sample FAANG questions
   - Interview case study
   - Common answer patterns

2. **Notebook file** (`notebooks/{NN}-{name}.ipynb`)
   - Introduction & learning objectives
   - Basic implementation (20-40 lines)
   - Advanced implementation (60-100 lines)
   - 3 real-world examples (40-60 lines each)
   - Interview scenario (annotated code)
   - Key takeaways

## Quality Standards

- All imports must be real (MLflow, Feast, Airflow, etc.)
- Production patterns throughout (error handling, logging)
- Code must run without errors
- Multi-tool approach: show 2-3 implementations
- Interview-focused Q&A (judgment calls, not memorization)

## Adding a Concept

1. Create `concepts/{NN}-{name}.md` using template
2. Create `notebooks/{NN}-{name}.ipynb` using template
3. Add 3-5 interview questions to `interview-questions/questions.json`
4. Update tests to validate new concept
5. Commit with clear message
```

- [ ] **Step 4: Create mlops/MLOPS_ROADMAP.md**

```markdown
# ML Ops Interview Prep Roadmap

## Quick Reference

| Path | Time | Focus | Best For |
|------|------|-------|----------|
| System Design | 1-2 weeks | Concepts 13-24 | ML System Design interviews |
| Full Stack | 2-3 weeks | Concepts 1-32 | Senior ML Engineer roles |
| Data Engineer | 1-2 weeks | Concepts 1-4, 21-24 | Data Engineer roles |
| ML Engineer | 1-2 weeks | Concepts 5-12, 13-16 | ML Engineer roles |

## Path 1: System Design Interview (1-2 weeks)

Study these concepts in order:
- Concept 13: Containerization
- Concept 14: Model Serving
- Concept 16: Deployment Strategies
- Concept 21: Kubernetes
- Concept 22: Orchestration
- Concept 17: Model Monitoring
- Concept 18: Drift Detection
- Concept 19: Logging & Observability
- Concept 20: Alerting
- Concept 23: Distributed Training
- Concept 24: Resource Management
- Concept 15: Model Registry

Practice: 10 case studies + 30 interview questions

## Path 2: Full ML Engineering (2-3 weeks)

Study concepts 1-32 in order.
Practice: All 150+ interview questions + 8 case studies.

## Path 3: Data Engineer (1-2 weeks)

Focus on data and infrastructure:
- Concepts 1-4 (Data & Feature Engineering)
- Concepts 21-24 (Infrastructure & Scaling)
- Concepts 25-26 (Governance & Compliance)

## Path 4: ML Engineer (1-2 weeks)

Focus on model development and serving:
- Concepts 5-12 (Model Development & Testing)
- Concepts 13-16 (Deployment & Serving)
- Concepts 17-20 (Monitoring & Observability)
```

- [ ] **Step 5: Verify structure**

```bash
ls -la /home/sbisw/github/interviewprep-ml/mlops/
# Expected: directories for concepts, notebooks, interview-questions, case-studies
# and files: README.md, CONTRIBUTING.md, MLOPS_ROADMAP.md
```

- [ ] **Step 6: Commit**

```bash
cd /home/sbisw/github/interviewprep-ml
git add mlops/
git commit -m "feat: initialize ML Ops project structure and roadmap

Create mlops directory with:
- concepts/ for 32 markdown concept files
- notebooks/ for 33 Jupyter notebooks
- interview-questions/ for FAANG questions
- case-studies/ for full interview walkthroughs

Add project documentation:
- README.md with overview
- CONTRIBUTING.md with templates and standards
- MLOPS_ROADMAP.md with learning paths

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Create Concept 1-4 Markdown Files (Data & Feature Engineering)

**Files:**
- Create: `mlops/concepts/01-data-pipelines.md`
- Create: `mlops/concepts/02-feature-stores.md`
- Create: `mlops/concepts/03-data-validation.md`
- Create: `mlops/concepts/04-data-versioning.md`

- [ ] **Step 1: Create Concept 1 markdown (Data Pipelines)**

Create `mlops/concepts/01-data-pipelines.md`:

```markdown
# Data Pipelines: Building Reliable ETL for ML

## Comprehensive Overview

Data pipelines form the backbone of ML systems, responsible for ingesting, transforming, and delivering data to models at scale. A production data pipeline must handle millions of events per second, recover from failures gracefully, and maintain data quality while transforming raw inputs into feature-ready datasets. The core challenge: ETL is not just engineering—it's critical infrastructure that determines both model training velocity and inference reliability. Unlike traditional software systems, data pipelines cannot simply drop events; they must preserve completeness while ensuring timeliness and correctness.

Data pipeline design reflects fundamental constraints: batch pipelines excel at historical analytics and model training (high throughput, days of latency acceptable) while streaming pipelines serve real-time features and online serving (millisecond latency, stateful computation). Most production systems need both. The decision between Airflow (DAG orchestration, task dependencies, scheduling), Kubeflow (Kubernetes-native ML workflows, distributed execution), and Luigi (simpler Python workflows) hinges on team expertise, existing infrastructure, and tolerance for complexity. Choosing wrong costs months in context-switching and operational overhead.

Production pipelines face four persistent challenges: data quality (garbage in, garbage out), latency (when should data be available?), cost (compute and storage at scale), and debuggability (why did a run fail at 3am?). Monitoring pipelines is harder than monitoring services—success looks like "data arrived on time," failure looks like "data arrived late" or "data arrived wrong," both requiring different responses. Modern teams implement data contracts (schema validation, completeness checks, freshness guarantees) alongside orchestration, treating data quality as equivalent to code quality.

The business impact is outsized. Netflix features that train on stale data degrade recommendation quality. Uber's surge pricing trained on hours-old traffic patterns misses real-time demand. Stripe's fraud detection trained on day-old transactions misses emerging fraud patterns. Data freshness, quality, and completeness are not engineering details—they directly determine model performance and business outcomes.

## How It Works

### Batch Pipelines

```
Source Systems (APIs, DBs)
    ↓
Ingestion (extract data)
    ↓
Transformation (clean, aggregate, enrich)
    ↓
Feature Store or Data Warehouse
    ↓
Model Training or Inference
```

**Schedule-based execution:** Run at fixed times (daily, hourly) via cron or orchestrator.
**Example:** Daily job runs 2am, extracts yesterday's transactions, computes features, trains model at 3am.
**Latency:** Minutes to hours (acceptable for training, risky for serving).

### Streaming Pipelines

```
Event Stream (Kafka, Pub/Sub)
    ↓
Stateful Transformation (windowing, aggregation)
    ↓
Real-Time Feature Store
    ↓
Online Model Inference
```

**Event-driven execution:** Process data as it arrives.
**Example:** User behavior events → compute real-time features → serve predictions immediately.
**Latency:** Milliseconds (necessary for real-time recommendations, fraud detection).

### Hybrid Architecture (Batch + Streaming)

Most production systems use both:
- **Batch:** Model training (needs historical data)
- **Streaming:** Feature serving (needs real-time features)

The integration point is the feature store, which maintains both historical (batch) and real-time (streaming) features.

## Tool Comparisons

| Tool | Type | Strengths | Weaknesses | Best For |
|------|------|-----------|-----------|----------|
| **Apache Airflow** | Batch orchestration | DAG clarity, large community, great UI, many integrations | Stateless (bad for streaming), scheduling overhead | Medium-complexity batch pipelines, teams with Python expertise |
| **Kubeflow Pipelines** | Kubernetes-native | Native Kubernetes, container-first, good for ML workflows | Steeper learning curve, smaller community, K8s required | ML teams already on Kubernetes |
| **Luigi** | Python-first orchestration | Simple, lightweight, dependency tracking | Less powerful than Airflow, smaller ecosystem | Small teams, simple workflows |
| **Apache Flink** | Streaming | Low-latency processing, stateful computation, complex windowing | Operational complexity, JVM overhead | Real-time feature computation, complex streaming |
| **Spark Streaming** | Batch-like streaming | SQL integration, Hadoop ecosystem, structured APIs | Micro-batch model (not true streaming), higher latency | Medium-latency streaming combined with batch analytics |

**Decision Framework:**
- Batch pipelines: Airflow (safe bet) or Kubeflow (K8s teams)
- Streaming: Flink (low-latency) or Spark (SQL-first)
- Starting out: Luigi (simplicity) or Airflow (community)

## Interview Q&A

**Q: Design a data pipeline for a recommendation system processing 1M events/second. Walk me through your approach.**
A: I'd separate into batch and streaming. Batch: daily job ingests historical user behavior (clicks, purchases), computes user embeddings, updates recommendation model. Streaming: real-time user events compute session features (current browsing context), served to recommendation API. The tension: batch is cheaper (precomputed), streaming is fresher. I'd use batch for heavy computation (embeddings), streaming for lightweight aggregations (session stats). Cost trade-off: streaming infrastructure (Flink or Spark) is expensive; batch is cheap but introduces staleness.

**Q: Your data pipeline failed at 2am, missing the 3am model training deadline. How do you debug?**
A: Systematic debugging: (1) Check orchestrator logs—did the job start? (2) If started, check each task: data ingestion (source available?), transformation (errors?), output (data produced?). (3) Check data quality—did it arrive but fail validation? (4) Check resources—did it timeout due to memory/CPU? (5) Implement data freshness monitoring so I'm alerted before training misses deadline. Prevention: automated retries on transient failures, alerting on lateness, fallback to cached data if real-time fails.

**Q: Your pipeline ingests data from 3 sources with different schemas. How do you handle schema evolution?**
A: Schemas evolve: new fields appear, types change, fields disappear. I'd implement schema validation using tools like Great Expectations. For ingestion: validate incoming data against expected schema, log schema violations. For transformation: use schema-aware libraries (Pydantic, Spark StructType) that enforce types. For breaking changes: plan ahead—deprecate fields before removing, add new fields as optional. Version schemas in data contracts so consumers know expectations. Alert on schema violations rather than silently failing.

**Q: Batch pipeline costs $10K/month. How do you optimize?**
A: Data pipeline costs are compute + storage. To optimize: (1) Reduce data volume—filter unnecessary data early, aggregate before storing. (2) Optimize compute—better algorithms (10M rows in 1 hour vs 4 hours is 4x cheaper), batch size tuning, remove redundant computations. (3) Schedule smartly—run when compute is cheaper, combine related jobs to share overhead. (4) Storage—compress data, delete old snapshots, archive rarely-accessed data. Measurement: track cost per GB processed, aim for cost reduction without sacrificing freshness.

**Q: How do you ensure pipeline reliability and recovery from failures?**
A: Reliability requires: (1) Idempotency—re-run same job 10x, get same result. (2) Checkpointing—restart from failure point, not from beginning. (3) Retries—transient failures (network timeouts) auto-retry; permanent failures (bad code) fail fast. (4) Monitoring—alert on lateness, data quality issues, resource failures. (5) Fallback—if today's data unavailable, use yesterday's cached data. (6) Testing—unit test transformations, integration test with sample data, chaos test failure scenarios.

**Q: Compare batch and streaming pipelines. When would you choose each?**
A: Batch: high throughput, acceptable latency (hours), cost-efficient, stateless. Use for model training (needs historical data), daily reports. Streaming: low latency (milliseconds), event-driven, stateful, higher cost. Use for real-time serving (recommendations, fraud detection), real-time alerts. Most systems need both—batch for training, streaming for serving. The integration point is the feature store.

**Q: Your data pipeline is running slower. Debug the bottleneck.**
A: Profile pipeline stages: which step is slow? (1) I/O bottleneck—data ingestion slow? Parallelize connections, batch fetches. (2) Compute bottleneck—transformation slow? Optimize algorithm, increase compute resources, parallelize processing. (3) Storage bottleneck—writing output slow? Optimize serialization format (Parquet > CSV), increase parallelism. (4) Memory bottleneck—out of memory? Stream data instead of loading all, increase memory. Use observability: log each stage's duration, find the slowest stage, optimize that first.

## Best Practices

1. **Idempotency First:** Design pipelines so re-running same job produces same result. This enables retry-on-failure and manual re-runs without data corruption.

2. **Data Quality as Code:** Use schema validation, row-level checks, and statistical validation (Great Expectations). Treat data quality like code quality—test it, alert on failures.

3. **Monitor Freshness:** Alert if data hasn't arrived by expected time. Freshness is often more critical than perfection (late good data worse than missing data).

4. **Separate Orchestration from Transformation:** Airflow orchestrates; Spark/Pandas do transformation. Don't embed complex logic in DAG definitions.

5. **Versioning and Rollback:** Version schemas, transformations, and feature definitions. Enable rolling back to previous data if issues discovered.

6. **Cost Monitoring:** Track cost per pipeline, per stage. Optimize the most expensive pipelines first.

## Common Pitfalls

1. **Ignoring Skew:** Data isn't uniform. Million-user dataset has power-law distribution—optimize for hot keys, not average case.

2. **Tight Coupling:** Tight coupling between stages causes cascading failures. Build in buffers, use intermediate caching, enable independent stage failure.

3. **Over-Engineering:** Don't build for 1000x scale day one. Build for current scale, optimize when measurement shows bottlenecks.

4. **Silent Failures:** Data arrives but is corrupted. Validate early, fail loud, alert on quality issues.

5. **Forgetting Observability:** Don't realize pipelines failed until models degrade. Implement comprehensive monitoring from day one.

## Real-World Examples

### Netflix: Feature Pipeline

Netflix ingests 1B+ events/day across movies, shows, user interactions. Their batch pipeline:
1. Daily job ingests yesterday's events
2. Aggregates into viewing history (user × content × watch time)
3. Computes content-based features (genre, similar titles)
4. Streams features to Kafka
5. Recommendation models consume in real-time

Cost: optimize by pre-aggregating common queries, caching hot features, archiving old data.

### Uber: Real-Time Surge Pricing

Surge pricing requires real-time features: current demand, supply, travel times. Uber's pipeline:
1. Real-time events from driver/rider apps → Kafka
2. Stream processor computes features: surge_ratio = demand / supply
3. Real-time features served to pricing model
4. Decision returned in <100ms

Latency matters here—old data (5 min old) produces wrong prices.

### Google: Ads ML Pipeline

Google processes 1B+ queries/day with personalized ads. Their pipeline:
1. Batch: historical queries, user behavior → model training
2. Streaming: real-time queries → feature serving
3. Integration: models trained on batch, served with streaming features
4. Feedback loop: model predictions logged, future batch uses to retrain

## Sample Interview Questions

1. "You're building the feature pipeline for an e-commerce recommendation system with 100M users. Walk me through your design."
2. "Your batch pipeline runs at 6am but sometimes finishes at 8am, breaking the 7am report deadline. How do you fix this?"
3. "Design a data pipeline for fraud detection that needs to flag suspicious transactions in <100ms."

## Interview Case Study

**Scenario:** You're hired at Stripe to build the data pipeline for their ML fraud detection system. Current state: fraud model trained weekly on historical transactions. Problem: fraudsters evolve their tactics daily; weekly training is stale.

**Constraints:**
- Process 1M transactions/day
- Label delay: fraudulent transactions confirmed 5 days after fact
- Real-time features needed for inference (fraud score in <50ms)
- Cost sensitive (processing 1M transactions daily is expensive)

**Expected Solution Structure:**
1. Batch pipeline: weekly re-training on confirmed-fraud labels (high latency acceptable)
2. Streaming pipeline: real-time features for active transactions (low latency required)
3. Feature integration: merge batch and streaming features
4. Monitoring: alert if model performance degrades

**What Interviewers Listen For:**
- Understanding of latency/freshness trade-off
- Separating batch (training) from streaming (serving)
- Cost considerations (1M daily events is expensive)
- Monitoring and iteration (feedback loops)

**Strong Answer Pattern:**
"I'd build batch for training—weekly job on confirmed frauds, compute historical features. I'd build streaming for serving—real-time features like transaction velocity, merchant risk. Features merge before model scoring. Monitor: if fraud detection accuracy drops (true positive rate <80%), trigger retraining. If latency increases (inference time >100ms), optimize feature computation or switch to streaming computation."

**Weak Answer Pattern:**
"I'd train the model daily." (Why daily? What if labels not confirmed? Cost implications? How do you handle real-time serving?)

## Common Answer Patterns

**Strong Answer Characteristics:**
1. Clarify constraints (throughput, latency, cost)
2. Separate batch (training) from streaming (serving)
3. Address feature freshness trade-offs
4. Mention monitoring and iteration
5. Show understanding of operational reality (failures happen)

**Weak Answer Characteristics:**
1. "Use Airflow" without explaining why
2. No discussion of latency or cost trade-offs
3. Treating training and serving as same problem
4. No monitoring or observability plan
5. Ignoring schema evolution and data quality
```

- [ ] **Step 2: Create similar markdown files for Concepts 2-4**

Create `mlops/concepts/02-feature-stores.md`, `03-data-validation.md`, `04-data-versioning.md` following same structure as Concept 1.

Use this template for each:
- Comprehensive overview (4-5 paragraphs on the concept)
- How it works (architecture + flow)
- Tool comparisons (2-3 tools, trade-off table)
- Interview Q&A (5-8 questions)
- Best practices (5+ items)
- Real-world examples (2-3 case studies)
- Sample interview questions (3-5)
- Interview case study (full scenario)
- Common answer patterns (strong vs weak)

For Concept 2 (Feature Stores), focus on:
- Feature engineering, versioning, serving
- Tools: Feast, Tecton, Hopsworks
- Real-time vs batch feature serving

For Concept 3 (Data Validation), focus on:
- Data quality, schema validation, anomaly detection
- Tools: Great Expectations, Pandera, Soda
- Data contracts and testing

For Concept 4 (Data Versioning), focus on:
- DVC, data lineage, reproducibility
- Tools: DVC, Delta Lake, Apache Iceberg
- Data lineage tracking

- [ ] **Step 3: Verify markdown structure**

```bash
for file in mlops/concepts/0{1,2,3,4}-*.md; do
  echo "=== $file ==="
  grep -c "##" "$file" # Should have ~9 sections
  wc -l "$file" # Should be 400-600 lines each
done
```

Expected: Each file has ~9 markdown sections, 400-600 lines.

- [ ] **Step 4: Commit**

```bash
cd /home/sbisw/github/interviewprep-ml
git add mlops/concepts/01-*.md mlops/concepts/02-*.md mlops/concepts/03-*.md mlops/concepts/04-*.md
git commit -m "feat: add Data & Feature Engineering concepts (1-4)

Add markdown concept files for:
01-data-pipelines.md - ETL design, batch vs streaming, orchestration
02-feature-stores.md - Feature engineering, versioning, serving
03-data-validation.md - Data quality, schema validation, contracts
04-data-versioning.md - DVC, lineage, reproducibility

Each concept includes:
- Comprehensive overview (4-5 paragraphs)
- How it works with architecture diagrams
- Tool comparisons with trade-off tables
- 5-8 interview Q&A (judgment-focused)
- Best practices from production systems
- Real-world case studies
- FAANG interview question samples
- Full interview case studies
- Strong vs weak answer patterns

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Create Concept 1-4 Jupyter Notebooks

**Files:**
- Create: `mlops/notebooks/00-concept-map.ipynb`
- Create: `mlops/notebooks/01-data-pipelines.ipynb`
- Create: `mlops/notebooks/02-feature-stores.ipynb`
- Create: `mlops/notebooks/03-data-validation.ipynb`
- Create: `mlops/notebooks/04-data-versioning.ipynb`

- [ ] **Step 1: Create master concept map notebook**

Create `mlops/notebooks/00-concept-map.ipynb` with structure:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# ML Ops Concept Map: 32 Core Concepts for Production ML\n",
        "\n",
        "This notebook maps all 32 ML Ops concepts and shows how they interconnect.\n",
        "\n",
        "## ML Lifecycle Stages\n",
        "\n",
        "### Stage 1: Data & Feature Engineering (Concepts 1-4)\n",
        "- 1. Data Pipelines\n",
        "- 2. Feature Stores\n",
        "- 3. Data Validation\n",
        "- 4. Data Versioning\n",
        "\n",
        "### Stage 2: Model Development (Concepts 5-8)\n",
        "- 5. Experiment Tracking\n",
        "- 6. Model Versioning & Registry\n",
        "- 7. Reproducibility\n",
        "- 8. Hyperparameter Optimization\n",
        "\n",
        "... [all stages listed]\n",
        "\n",
        "## Interview Learning Paths\n",
        "\n",
        "### Path 1: System Design (1-2 weeks)\n",
        "Focus: Concepts 13-24\n",
        "\n",
        "### Path 2: Full Stack (2-3 weeks)\n",
        "Focus: Concepts 1-32\n",
        "\n",
        "### Path 3: Role-Specific (1-2 weeks)\n",
        "- Data Engineer: 1-4, 21-24\n",
        "- ML Engineer: 5-12, 13-16\n",
        "- Platform Engineer: 15-24\n",
        "\n",
        "## Statistics\n",
        "- Total concepts: 32\n",
        "- Total notebooks: 33\n",
        "- Interview questions: 150+\n",
        "- Case studies: 8+\n",
        "- Estimated study time: 64 hours (2 hours per concept)\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}
```

- [ ] **Step 2: Create notebooks for Concepts 1-4**

For each concept, create notebook with cells:
1. Title + learning objectives
2. Basic implementation (20-40 lines)
3. Advanced implementation (60-100 lines)
4. Real-world example 1 (40-60 lines)
5. Real-world example 2 (40-60 lines)
6. Real-world example 3 (40-60 lines)
7. Interview scenario (annotated code)
8. Key takeaways

Example for Concept 1 (Data Pipelines):

**Cell 1: Introduction**
```python
# Data Pipelines: Designing ETL at Scale
# Learning Objectives:
# - Understand batch vs streaming pipelines
# - Build a simple Airflow DAG
# - Implement data validation
# - Design for failure recovery
# - Monitor pipeline health

# Prerequisites: Python, understanding of distributed systems
# Interview questions this covers: 10+ real FAANG questions
```

**Cell 2: Basic Airflow DAG**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def extract_data():
    """Extract data from source"""
    import pandas as pd
    data = pd.read_csv('source_data.csv')
    return data.to_dict()

def transform_data(ti):
    """Transform data"""
    data = ti.xcom_pull(task_ids='extract')
    # transformation logic
    return {'rows': len(data), 'columns': len(data[0])}

dag = DAG(
    'simple_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily'
)

extract = PythonOperator(task_id='extract', python_callable=extract_data, dag=dag)
transform = PythonOperator(task_id='transform', python_callable=transform_data, dag=dag)

extract >> transform  # Dependencies
```

**Cell 3: Production Airflow with Error Handling**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException
import logging

def extract_with_retry():
    """Extract with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Extraction logic
            logging.info(f"Extracted {1000} rows")
            return {'rows': 1000, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Extraction failed, retry {attempt+1}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise AirflowException(f"Extraction failed after {max_retries} attempts")

def validate_data(ti):
    """Validate extracted data"""
    data_info = ti.xcom_pull(task_ids='extract')
    if data_info['rows'] < 100:
        raise AirflowException("Data validation failed: insufficient rows")
    logging.info(f"Validation passed: {data_info['rows']} rows")

def transform_data(ti):
    """Transform data with monitoring"""
    data_info = ti.xcom_pull(task_ids='extract')
    start_time = time.time()
    
    # Transformation
    rows_processed = data_info['rows']
    
    duration = time.time() - start_time
    throughput = rows_processed / duration
    
    logging.info(f"Transform: {rows_processed} rows in {duration:.2f}s ({throughput:.0f} rows/s)")
    return {'rows_processed': rows_processed, 'duration': duration}

# DAG with task groups
dag = DAG(
    'production_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    default_view='graph',
    catchup=False,  # Don't backfill missing runs
)

with TaskGroup(name='ingestion', dag=dag) as ingestion:
    extract = PythonOperator(task_id='extract', python_callable=extract_with_retry)

with TaskGroup(name='quality_checks', dag=dag) as quality:
    validate = PythonOperator(task_id='validate', python_callable=validate_data)

transform = PythonOperator(task_id='transform', python_callable=transform_data)

ingestion >> quality >> transform
```

**Cell 4-6: Real-world Examples** (follow similar pattern for different scenarios)

**Cell 7: Interview Scenario**
```python
# Interview Question:
# "Design a data pipeline for fraud detection at a payment company.
# Constraints: 1M transactions/day, fraud labels delayed 5 days,
# need fraud scores in <100ms, cost-sensitive"

# Solution walkthrough with code and explanations
```

**Cell 8: Key Takeaways**
```
# Key concepts from this notebook:
# 1. Batch pipelines for training (daily job on historical data)
# 2. Streaming for inference (real-time features)
# 3. Error handling and retry logic are essential
# 4. Monitoring at each stage (latency, throughput, data quality)
# 5. Idempotency enables safe retries

# For interviews:
# - Understand trade-offs between batch and streaming
# - Know tools: Airflow, Kubeflow, Luigi
# - Be ready to design for 1M+ events/day
# - Think about cost and latency implications
```

- [ ] **Step 3: Verify notebook structure**

```bash
python3 << 'EOF'
import json
import glob

for notebook_path in sorted(glob.glob('/home/sbisw/github/interviewprep-ml/mlops/notebooks/*.ipynb')):
    with open(notebook_path) as f:
        nb = json.load(f)
    cell_count = len(nb['cells'])
    print(f"{notebook_path}: {cell_count} cells")
    # Expected: 8-10 cells per notebook
EOF
```

- [ ] **Step 4: Commit**

```bash
cd /home/sbisw/github/interviewprep-ml
git add mlops/notebooks/00-concept-map.ipynb mlops/notebooks/0{1,2,3,4}-*.ipynb
git commit -m "feat: add Jupyter notebooks for concepts 1-4

Add master concept map notebook (00) showing:
- All 32 concepts organized by ML lifecycle stage
- Interview learning paths (1-3 weeks)
- Study time estimates
- Prerequisite relationships

Add implementation notebooks for concepts 1-4:
- 01-data-pipelines.ipynb: Batch vs streaming, Airflow DAG examples
- 02-feature-stores.ipynb: Feature versioning, real-time serving
- 03-data-validation.ipynb: Schema validation, data contracts
- 04-data-versioning.ipynb: DVC, data lineage, reproducibility

Each notebook includes:
- Learning objectives and prerequisites
- Basic implementation (20-40 lines)
- Advanced implementation with production patterns (60-100 lines)
- 3 real-world examples (40-60 lines each)
- Interview scenario with annotated code
- Key takeaways for interview prep

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Create Interview Questions & Case Studies (Phase 1)

**Files:**
- Create: `mlops/interview-questions/questions.json`
- Create: `mlops/interview-questions/answers.md`
- Create: `mlops/case-studies/stripe-fraud-detection.md`
- Create: `mlops/case-studies/netflix-feature-pipeline.md`

- [ ] **Step 1: Create interview questions JSON**

Create `mlops/interview-questions/questions.json`:

```json
{
  "questions": [
    {
      "id": "dp-001",
      "concept": "01-data-pipelines",
      "difficulty": "medium",
      "company": ["Google", "Meta"],
      "question": "Design a data pipeline for a recommendation system processing 1M events/second. Walk me through your approach.",
      "answer_outline": "Separate batch and streaming. Batch: daily job ingests historical data, trains models. Streaming: real-time events compute session features. Integration point: feature store. Discuss latency vs cost trade-offs.",
      "follow_ups": [
        "How would you handle failures in the batch job?",
        "What if streaming features are unavailable during inference?",
        "How do you monitor data quality?"
      ]
    },
    {
      "id": "dp-002",
      "concept": "01-data-pipelines",
      "difficulty": "hard",
      "company": ["Uber"],
      "question": "Your batch pipeline failed at 2am, missing the 3am model training deadline. How do you debug and prevent future occurrences?",
      "answer_outline": "Debug: check orchestrator logs, task-level logs, data validation errors. Prevent: add monitoring, alerting, retry logic, fallback to cached data.",
      "follow_ups": [
        "What metrics would you monitor?",
        "How do you trade off lateness vs correctness?"
      ]
    },
    ...
  ]
}
```

Add 15-20 questions for concepts 1-4 (5 per concept, varying difficulty).

- [ ] **Step 2: Create case study: Stripe Fraud Detection**

Create `mlops/case-studies/stripe-fraud-detection.md`:

```markdown
# Case Study: Stripe Fraud Detection Pipeline

## Scenario

You're hired at Stripe (payment processing company) to improve their ML Ops for fraud detection.

**Current State:**
- Fraud model trained weekly on historical transactions
- Label delay: fraudulent transactions confirmed 5 days after fact
- Real-time fraud scoring needed (<100ms latency)
- Manual model deployment by on-call engineer
- No monitoring; fraud detection accuracy unknown

**Problem Statement:**
- Weekly training is stale (fraudsters evolve daily)
- Manual deployment is slow (can't respond to new fraud patterns quickly)
- No feedback loop (model never learns from past mistakes)
- Cost concerns (processing 1M transactions/day is expensive)

## Constraints

- **Throughput:** 1M transactions/day
- **Latency:** Real-time fraud score needed in <100ms
- **Label Delay:** Fraudulent transactions confirmed 5 days after fact
- **Cost:** Currently $50K/month on infrastructure; pressure to optimize
- **Team Size:** 2 ML engineers, 1 data engineer

## Design Problem

"Design an end-to-end ML Ops system for fraud detection that: (1) detects fraud patterns faster than weekly training, (2) serves fraud scores in <100ms, (3) automatically improves over time, (4) maintains cost under $60K/month."

## Expected Solution Components

### 1. Batch Pipeline (Training)

Frequency: Daily (more responsive than weekly)
Inputs: Confirmed fraud labels from past 5 days
Process:
```
Daily Confirmed Frauds → Feature Engineering → Model Training → New Model Version
```

Key decisions:
- Train on sliding 5-day window (enables faster response to new fraud)
- Automated deployment on passing validation tests
- Rollback if production metrics degrade

### 2. Streaming Features (Serving)

Real-time features needed for <100ms scoring:
- Transaction velocity (how many transactions in past hour?)
- Merchant risk (how many frauds at this merchant?)
- User behavior (typical for this user?)

Implementation:
```
Kafka Topic (Real-time Transactions)
    ↓
Stream Processor (compute features in real-time)
    ↓
Feature Cache (Redis, low-latency serving)
    ↓
Fraud Model Serving (retrieve features, score)
```

### 3. Feedback Loop

Current issue: Model never learns from mistakes
Solution:
```
Transaction → Prediction → User Confirmed Fraud (5 days later)
    ↓
Log (was this fraud correctly detected?)
    ↓
Offline Evaluation (what's model accuracy?)
    ↓
Trigger Retraining if accuracy drops
```

### 4. Monitoring & Alerting

What to monitor:
- **Latency:** fraud scores taking >100ms? Alert.
- **Accuracy:** fraud detection rate dropping? Alert.
- **Drift:** fraud patterns changing? Flag for investigation.
- **Cost:** pipeline costs increasing? Investigate optimization.

## Strong Answer Pattern

"I'd build three components:

1. **Batch pipeline:** Daily training on past 5 days of confirmed fraud labels. This gives us daily model updates, responding faster than weekly. Automated deployment on validation pass, rollback if metrics degrade. Cost: optimize by sampling negative examples (most transactions are legitimate), caching feature computation.

2. **Streaming features:** Real-time transaction processing computes user/merchant/transaction features in <10ms. Cached in Redis for fraud model serving. Ensemble: combine batch-trained model with rule-based checks (blacklisted merchants, impossible velocity).

3. **Feedback loop:** Log predictions + actual fraud status. Daily offline evaluation computes precision/recall. Alert if recall drops below 95% (missing fraud) or precision drops below 50% (false positives). Trigger retraining if threshold breached.

4. **Monitoring:** Dashboard shows:
   - Latency: p50/p99 fraud score latency
   - Accuracy: daily precision/recall on confirmed frauds
   - Cost: daily pipeline cost, cost per transaction
   - Drift: fraud patterns changed? Investigate.

Trade-offs: Daily training vs weekly (more responsive, more operational overhead). Streaming vs batch features (streaming fresher, higher latency cost). Rule-based vs ML (rules catch obvious fraud, ML catches subtle patterns).

Cost optimization: (1) Sample training data (not all 1M transactions needed for daily training), (2) Cache features, (3) Batch score non-time-sensitive transactions, (4) Archive old data, (5) Schedule expensive computations during off-peak hours."

## Weak Answer Pattern

"I'd train the model daily using Airflow. I'd deploy it automatically. I'd add monitoring."

Problems:
- No mention of latency constraints (100ms is very tight!)
- No discussion of streaming vs batch trade-offs
- No feedback loop design (how does model improve over time?)
- No cost considerations
- Generic approach (what about fraud-specific challenges?)

## Interviewer Follow-ups & Responses

**Follow-up: "Model accuracy drops from 95% to 90%. What do you do?"**

Weak response: "Retrain the model."

Strong response: "First, investigate why: (1) Did fraud patterns change? (2) Did data distribution shift? (3) Did we introduce a bug? Once diagnosed:
- If fraud patterns changed: retraining should help (new patterns in training data)
- If distribution shifted: might need feature engineering changes
- If bug introduced: rollback to previous model version, fix bug, retrain
- Meanwhile: enable fallback to rule-based fraud detection until model fixed"

**Follow-up: "Your pipeline costs are now $80K/month, exceeding budget. Optimize."**

Weak response: "Use cheaper compute."

Strong response: "Diagnosis first: (1) Where's the cost? Feature computation? Model training? Serving? (2) What's driving cost growth?

Options:
- Feature computation: cache intermediate results, reduce feature freshness requirement
- Model training: sample training data (not all 1M transactions needed daily), optimize algorithm
- Serving: batch score non-time-sensitive transactions, use edge model (smaller, faster)
- Data storage: archive old transactions, compress data

For 1M transactions/day: feature computation likely bottleneck. Instead of computing all features real-time, precompute batch features daily, stream only essential real-time features."

## What Interviewers Evaluate

1. **Problem decomposition:** Can you break fraud detection into batch (training) and streaming (serving)?
2. **Latency awareness:** Understand <100ms constraint and how to design for it
3. **Operational thinking:** Monitoring, alerting, rollback strategies
4. **Cost consciousness:** Optimize without sacrificing quality
5. **Feedback loops:** Models should learn over time
6. **Trade-off discussion:** Batch vs streaming, manual vs automatic deployment

## Lessons from Real Stripe System

1. Daily training responds to fraud faster than weekly
2. Rule-based catches obvious fraud (instant, cheap), ML catches subtle (slower, expensive)
3. Ensemble approach: rules + ML gives coverage and interpretability
4. Monitoring accuracy is hard (labels delayed 5 days; how to know if model works?)
5. Cost is real constraint; optimize feature computation first

```

- [ ] **Step 3: Create additional case studies**

Create `mlops/case-studies/netflix-feature-pipeline.md` following similar structure.

Focus on:
- Recommendation system with 1B+ events/day
- Batch features (user embeddings, content similarity)
- Streaming features (current session)
- Integration at feature store

- [ ] **Step 4: Verify interview materials**

```bash
python3 << 'EOF'
import json

with open('/home/sbisw/github/interviewprep-ml/mlops/interview-questions/questions.json') as f:
    data = json.load(f)

questions = data.get('questions', [])
concepts = set(q['concept'] for q in questions)

print(f"Total questions: {len(questions)}")
print(f"Concepts covered: {sorted(concepts)}")
print(f"Difficulty distribution:")
for diff in ['easy', 'medium', 'hard']:
    count = len([q for q in questions if q.get('difficulty') == diff])
    print(f"  {diff}: {count}")
EOF
```

Expected: 15-20 questions, 5 per concept, diverse difficulty levels.

- [ ] **Step 5: Commit**

```bash
cd /home/sbisw/github/interviewprep-ml
git add mlops/interview-questions/ mlops/case-studies/
git commit -m "feat: add interview questions and case studies for Phase 1

Add interview question bank (interview-questions/questions.json):
- 20 real FAANG interview questions for concepts 1-4
- Organized by concept, difficulty, company
- Includes answer outlines and follow-up questions

Add case studies (case-studies/):
- stripe-fraud-detection.md: Full interview scenario with constraints, solution, strong/weak answers, follow-ups
- netflix-feature-pipeline.md: Real-time feature serving at scale
- Includes problem setup, constraints, design approach, evaluation criteria

Each case study demonstrates:
- Problem decomposition
- Architectural thinking
- Trade-off discussion
- Cost and latency awareness
- Operational considerations

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"
```

---

## Summary: Phase 1 Complete

**Concepts 1-4 complete:**
- ✅ 4 comprehensive markdown concept files (1600+ lines total)
- ✅ 5 Jupyter notebooks (master map + 4 implementation notebooks)
- ✅ 20+ interview questions with answer outlines
- ✅ 2+ detailed case studies
- ✅ Common answer patterns documented

**Remaining work (Phases 2-4):**
- Concepts 5-32 (repeat pattern for each batch)
- Expand case studies to 8+ (one per stage)
- Test infrastructure and validation
- Final integration and polish

---

## Phase 2-4 Tasks (High-Level)

### Phase 2: Model Development (Concepts 5-8)
- Experiment tracking, versioning, reproducibility, hyperparameter optimization
- 4 concepts, 4 notebooks, 20+ questions, 1-2 case studies
- Estimated effort: 40-50 hours

### Phase 3: Serving & Monitoring (Concepts 9-16)
- Testing, A/B testing, deployment, monitoring, drift detection
- 8 concepts, 8 notebooks, 40+ questions, 2-3 case studies
- Estimated effort: 60-70 hours

### Phase 4: Infrastructure & Advanced (Concepts 17-32)
- Kubernetes, orchestration, distributed training, governance, advanced topics
- 16 concepts, 16 notebooks, 70+ questions, 2+ case studies
- Estimated effort: 70-80 hours

---

## Testing & Validation

Create tests to validate:
- Markdown structure (required sections present)
- Notebook structure (required cells)
- Code validity (all imports work, code runs)
- Content completeness (no TBD placeholders)

```bash
# Run tests after each phase
pytest tests/test_mlops_concepts.py -v
pytest tests/test_mlops_notebooks.py -v
pytest tests/test_mlops_interviews.py -v
```

---

## Timeline

- **Phase 1:** 2-3 days (foundational concepts, templates established)
- **Phase 2:** 2-3 days (model development, building on templates)
- **Phase 3:** 3-4 days (more complex concepts, more case studies)
- **Phase 4:** 3-4 days (advanced topics, final integration, testing)

**Total:** 2-3 weeks for complete implementation

---

**Next Step:** Execute Phase 1 tasks sequentially, then proceed to Phase 2-4.
