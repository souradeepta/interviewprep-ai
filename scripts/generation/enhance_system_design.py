#!/usr/bin/env python3
"""Enhance system-design/patterns files with Mermaid diagrams, code examples, and best practices."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml/system-design/patterns"

ENHANCEMENTS = {
    "02-data-pipelines.md": {
        "mermaid": """```mermaid
graph LR
    A[Raw Data Sources] -->|Connectors| B[Data Lake]
    B -->|Validation| C[Data Quality Checks]
    C -->|Transform| D[Feature Engineering]
    D -->|Store| E[Warehouse/Feature Store]
    E -->|Serve| F[ML Models]
    F -->|Monitor| G[Data Quality Dashboard]
    G -->|Alert on Issues| H[Data Engineering Team]
```""",
        "code": """```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import great_expectations as ge

def validate_data(**context):
    df = pd.read_csv('raw_data.csv')
    ge_df = ge.from_pandas(df)

    # Run expectations
    ge_df.expect_column_values_to_not_be_null('user_id')
    ge_df.expect_column_values_to_be_between('age', 0, 150)

    validation_result = ge_df.validate()
    assert validation_result['success'], "Data validation failed"

def transform_data(**context):
    df = pd.read_csv('raw_data.csv')
    df['age_bucket'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 150])
    df['signup_month'] = pd.to_datetime(df['signup_date']).dt.to_period('M')
    df.to_parquet('processed_data.parquet')

# Define DAG
dag = DAG('data_pipeline', start_date=datetime(2024, 1, 1), schedule_interval='@daily')

validate_task = PythonOperator(task_id='validate', python_callable=validate_data, dag=dag)
transform_task = PythonOperator(task_id='transform', python_callable=transform_data, dag=dag)

validate_task >> transform_task
```""",
        "best_practices": """## Best Practices

- **Idempotent operations:** Pipeline tasks must be rerunnable without side effects (same input → same output).
- **Data quality checks:** Validate at every stage—null checks, type checks, range checks, uniqueness.
- **Incremental processing:** Process only new/changed data. Use watermarks (max timestamp processed) not full reruns.
- **Schema evolution:** Version data schema; handle backward compatibility when columns change.
- **Lineage tracking:** Record data transformations, sources, destinations. Enable debugging and compliance audits.
- **Partitioning strategy:** Partition by time (YYYY/MM/DD/) and domain (org_id/) for fast querying and deletion.
"""
    },

    "03-feature-store.md": {
        "mermaid": """```mermaid
graph LR
    A[Raw Features] -->|Compute| B[Feature Transformations]
    B -->|Store| C[Online Store<br/>Redis/DynamoDB]
    B -->|Store| D[Offline Store<br/>Parquet/Snowflake]
    C -->|Serve| E[Inference API]
    D -->|Backfill| F[Training Dataset]
    F --> G[Model Training]
    G -->|Deploy| E
```""",
        "code": """```python
from feast import FeatureStore, Entity, FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.value_type import ValueType

# Define entities
user = Entity(name="user_id", join_keys=["user_id"])
merchant = Entity(name="merchant_id", join_keys=["merchant_id"])

# Define features
user_features = FeatureView(
    name="user_features",
    entities=[user],
    features=[
        Field(name="account_age_days", value_type=ValueType.INT64),
        Field(name="total_spent", value_type=ValueType.FLOAT),
        Field(name="is_vip", value_type=ValueType.BOOL),
    ],
    source=FileSource(path="data/user_features.parquet"),
    ttl=86400,  # 1 day TTL for online cache
)

# Initialize feature store
store = FeatureStore(repo_path=".")

# Get features for training
training_data = store.get_historical_features(
    entities="SELECT user_id FROM events WHERE date = '2024-01-01'",
    features=["user_features:account_age_days", "user_features:total_spent"],
)

# Get features for inference
online_features = store.get_online_features(
    entity_rows=[{"user_id": 12345}],
    features=["user_features:account_age_days"],
)
```""",
        "best_practices": """## Best Practices

- **Single source of truth:** Define features once; compute once; use everywhere (training, inference, analysis).
- **Offline-online consistency:** Ensure training features = inference features (avoid train-serve skew).
- **Feature versioning:** Track feature definitions, transformations, and lineage over time.
- **TTL for online:** Set appropriate cache TTL; balance freshness vs compute cost.
- **Backfilling strategy:** When adding new features, backfill historical training data to retrain models.
"""
    },

    "05-model-serving.md": {
        "mermaid": """```mermaid
graph LR
    A[Load Balancer] -->|Route Requests| B[Inference API<br/>Replicas]
    B -->|GPU/CPU| C[Model Runtime<br/>ONNX/TorchScript]
    B -->|Cache| D[Response Cache<br/>Redis]
    B -->|Monitor| E[Metrics & Logging]
    D -->|Hit| A
    C -->|Feature Store| F[Feature Lookup]
    F -->|Return| C
```""",
        "code": """```python
from fastapi import FastAPI
from typing import List
import numpy as np
import aioredis
from functools import lru_cache

app = FastAPI()

# Load model once (on startup)
@lru_cache(maxsize=1)
def load_model():
    import onnx
    import onnxruntime as rt
    return rt.InferenceSession("model.onnx")

session = load_model()

# Cache for inference results
cache = None

async def startup_event():
    global cache
    cache = await aioredis.create_redis_pool('redis://localhost')

@app.on_event("startup")
async def startup():
    await startup_event()

@app.post("/predict")
async def predict(features: List[float]):
    # Check cache
    cache_key = str(hash(tuple(features)))
    cached = await cache.get(cache_key)
    if cached:
        return {"prediction": float(cached)}

    # Inference
    input_name = session.get_inputs()[0].name
    input_array = np.array(features).reshape(1, -1).astype(np.float32)

    output = session.run(None, {input_name: input_array})
    prediction = float(output[0][0])

    # Cache result
    await cache.setex(cache_key, 3600, prediction)

    return {"prediction": prediction}
```""",
        "best_practices": """## Best Practices

- **Containerize models:** Use Docker with ONNX/TorchScript for language-agnostic serving.
- **Batch requests:** Group inference into batches for 10-100x throughput improvement.
- **GPU resource allocation:** Use resource limits (nvidia.com/gpu=1) and request limits in Kubernetes.
- **Health checks:** Implement /health endpoint that checks model load, dependencies, GPU memory.
- **Graceful shutdown:** Drain in-flight requests before killing replicas during updates.
"""
    },

    "16-monitoring-and-observability.md": {
        "mermaid": """```mermaid
graph LR
    A[Model Predictions] -->|Log| B[Logging System<br/>ELK/Loki]
    A -->|Metrics| C[Metrics<br/>Prometheus]
    A -->|Traces| D[Tracing<br/>Jaeger]
    B -->|Analyze| E[Dashboard<br/>Grafana]
    C --> E
    D --> E
    E -->|Alert| F[PagerDuty/Slack]
    B -->|Detect Drift| G[Data Drift Monitor]
    G -->|Retrain| H[Retraining Pipeline]
```""",
        "code": """```python
from prometheus_client import Counter, Histogram, Gauge
import logging

# Metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions')
inference_latency = Histogram('model_inference_seconds', 'Inference latency')
prediction_distribution = Gauge('model_output_mean', 'Mean prediction value')

def log_prediction(features, prediction, ground_truth=None):
    prediction_counter.inc()

    with inference_latency.time():
        # Prediction already made; log it
        pass

    # Log to structured logging
    logger.info({
        'event': 'prediction',
        'prediction': prediction,
        'ground_truth': ground_truth,
        'timestamp': datetime.now().isoformat()
    })

    prediction_distribution.set(prediction)

# Drift detection
from scipy.stats import ks_2samp

baseline_predictions = [0.3, 0.45, 0.55, 0.62, 0.71]  # Load from production baseline
recent_predictions = []

def check_prediction_drift():
    if len(recent_predictions) > 100:
        stat, p_value = ks_2samp(baseline_predictions, recent_predictions[-100:])
        if p_value < 0.05:
            logger.warning(f"Prediction distribution drift detected (p={p_value:.4f})")
            # Trigger retraining
```""",
        "best_practices": """## Best Practices

- **Separate concerns:** Logs (what happened), metrics (numbers), traces (latency breakdown).
- **Structured logging:** Use JSON logs with standardized fields for easy parsing and analysis.
- **Alert on business metrics:** Not just technical metrics. Monitor error rate, latency, and model performance together.
- **Baseline + thresholds:** Compare current distributions to baseline; alert on significant drift.
- **SLA-aware alerting:** Don't alert on every anomaly; define SLOs and alert on breaches.
"""
    }
}

def enhance_file(filename, enhancements):
    filepath = os.path.join(BASE, filename)
    if not os.path.exists(filepath):
        print(f"  ✗ {filename} not found")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    # Find insertion points
    has_diagram = "```mermaid" in content
    has_code = "```python" in content
    has_best_practices = "## Best Practices" in content

    improved = False

    # Replace ASCII diagram with Mermaid (if enhancement has it and file doesn't)
    if "mermaid" in enhancements and not has_diagram:
        # Find "## How It Works" or similar
        match = re.search(r'(## How It Works\n\n)```\n.*?\n```', content, re.DOTALL)
        if match:
            # Replace the entire ASCII block
            content = re.sub(
                r'(## How It Works\n\n)```\n.*?\n```\n\n\*\*Key components:\*\*.*?(?=\n\n## )',
                f'\\1{enhancements["mermaid"]}\n\n**Key components:**\n' +
                    (re.search(r'(?<=\*\*Key components:\*\*\n)(.*?)(?=\n\n##)', content, re.DOTALL).group(0)
                     if re.search(r'(?<=\*\*Key components:\*\*\n)(.*?)(?=\n\n##)', content, re.DOTALL)
                     else ''),
                content,
                flags=re.DOTALL
            )
            improved = True

    # Add Best Practices if missing
    if "best_practices" in enhancements and not has_best_practices:
        # Insert before Interview Q&A or before Quick-Reference
        insert_pos = content.find("## Interview")
        if insert_pos != -1:
            content = content[:insert_pos] + enhancements["best_practices"] + "\n\n" + content[insert_pos:]
            improved = True

    # Add Code Examples if missing
    if "code" in enhancements and not has_code:
        # Insert after Best Practices or before Interview Q&A
        if "## Best Practices" in content:
            insert_pos = content.find("## Interview")
            if insert_pos != -1:
                code_section = f"""## Code Examples

### Example: Production Implementation

{enhancements["code"]}

"""
                content = content[:insert_pos] + code_section + content[insert_pos:]
                improved = True

    if improved:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ {filename}")
    else:
        print(f"  - {filename} (skipped)")

print("=== Enhancing system-design/patterns ===\n")

for filename, enhancements in ENHANCEMENTS.items():
    enhance_file(filename, enhancements)

print("\n✅ Done!")
