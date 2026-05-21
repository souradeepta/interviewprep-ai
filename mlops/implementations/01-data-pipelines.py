"""
Auto-generated from 01-data-pipelines.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Data Pipelines: Designing ETL at Scale
# ## Learning Objectives
# - Understand batch vs streaming pipeline architecture
# - Build production Airflow DAGs with error handling
# ======================================================================

# ======================================================================
# ## Basic Implementation: Simple Airflow DAG
# Minimal example showing core concepts
# ======================================================================

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

# Define tasks
def extract_data():
    """Extract data from source"""
    data = pd.DataFrame({'user_id': [1, 2, 3], 'amount': [10, 20, 30]})
    return data.to_json()

def transform_data(ti):
    """Transform and aggregate"""
    data_json = ti.xcom_pull(task_ids='extract')
    data = pd.read_json(data_json)
    data['amount_scaled'] = data['amount'] * 1.1
    return data.to_json()

# Define DAG
dag = DAG(
    'simple_pipeline',
    start_date=datetime(2026, 5, 16),
    schedule_interval='@daily',
    catchup=False
)

# Define tasks
extract = PythonOperator(task_id='extract', python_callable=extract_data, dag=dag)
transform = PythonOperator(task_id='transform', python_callable=transform_data, dag=dag)

# Set dependencies
extract >> transform

print("✓ Basic DAG created")


# ======================================================================
# ## Advanced Implementation: Production Airflow with Error Handling
# ======================================================================

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_with_retry(max_retries=3):
    """Extract with automatic retry and exponential backoff"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Extracting data, attempt {attempt + 1}")
            # Simulate extraction
            data_rows = 1000
            logger.info(f"✓ Extracted {data_rows} rows")
            return {'rows': data_rows, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"Extraction failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                raise AirflowException(f"Extraction failed after {max_retries} attempts")

def validate_data(ti):
    """Validate extracted data meets quality requirements"""
    data_info = ti.xcom_pull(task_ids='ingestion.extract')
    if data_info['rows'] < 100:
        raise AirflowException(f"Data validation failed: {data_info['rows']} rows (minimum 100)")
    logger.info(f"✓ Data validation passed: {data_info['rows']} rows")

def transform_data(ti):
    """Transform with performance monitoring"""
    data_info = ti.xcom_pull(task_ids='ingestion.extract')
    rows = data_info['rows']
    
    start_time = time.time()
    # Simulate transformation
    transformed_rows = rows
    duration = time.time() - start_time
    
    throughput = transformed_rows / duration if duration > 0 else 0
    logger.info(f"✓ Transformed {transformed_rows} rows in {duration:.2f}s ({throughput:.0f} rows/s)")
    return {'rows_processed': transformed_rows, 'duration': duration}

# Production DAG with task groups
dag = DAG(
    'production_pipeline',
    start_date=datetime(2026, 1, 1),
    schedule_interval='@daily',
    default_view='graph',
    catchup=False,
    tags=['production', 'data-pipeline']
)

with TaskGroup(name='ingestion', dag=dag) as ingestion:
    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_with_retry
    )

with TaskGroup(name='quality', dag=dag) as quality:
    validate = PythonOperator(
        task_id='validate',
        python_callable=validate_data
    )

transform = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag
)

# Define workflow
ingestion >> quality >> transform

logger.info("✓ Production pipeline DAG created")


# ======================================================================
# ## Real-World Example 1: Netflix Feature Pipeline
# Batch pipeline for recommendation features
# ======================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def netflix_batch_pipeline():
    """Netflix daily batch: ingest 1B events → compute features"""

    print("NETFLIX: Feature Pipeline at 1B+ Events/Day")
    print("=" * 60)

    # Simulate 1B events in 24 hours
    events_per_second = 1_000_000_000 / (24 * 3600)
    print(f"Processing: {events_per_second:,.0f} events/second")

    # Simulate processing
    total_events = 1_000_000_000
    num_users = 100_000_000

    # Features computed
    user_embeddings = num_users * 1000  # 1000-dim vectors
    content_similarity = 10_000 * 10_000  # 10K titles

    print(f"\nFeatures computed:")
    print(f"  User embeddings: {num_users:,} users × 1000-dim")
    print(f"  Content similarity: {10_000:,} titles × {10_000:,}")
    print(f"  Total size: ~{(user_embeddings * 8 + content_similarity * 8) / 1e9:.1f} GB")

    print(f"\nTimings:")
    print(f"  Ingestion: 1-2 hours (extract 100GB+ events)")
    print(f"  Features: 4-6 hours (embeddings take time)")
    print(f"  Load to storage: 1 hour")
    print(f"  Total runtime: ~8 hours")

    print(f"\nCost: $1000/day (expensive embeddings)")
    print(f"Freshness: daily (acceptable for embeddings)")

    # Result
    features_ready = {
        'user_embeddings': user_embeddings,
        'content_similarity': content_similarity,
        'timestamp': datetime.now().isoformat(),
        'status': 'ready_for_inference'
    }

    return features_ready

result = netflix_batch_pipeline()
print(f"\n✓ Features ready for {result['timestamp']}")



# ======================================================================
# ## Real-World Example 2: Uber Surge Pricing Pipeline
# Real-time features for pricing decisions
# ======================================================================

import time
import numpy as np

def uber_real_time_pricing():
    """Real-time surge pricing requires <100ms latency"""

    print("UBER: Real-Time Surge Pricing")
    print("=" * 60)

    # Simulate: peak hour in San Francisco
    demand = 500  # ride requests/5min
    supply = 200  # available drivers
    surge_ratio = demand / supply

    print(f"Metrics (live snapshot):")
    print(f"  Demand: {demand} ride requests/5min")
    print(f"  Supply: {supply} drivers available")
    print(f"  Surge ratio: {surge_ratio:.2f}x")

    # Feature computation budget
    print(f"\nLatency budget: 100ms total")

    timings = {}

    # Step 1: Fetch demand from Kafka
    t1 = time.time()
    time.sleep(0.020)  # 20ms
    timings['demand_fetch'] = (time.time() - t1) * 1000

    # Step 2: Fetch supply from Redis cache
    t1 = time.time()
    time.sleep(0.008)  # 8ms
    timings['supply_fetch'] = (time.time() - t1) * 1000

    # Step 3: Compute surge
    t1 = time.time()
    time.sleep(0.015)  # 15ms
    timings['surge_compute'] = (time.time() - t1) * 1000

    # Step 4: Apply pricing model
    t1 = time.time()
    time.sleep(0.025)  # 25ms (model inference)
    timings['model_inference'] = (time.time() - t1) * 1000

    total_latency = sum(timings.values())

    print(f"\nLatency breakdown:")
    for step, latency in timings.items():
        print(f"  {step}: {latency:.1f}ms")
    print(f"  Total: {total_latency:.1f}ms")

    sla_ok = total_latency < 100
    print(f"\n{'✓' if sla_ok else '✗'} Within 100ms SLA: {sla_ok}")

    # Final pricing
    base_price = 15.00
    surge_price = base_price * surge_ratio

    print(f"\nPricing:")
    print(f"  Base: ${base_price:.2f}")
    print(f"  Surge multiplier: {surge_ratio:.2f}x")
    print(f"  Final price: ${surge_price:.2f}")

uber_real_time_pricing()



# ======================================================================
# ## Real-World Example 3: Stripe Fraud Detection Pipeline
# Hybrid batch + streaming for fraud detection
# ======================================================================

import pandas as pd
import numpy as np

def stripe_fraud_pipeline():
    """Hybrid batch + streaming fraud detection"""

    print("STRIPE: Fraud Detection Pipeline")
    print("=" * 60)

    # Batch component (daily training)
    print("\n1. BATCH PIPELINE (Daily Training)")
    batch_stats = {
        'transactions_processed': 1_000_000,
        'confirmed_frauds': 5_000,
        'fraud_rate': 0.5,
    }

    print(f"   Processing: {batch_stats['transactions_processed']:,} txns")
    print(f"   Confirmed frauds: {batch_stats['confirmed_frauds']:,}")
    print(f"   Fraud rate: {batch_stats['fraud_rate']:.2f}%")
    print(f"   Time: 2 hours")
    print(f"   Output: fraud detection model v5")

    # Streaming component (real-time)
    print("\n2. STREAMING PIPELINE (Real-Time)")

    # Simulate transactions
    np.random.seed(42)
    transactions = pd.DataFrame({
        'amount': np.random.lognormal(4.5, 1.2, 100),
        'merchant_type': np.random.choice(['retail', 'online', 'atm'], 100),
        'country': np.random.choice(['US', 'UK', 'FR'], 100),
    })

    # Real-time features
    velocity_score = np.random.uniform(0, 1, len(transactions))
    location_risk = np.random.uniform(0, 1, len(transactions))

    transactions['velocity'] = velocity_score
    transactions['location_risk'] = location_risk

    print(f"   Transactions/second: 10,000")
    print(f"   Real-time features computed: {len(transactions)}")
    print(f"   Features: velocity score, location risk, amount anomaly")
    print(f"   Latency: <50ms per transaction")

    # Fraud scoring
    transactions['fraud_score'] = (
        transactions['velocity'] * 0.4 +
        transactions['location_risk'] * 0.3 +
        (transactions['amount'] / 1000) * 0.3
    )

    fraud_threshold = 0.7
    flagged = (transactions['fraud_score'] > fraud_threshold).sum()

    print(f"\n3. FRAUD DETECTION RESULTS")
    print(f"   Transactions analyzed: {len(transactions)}")
    print(f"   Flagged as fraud (score > {fraud_threshold}): {flagged}")
    print(f"   Action: {flagged} require manual review")

    return transactions

result = stripe_fraud_pipeline()



# ======================================================================
# ## Interview Scenario: Design Fraud Detection Pipeline
# **Question:** "You're at a payment company processing 1M transactions/day. Design a data pipeline for fraud detection that needs to flag fraudulent transactions in <100ms. Fraud labels are confirmed 5 days after the transaction. How would you architect this?"
# **Solution Walkthrough:**
# ======================================================================

# Interview Solution Structure

print("INTERVIEW ANSWER STRUCTURE:")
print()
print("1. CLARIFYING CONSTRAINTS")
print("   - 1M transactions/day = ~10 transactions/second")
print("   - Must score in <100ms (real-time serving)")
print("   - Fraud labels delayed 5 days (training data lag)")
print("   - Implies: batch training + streaming inference")
print()

print("2. ARCHITECTURE (Batch + Streaming)")
print("   Batch (Daily):")
print("   - Ingest confirmed fraud labels from past 5 days")
print("   - Compute features: user history, merchant risk, amount patterns")
print("   - Train fraud detection model")
print("   - Deploy new model version")
print()
print("   Streaming (Real-time):")
print("   - Real-time transaction events → Kafka")
print("   - Compute cheap features: transaction velocity, geographic anomaly")
print("   - Cache features in Redis (<5ms lookup)")
print("   - Score with model + real-time features in <100ms")
print()

print("3. KEY TRADE-OFFS")
print("   - Batch model: accurate (trained on 5 days of data) but stale")
print("   - Streaming features: fresh (real-time velocity) but simple")
print("   - Ensemble: combine model + rules for coverage")
print()

print("4. MONITORING & ITERATION")
print("   - Monitor: fraud detection rate, false positive rate")
print("   - Alert if accuracy drops below 95%")
print("   - Trigger retraining if fraud patterns shift")
print()

print("WHY THIS ANSWER WINS:")
print("✓ Separates batch (training) from streaming (serving)")
print("✓ Handles label delay constraint (5-day lag)")
print("✓ Achieves <100ms latency requirement")
print("✓ Discusses monitoring and iteration")
print("✓ Shows understanding of cost/latency trade-offs")


# ======================================================================
# ## Key Takeaways for Interviews
# **What Interviewers Listen For:**
# 1. Do you understand batch vs streaming trade-offs?
# 2. Can you design for the constraints (throughput, latency, cost)?
# ======================================================================
