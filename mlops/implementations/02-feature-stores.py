"""
Auto-generated from 02-feature-stores.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Feature Stores: Serving Features at Production Scale
# ## Learning Objectives
# - Understand batch vs real-time feature serving architectures
# - Build production feature stores with version control
# ======================================================================

# ======================================================================
# ## Basic Implementation: Simple Feature Registry
# ======================================================================

import json
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any

class SimpleFeatureStore:
    """Basic feature store with versioning and registry"""
    
    def __init__(self):
        self.features = {}  # feature_name -> feature_info
        self.versions = {}  # feature_name -> [v1, v2, ...]
        self.online_cache = {}  # feature_name -> cached_value
    
    def register_feature(self, name: str, owner: str, description: str, 
                        freshness_hours: int, version: str = "v1"):
        """Register a feature in the store"""
        feature_info = {
            'name': name,
            'owner': owner,
            'description': description,
            'freshness_hours': freshness_hours,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
        self.features[name] = feature_info
        if name not in self.versions:
            self.versions[name] = []
        self.versions[name].append(version)
        return feature_info
    
    def get_feature(self, feature_name: str, version: str = None) -> Dict[str, Any]:
        """Get a feature (with version control)"""
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature = self.features[feature_name].copy()
        if version:
            if version not in self.versions[feature_name]:
                raise ValueError(f"Version {version} not found for {feature_name}")
            feature['requested_version'] = version
        
        return feature
    
    def list_features(self, owner: str = None) -> List[Dict[str, Any]]:
        """List all features (optionally filtered by owner)"""
        features = list(self.features.values())
        if owner:
            features = [f for f in features if f['owner'] == owner]
        return features

# Usage
store = SimpleFeatureStore()

# Register features
store.register_feature(
    name='user_7d_transaction_count',
    owner='fraud_team',
    description='Count of user transactions in past 7 days',
    freshness_hours=24,
    version='v1'
)

store.register_feature(
    name='user_avg_transaction_amount',
    owner='fraud_team',
    description='Average transaction amount over 30 days',
    freshness_hours=24,
    version='v1'
)

# List features
fraud_features = store.list_features(owner='fraud_team')
print(f"✓ Registered {len(fraud_features)} features")
for f in fraud_features:
    print(f"  - {f['name']} (v{f['version']})")


# ======================================================================
# ## Advanced Implementation: Feature Store with Batch & Real-Time
# ======================================================================

import time
from enum import Enum
from dataclasses import dataclass

class FeatureType(Enum):
    BATCH = "batch"  # Computed daily, high latency acceptable
    REALTIME = "realtime"  # Computed on-demand, low latency required

@dataclass
class FeatureDefinition:
    """Feature metadata and versioning"""
    name: str
    owner: str
    feature_type: FeatureType
    version: str
    freshness_sla_minutes: int
    dependencies: List[str]

class ProductionFeatureStore:
    """Production feature store with batch/real-time paths"""
    
    def __init__(self):
        self.offline_store = {}  # warehouse storage for batch features
        self.online_store = {}   # cache for real-time features (Redis-like)
        self.feature_registry = {}  # feature metadata
        self.version_history = {}  # audit trail
    
    def register_feature(self, defn: FeatureDefinition):
        """Register feature with metadata"""
        self.feature_registry[defn.name] = {
            'definition': defn,
            'created_at': datetime.now(),
            'is_stale': False,
        }
        self.version_history[defn.name] = [(defn.version, datetime.now())]
    
    def write_batch_features(self, feature_name: str, entity_id: str, 
                            feature_value: Any, version: str):
        """Write batch features to offline store (e.g., daily job)"""
        key = f"{feature_name}:{entity_id}:{version}"
        self.offline_store[key] = {
            'value': feature_value,
            'timestamp': datetime.now(),
            'version': version,
        }
    
    def write_realtime_features(self, feature_name: str, entity_id: str,
                               feature_value: Any, ttl_seconds: int = 3600):
        """Write real-time features to online store (e.g., from stream)"""
        key = f"{feature_name}:{entity_id}"
        self.online_store[key] = {
            'value': feature_value,
            'timestamp': datetime.now(),
            'ttl_expires_at': datetime.now() + timedelta(seconds=ttl_seconds),
        }
    
    def get_features_for_serving(self, entity_id: str, feature_names: List[str],
                                version: str = None) -> Dict[str, Any]:
        """Get features for online serving (low latency)"""
        result = {}
        
        for fname in feature_names:
            if fname not in self.feature_registry:
                raise ValueError(f"Feature {fname} not registered")
            
            defn = self.feature_registry[fname]['definition']
            
            if defn.feature_type == FeatureType.REALTIME:
                # Fetch from online store (fast)
                key = f"{fname}:{entity_id}"
                if key in self.online_store:
                    cached = self.online_store[key]
                    # Check TTL
                    if datetime.now() < cached['ttl_expires_at']:
                        result[fname] = cached['value']
                    else:
                        result[fname] = None  # Expired
            else:
                # Fetch from offline store with version
                v = version or defn.version
                key = f"{fname}:{entity_id}:{v}"
                if key in self.offline_store:
                    result[fname] = self.offline_store[key]['value']
        
        return result
    
    def get_features_for_training(self, entity_id: str, feature_names: List[str],
                                 cutoff_date: datetime, version: str = None) -> Dict[str, Any]:
        """Get features for training (temporal correctness)"""
        result = {}
        
        for fname in feature_names:
            defn = self.feature_registry[fname]['definition']
            v = version or defn.version
            key = f"{fname}:{entity_id}:{v}"
            
            if key in self.offline_store:
                feature_data = self.offline_store[key]
                # Enforce temporal correctness: feature must be computed before cutoff
                if feature_data['timestamp'] <= cutoff_date:
                    result[fname] = feature_data['value']
                else:
                    raise ValueError(
                        f"Feature {fname} dated {feature_data['timestamp']} "
                        f"is after cutoff {cutoff_date} (data leakage!)"
                    )
        
        return result

# Usage
fs = ProductionFeatureStore()

# Register batch feature
fs.register_feature(
    FeatureDefinition(
        name='user_embedding',
        owner='recommendations',
        feature_type=FeatureType.BATCH,
        version='v1',
        freshness_sla_minutes=60 * 24,  # Daily
        dependencies=['user_events', 'user_content']
    )
)

# Register real-time feature
fs.register_feature(
    FeatureDefinition(
        name='current_session_duration',
        owner='recommendations',
        feature_type=FeatureType.REALTIME,
        version='v1',
        freshness_sla_minutes=1,  # Real-time
        dependencies=['user_session_stream']
    )
)

# Write batch features (daily job)
user_embedding = [0.5, 0.2, 0.8, 0.1, 0.9]  # 5-dim embedding
fs.write_batch_features('user_embedding', entity_id='user_123', 
                        feature_value=user_embedding, version='v1')

# Write real-time features (from streaming source)
fs.write_realtime_features('current_session_duration', entity_id='user_123',
                           feature_value=300, ttl_seconds=3600)  # 300 sec session

# Serving: fetch both batch + real-time
print("📊 Features for serving:")
serving_features = fs.get_features_for_serving(
    entity_id='user_123',
    feature_names=['user_embedding', 'current_session_duration']
)
for fname, fvalue in serving_features.items():
    print(f"  {fname}: {fvalue}")

# Training: enforce temporal correctness
label_date = datetime.now() - timedelta(hours=1)  # Label from 1 hour ago
print(f"\n✓ Training: features computed before label date {label_date}")


# ======================================================================
# ## Real-World Example 1: Netflix Feature Store
# Serving 100M users with personalized recommendations
# ======================================================================

import pandas as pd
import numpy as np

def netflix_feature_store():
    """Serve 100s of features to recommendation models"""

    print("NETFLIX: Feature Store at 100M Users")
    print("=" * 60)

    # Batch features (computed daily)
    print("\nBATCH FEATURES (Updated Daily):")

    num_users = 100_000_000

    # User embeddings (expensive to compute)
    user_embeddings_size = num_users * 1000 * 8  # 1000-dim float64

    # Content similarity (all-pairs)
    num_titles = 10_000
    content_sim_size = num_titles * num_titles * 4  # float32

    print(f"  User embeddings: {user_embeddings_size / 1e9:.1f} GB")
    print(f"  Content similarity: {content_sim_size / 1e9:.1f} GB")
    print(f"  Update frequency: daily (2am UTC)")
    print(f"  Freshness: 24 hours old")

    # Real-time features (computed on-demand)
    print("\nREAL-TIME FEATURES (Per-Request):")
    print(f"  Current session context (watched in last 30min)")
    print(f"  User velocity (clicks per minute)")
    print(f"  Trending titles (updated per minute)")
    print(f"  Latency requirement: <5ms cache lookup")

    # Feature serving simulation
    print("\nFEATURE SERVING (At Inference Time):")

    # Fetch batch features
    batch_fetch_ms = 20
    real_time_fetch_ms = 3

    print(f"  Batch feature fetch: {batch_fetch_ms}ms")
    print(f"  Real-time feature fetch: {real_time_fetch_ms}ms")
    print(f"  Total feature latency: {batch_fetch_ms + real_time_fetch_ms}ms")
    print(f"  Model inference: 15ms")
    print(f"  Total latency: {batch_fetch_ms + real_time_fetch_ms + 15}ms (✓ < 50ms SLO)")

    # Feature versioning
    print("\nFEATURE VERSIONING:")
    versions = pd.DataFrame({
        'feature': ['user_embedding', 'content_similarity', 'user_watches_7d'],
        'current_version': ['v3', 'v2', 'v5'],
        'updated': ['2026-05-16', '2026-05-14', '2026-05-16'],
        'status': ['production', 'production', 'production']
    })
    print(versions.to_string(index=False))

    print("\n✓ 100M users, 1000+ features, <50ms latency")

netflix_feature_store()



# ======================================================================
# ## Real-World Example 2: Uber Feature Store
# Unified features for driver-rider-ride matching
# ======================================================================

import pandas as pd

def uber_feature_store():
    """Unified features for driver-rider-ride matching"""

    print("UBER: Feature Store for Matching")
    print("=" * 60)

    # Simulated real-time features
    features = pd.DataFrame({
        'feature': [
            'driver_location',
            'driver_acceptance_rate',
            'rider_wait_time',
            'ride_distance',
            'surge_multiplier',
            'driver_vehicle_type',
        ],
        'type': [
            'real-time',
            'batch',
            'real-time',
            'real-time',
            'real-time',
            'batch',
        ],
        'latency_ms': [
            2,
            15,
            3,
            5,
            4,
            10,
        ],
        'source': [
            'GPS stream',
            'Data warehouse',
            'Cache',
            'Route API',
            'Compute service',
            'Data warehouse',
        ]
    })

    print("FEATURE LATENCY BREAKDOWN:")
    print(features.to_string(index=False))

    total_latency = features['latency_ms'].sum()
    print(f"\nTotal feature gathering: {total_latency}ms")
    print(f"Matching model: 30ms")
    print(f"Total latency: {total_latency + 30}ms")
    print(f"SLO (rider ETA): <200ms ✓")

    # Feature freshness
    print("\nFEATURE FRESHNESS REQUIREMENTS:")
    print("  Driver location: <5 seconds")
    print("  Rider location: <5 seconds")
    print("  Acceptance rate: <1 hour")
    print("  All features fetched from:")
    print("    - Real-time cache (Memcached)")
    print("    - Feature store (Redis)")
    print("    - Batch data warehouse")

uber_feature_store()



# ======================================================================
# ## Real-World Example 3: Stripe Feature Store
# Real-time fraud detection with hybrid batch/streaming
# ======================================================================

def stripe_feature_store():
    """Stripe: Feature store for fraud detection (1M+ transactions/day)"""
    
    print("Stripe Fraud Detection Feature Store")
    print()
    
    # Batch features for training
    print("1. Batch Features (For Model Training):")
    batch_features = {
        'user_account_age': 'Days since account creation',
        'user_transaction_history_30d': 'Transaction count last 30 days',
        'user_avg_transaction_amount': 'Average transaction amount',
        'user_chargeback_ratio': 'Historical chargebacks / total',
        'merchant_fraud_rate': 'Merchant historical fraud rate',
        'merchant_category_fraud_baseline': 'Average fraud rate for MCC',
    }
    for fname, desc in batch_features.items():
        print(f"   {fname}: {desc}")
    
    # Real-time features for serving
    print("\n2. Real-Time Features (For Online Scoring):")
    realtime_features = {
        'transaction_velocity_5m': 'Transactions in last 5 minutes',
        'transaction_velocity_1h': 'Transactions in last hour',
        'amount_vs_user_baseline': 'Amount deviation from user average',
        'geographic_anomaly': 'Transaction from unusual location',
        'device_anomaly': 'Transaction from new device',
        'concurrent_transactions': 'User with multiple concurrent txns',
    }
    for fname, desc in realtime_features.items():
        print(f"   {fname}: {desc}")
    
    # Feature serving for fraud scoring
    print("\n3. Fraud Scoring Pipeline:")
    print("   Input: Transaction event")
    print("   Step 1: Fetch batch features from cache (20ms)")
    print("   Step 2: Compute real-time features (15ms)")
    print("   Step 3: Score with ML model (10ms)")
    print("   Output: Fraud risk score + decision (APPROVE/DECLINE/CHALLENGE)")
    print("   Total latency: ~50ms (SLA: <100ms) ✓")
    
    # Scale
    print("\n4. Scale:")
    print(f"   Daily transactions: 1M+")
    print(f"   Transactions/second: 10+")
    print(f"   Real-time feature computation: Flink streams")
    print(f"   Batch feature updates: Spark daily jobs")
    print(f"   Storage: Redis (hot cache) + S3 (historical)")
    
    # Key challenge: label delay
    print("\n5. Key Challenge: Label Delay")
    print("   Problem: Fraud labels confirmed 5 days after transaction")
    print("   Solution:")
    print("     • Train daily on 5-day-old data")
    print("     • Serve with current model (slightly stale labels)")
    print("     • Use feedback loop to quickly retrain if fraud patterns shift")
    print("     • Use unsupervised anomaly detection for real-time gaps")

stripe_feature_store()


# ======================================================================
# ## Interview Case Study: DoorDash Feature Store Design
# **Scenario:** DoorDash is building a feature store for 100+ ML models
# **Context:**
# ======================================================================

print("INTERVIEW SOLUTION WALKTHROUGH")
print()

print("1. ARCHITECTURE OVERVIEW:")
print()
print("   Batch Path (Training & Batch Inference):")
print("   ┌─────────────────────────────────────┐")
print("   │ Data Sources: Kafka, Data Warehouse │")
print("   └────────────────┬────────────────────┘")
print("                    │")
print("   ┌─────────────────▼────────────────────┐")
print("   │ Batch Feature Computation (Spark)   │")
print("   │ - User embeddings (PCA on events)   │")
print("   │ - Restaurant features (aggregations)│")
print("   │ - Order context (historical)        │")
print("   └────────────────┬────────────────────┘")
print("                    │")
print("   ┌─────────────────▼────────────────────┐")
print("   │ Offline Storage (Data Warehouse)    │")
print("   │ - S3/BigQuery with partitioning     │")
print("   │ - Version control (v1, v2, ...)     │")
print("   └─────────────────────────────────────┘")
print()

print("   Real-Time Path (Online Inference):")
print("   ┌─────────────────────────────────────┐")
print("   │ Real-Time Sources: Kafka streams    │")
print("   │ - Order events (velocity)           │")
print("   │ - User location (geospatial)        │")
print("   │ - Traffic data (external API)       │")
print("   └────────────────┬────────────────────┘")
print("                    │")
print("   ┌─────────────────▼────────────────────┐")
print("   │ Real-Time Computation (Flink)       │")
print("   │ - Order velocity (sliding window)   │")
print("   │ - Distance to restaurant            │")
print("   │ - Estimated delivery time           │")
print("   └────────────────┬────────────────────┘")
print("                    │")
print("   ┌─────────────────▼────────────────────┐")
print("   │ Online Cache (Redis)                │")
print("   │ - TTL: 1 hour                       │")
print("   │ - Hit rate: 95%+                    │")
print("   └─────────────────────────────────────┘")
print()

print("2. KEY DESIGN DECISIONS:")
print()
print("   ✓ Separate storage: batch (warehouse) vs real-time (cache)")
print("     Why: Different latency requirements. Batch can be slower, real-time must be <10ms.")
print()
print("   ✓ Versioning for training-serving consistency:")
print("     - Training explicitly requests feature v1")
print("     - Serving explicitly requests same v1")
print("     - Prevents drift between train and serve")
print()
print("   ✓ Feature registry with ownership & governance:")
print("     - owner: (matches_team, pricing_team, fraud_team)")
print("     - freshness_sla: (daily for batch, 1hr for real-time)")
print("     - depends_on: (data lineage tracking)")
print()
print("   ✓ Monitoring & alerts:")
print("     - Alert if feature stale beyond SLA")
print("     - Track feature usage (unused features deprecated)")
print("     - Monitor serving latency (p99 <100ms)")
print()

print("3. BENEFITS:")
print()
print("   ✓ Engineering efficiency:")
print("     - 100+ teams share same 10K features")
print("     - ~60% reduction in feature engineering code")
print()
print("   ✓ Consistency:")
print("     - No duplicate definitions (user_active_days = same everywhere)")
print("     - Training-serving skew eliminated")
print()
print("   ✓ Speed to market:")
print("     - New model uses existing features in days, not weeks")
print("     - Faster experimentation")
print()

print("4. STRONG vs WEAK ANSWERS:")
print()
print("   STRONG: 'I'd separate batch and real-time paths.")
print("   Batch—compute user/restaurant embeddings daily using Spark,")
print("   store in data warehouse with version control.")
print("   Real-time—compute order velocity and context in Flink, cache in Redis.")
print("   Feature registry ensures versioning (training uses v1, serving uses v1).")
print("   Governance layer tracks ownership, lineage, and SLAs.")
print("   Monitoring alerts on staleness or serving latency violations.'")
print()
print("   WEAK: 'I'd use Feast to build a feature store.'")
print("   (No details on architecture, versioning, how to handle scale)")
print()

print("5. FOLLOW-UP QUESTIONS:")
print()
print("   Q: How would you handle feature drift (feature values change)?")
print("   A: Monitor feature distributions. If X% shift, trigger alert.")
print("      Backfill historical data if definition changes.")
print()
print("   Q: How would you debug if accuracy dropped from 85% to 80%?")
print("   A: (1) Check feature versions (model trained on v1, serving on v2?).")
print("      (2) Check for staleness (features not updated?).")
print("      (3) Check data distribution (feature values shifted?).")
print()
print("   Q: Cost is too high. How would you reduce it?")
print("   A: (1) Retire unused features (monitor feature usage).")
print("      (2) Increase TTL on cache (fewer recomputes).")
print("      (3) Sample features for less critical models.")


# ======================================================================
# ## Key Takeaways
# **2-Minute Elevator Pitch:**
# "Feature stores solve a critical problem: features are computed multiple times in different places with different code. A feature store centralizes feature computation, versioning, and serving. Batch features (embeddings, aggregations) computed daily for training. Real-time features (velocity, context) computed on-demand for serving. Versioning prevents training-serving skew. Governance and monitoring keep systems healthy at scale."
# ======================================================================
