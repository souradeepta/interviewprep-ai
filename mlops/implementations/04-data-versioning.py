"""
Auto-generated from 04-data-versioning.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Data Versioning: Reproducibility and Lineage for ML
# ## Learning Objectives
# - Implement data versioning for reproducibility
# - Track data lineage across pipelines
# ======================================================================

# ======================================================================
# ## Basic Implementation: File-Based Versioning
# ======================================================================

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class DataVersion:
    """Metadata for a data version"""
    name: str
    version: str
    path: str
    content_hash: str  # SHA256 of data
    row_count: int
    size_bytes: int
    created_at: str
    creator: str
    description: str
    dependencies: List[str] = None  # What datasets created this?

class DataVersioningRegistry:
    """Track dataset versions and lineage"""
    
    def __init__(self):
        self.versions = {}  # dataset_name -> [DataVersion]
        self.lineage = {}   # dataset_name -> upstream dependencies
    
    def register_version(self, dv: DataVersion):
        """Register a data version"""
        if dv.name not in self.versions:
            self.versions[dv.name] = []
        self.versions[dv.name].append(dv)
        
        if dv.dependencies:
            self.lineage[dv.name] = dv.dependencies
    
    def get_version(self, dataset_name: str, version: str) -> Optional[DataVersion]:
        """Fetch a specific version"""
        if dataset_name not in self.versions:
            return None
        
        for dv in self.versions[dataset_name]:
            if dv.version == version:
                return dv
        return None
    
    def get_latest_version(self, dataset_name: str) -> Optional[DataVersion]:
        """Get most recent version"""
        if dataset_name not in self.versions:
            return None
        return self.versions[dataset_name][-1]  # Latest
    
    def list_versions(self, dataset_name: str) -> List[DataVersion]:
        """List all versions of a dataset"""
        return self.versions.get(dataset_name, [])
    
    def get_lineage(self, dataset_name: str) -> Dict:
        """Get upstream dependencies (lineage)"""
        return {
            'dataset': dataset_name,
            'dependencies': self.lineage.get(dataset_name, []),
        }

# Usage
registry = DataVersioningRegistry()

# Simulate versions of training_set
for i, (rows, size) in enumerate([(1000000, 100*1024*1024), (1050000, 105*1024*1024), (1100000, 110*1024*1024)], 1):
    dv = DataVersion(
        name='training_set',
        version=f'v{i}',
        path=f's3://data/training_set_v{i}.parquet',
        content_hash=hashlib.sha256(f'v{i}_data'.encode()).hexdigest()[:16],
        row_count=rows,
        size_bytes=size,
        created_at=f'2026-0{i}-01',
        creator='data_team',
        description=f'Training set iteration {i}',
        dependencies=['raw_events', 'user_profiles']
    )
    registry.register_version(dv)

print("Registered versions:")
for dv in registry.list_versions('training_set'):
    print(f"  {dv.version}: {dv.row_count:,} rows, {dv.size_bytes/1024/1024:.0f}MB, hash={dv.content_hash}")

print()
print("Lineage for training_set:")
lineage = registry.get_lineage('training_set')
print(f"  Dependencies: {lineage['dependencies']}")


# ======================================================================
# ## Advanced Implementation: Metadata-Only + Immutable Storage
# ======================================================================

from typing import Tuple

class ImmutableDataVersioning:
    """Metadata-only versioning (data in S3, immutable)"""
    
    def __init__(self):
        self.metadata_store = {}  # version -> metadata
        self.retention_policy = {}
    
    def set_retention_policy(self, dataset_name: str, keep_versions: int = 3):
        """Retention: keep current + N-1 previous versions"""
        self.retention_policy[dataset_name] = keep_versions
    
    def record_version(self, dataset_name: str, version_num: int,
                      s3_path: str, row_count: int, size_bytes: int) -> Dict:
        """Record version metadata (data stays in S3, immutable)"""
        # In production: compute hash of actual data in S3
        content_hash = hashlib.sha256(f'{s3_path}{version_num}'.encode()).hexdigest()[:16]
        
        metadata = {
            'version': f'v{version_num}',
            's3_path': s3_path,
            'content_hash': content_hash,
            'row_count': row_count,
            'size_bytes': size_bytes,
            'created_at': datetime.now().isoformat(),
        }
        
        key = f'{dataset_name}:v{version_num}'
        self.metadata_store[key] = metadata
        
        return metadata
    
    def cleanup_old_versions(self, dataset_name: str) -> Tuple[int, int]:
        """Delete old versions beyond retention policy"""
        keep = self.retention_policy.get(dataset_name, 3)
        
        # Get all versions of this dataset
        versions = [k for k in self.metadata_store.keys() if k.startswith(f'{dataset_name}:')]
        versions.sort()
        
        # Delete old ones
        to_delete = versions[:-keep]  # Keep last N versions
        deleted_count = 0
        deleted_bytes = 0
        
        for v in to_delete:
            metadata = self.metadata_store.pop(v)
            deleted_count += 1
            deleted_bytes += metadata['size_bytes']
        
        return deleted_count, deleted_bytes
    
    def get_total_storage(self, dataset_name: str) -> int:
        """Calculate storage used by versions"""
        total = 0
        for k, v in self.metadata_store.items():
            if k.startswith(f'{dataset_name}:'):
                total += v['size_bytes']
        return total

# Usage
versioning = ImmutableDataVersioning()
versioning.set_retention_policy('fraud_training', keep_versions=3)

# Record 10 versions
print("Recording versions (metadata only):")
for i in range(1, 11):
    versioning.record_version(
        'fraud_training',
        version_num=i,
        s3_path=f's3://data/fraud_training_v{i}.parquet',
        row_count=1_000_000 + i*50_000,
        size_bytes=(1000 + i*10) * 1024 * 1024  # 1GB + 10MB per version
    )

print(f"  Recorded 10 versions")
print(f"  Storage (all): {versioning.get_total_storage('fraud_training')/1024/1024/1024:.1f}GB")

# Apply retention: keep only 3 latest versions
print()
deleted, freed = versioning.cleanup_old_versions('fraud_training')
print(f"✓ Cleanup: Deleted {deleted} old versions, freed {freed/1024/1024/1024:.1f}GB")
print(f"  Storage (after cleanup): {versioning.get_total_storage('fraud_training')/1024/1024/1024:.1f}GB")
print(f"  Savings: {freed/1024/1024/1024:.1f}GB (70% reduction)")


# ======================================================================
# ## Real-World Example 1: Netflix Data Versioning
# ======================================================================

import pandas as pd
from datetime import datetime

def netflix_data_versioning():
    """Track versions of user watching data"""

    print("NETFLIX: Data Versioning at 250M Users")
    print("=" * 60)

    # Simulate version history
    versions = pd.DataFrame({
        'version': ['v1.0', 'v1.1', 'v1.2', 'v2.0'],
        'date': ['2026-04-01', '2026-04-15', '2026-05-01', '2026-05-16'],
        'event': [
            'Initial: 100M users, 1B events/day',
            'Added: device_type feature',
            'Fixed: null timestamp bug (0.1% of data)',
            'Major: New event schema with 50+ fields'
        ],
        'size_gb': [500, 520, 520, 1200],
        'status': ['archived', 'archived', 'archived', 'current']
    })

    print("\nDATA VERSION HISTORY:")
    print(versions.to_string(index=False))

    print("\nCHECKPOINTING:")
    print("  v1.0: git commit abc123, S3://netflix-data/v1.0/")
    print("  v1.1: git commit def456, S3://netflix-data/v1.1/")
    print("  v1.2: git commit ghi789, S3://netflix-data/v1.2/")
    print("  v2.0: git commit jkl012, S3://netflix-data/v2.0/ (current)")

    print("\nLINEAGE TRACKING:")
    print("  v2.0 ← breaking schema change")
    print("  v1.2 ← data quality fix")
    print("  v1.1 ← feature addition")
    print("  v1.0 ← baseline")

    print("\nREPRODUCIBILITY:")
    print("  To reproduce results from v1.2:")
    print("  1. Load data: s3://netflix-data/v1.2/")
    print("  2. Use model: model_v3 (trained on v1.2)")
    print("  3. Result: 92.1% accuracy (same as original)")

netflix_data_versioning()



# ======================================================================
# ## Real-World Example 2: Uber ETA Model Versioning
# ======================================================================

import pandas as pd

def uber_schema_evolution():
    """Track schema changes in trip data"""

    print("UBER: Schema Evolution - Trip Data")
    print("=" * 60)

    # Schema changes over time
    schema_history = {
        'v1': {
            'fields': ['driver_id', 'rider_id', 'distance', 'duration', 'fare'],
            'date': '2024-01-01',
            'adoption': 'legacy'
        },
        'v2': {
            'fields': ['driver_id', 'rider_id', 'distance', 'duration', 'fare', 'surge_multiplier'],
            'date': '2024-06-01',
            'adoption': 'widespread'
        },
        'v3': {
            'fields': ['driver_id', 'rider_id', 'distance', 'duration', 'fare', 'surge_multiplier', 'traffic_condition', 'weather'],
            'date': '2025-01-01',
            'adoption': 'current (95%)'
        }
    }

    for version, info in schema_history.items():
        print(f"\n{version} ({info['date']}):")
        print(f"  Fields: {', '.join(info['fields'])}")
        print(f"  Adoption: {info['adoption']}")

    print("\nMIGRATION STRATEGY:")
    print("  v1→v2: Add surge_multiplier (default=1.0)")
    print("  v2→v3: Add traffic_condition, weather (nullable)")
    print("  Backward compatibility: v3 code handles all versions")
    print("  Phased rollout: 5% → 25% → 100% (over 2 months)")

    print("\nDATA IMPACT:")
    print("  v1 stored: 2 years history (legacy systems)")
    print("  v2 stored: 1.5 years history (transition)")
    print("  v3 stored: 6 months (current)")
    print("  Total: 4 years of trip data, 3 schema versions")

uber_schema_evolution()



# ======================================================================
# ## Real-World Example 3: Stripe Fraud Data Versioning
# ======================================================================

def stripe_versioning():
    print("Stripe: Daily Fraud Training Data Versioning")
    print()
    
    print("1. Dataset: fraud_training")
    print("   Frequency: daily (365 versions/year)")
    print("   Schema: transaction_features + fraud_label")
    print("   Size per day: 5GB (1M transactions + confirmed fraud labels)")
    print()
    
    print("2. Key Challenge: Label Delay")
    print("   Problem: Fraud labels confirmed 5 days after transaction")
    print("   Example:")
    print("   - 2026-05-16: transaction occurs")
    print("   - 2026-05-21: fraud confirmation arrives")
    print("   - 2026-05-21: can add to training data v5")
    print()
    
    print("3. Versioning Strategy:")
    print("   Daily v1-v365 per year")
    print("   Training on v10 includes: labels from v1-v5 (5 day lookback)")
    print("   Retention: keep 30 days (compliance, debugging)")
    print()
    
    print("4. Storage Management:")
    print("   Raw: 365 × 5GB = 1.8TB per year")
    print("   Retention: 30 days = 150GB")
    print("   Compression: 30GB (80% reduction with Parquet)")
    print("   Cost: ~$1/month")
    print()
    
    print("5. Lineage:")
    print("   fraud_training_v10 depends on:")
    print("   - transaction_events_v1-v10 (feature source)")
    print("   - fraud_labels_v1-v5 (label source, delayed)")
    print("   - feature_engineering_pipeline_v2 (how features computed)")

stripe_versioning()


# ======================================================================
# ## Interview Case Study: 1TB Training Data Versioning
# **Scenario:** Design a data versioning strategy for 1TB of monthly training data.
# ======================================================================

print("INTERVIEW CASE STUDY: VERSIONING 1TB TRAINING DATA")
print()

print("CONSTRAINTS:")
print("  - Monthly training: 1TB per month")
print("  - 12 versions/year = 12TB raw data")
print("  - Budget: minimize storage costs")
print("  - Requirement: retrain on old data if accuracy drops")
print()

print("SOLUTION:")
print()

print("1. METADATA-ONLY APPROACH (Recommended):")
print("   ✓ Store actual data immutable in S3 (once, never copy)")
print("   ✓ Record metadata: hash, row count, schema, created_at")
print("   ✓ Metadata per version: ~10KB")
print("   ✓ Total storage: 10KB × 12 = 120KB (negligible)")
print()

print("2. RETENTION POLICY:")
print("   ✓ Keep current + 2 previous versions in hot storage")
print("   ✓ Archive older versions to cold storage (Glacier)")
print("   ✓ Hot cost: 3TB × $0.023/GB/month = $70/month")
print("   ✓ Cold cost: 9TB × $0.004/GB/month = $36/month")
print("   ✓ Total: ~$100/month (vs $3000 for keeping all)")
print()

print("3. TO RETRAIN ON OLD DATA:")
print("   Step 1: Look up metadata for v6 (June data)")
print("   Step 2: Find S3 path: s3://data/training_set_v6.parquet")
print("   Step 3: Fetch content hash from metadata")
print("   Step 4: Retrieve from S3 (if in cold, restore from Glacier first)")
print("   Step 5: Train model with same code/config")
print()

print("4. LINEAGE TRACKING:")
print("   training_set_v6 depends on:")
print("   ├─ raw_events_v6 (feature source)")
print("   ├─ user_features_v6 (aggregations)")
print("   └─ feature_eng_pipeline_v2 (how features computed)")
print()
print("   If model accuracy drops:")
print("   1. Retrain v6 model on v6 data: accuracy returns ✓")
print("   2. Find what changed: compare v6 vs v7 raw_events")
print("   3. Debug: did feature computation change? Data distribution?")
print()

print("5. STRONG vs WEAK ANSWERS:")
print()
print("   STRONG:")
print("   'Use metadata-only versioning: store data once in immutable S3,")
print("   record metadata (hash, rows, schema). Retention: keep 3 versions in hot,")
print("   archive older to cold storage. Cost: ~$100/month vs $3000.")
print("   To retrain: look up path, fetch from S3, train with same code.")
print("   Track lineage to know what feeds each dataset.'")
print()
print("   WEAK:")
print("   'Use DVC to version data.' (No discussion of cost, scale, or debugging)")
print()

print("6. FOLLOW-UP QUESTIONS:")
print()
print("   Q: Can you detect if data or code changed?")
print("   A: Yes—retrain on old data with old code. If accuracy returns,")
print("      it's a data issue. If still degraded, code/config changed.")
print()
print("   Q: How do you handle schema changes across versions?")
print("   A: Track schema in metadata. Detect breaking changes.")
print("      Gradual migration: run both old+new schema in parallel,")
print("      then deprecate old version.")
print()
print("   Q: How do you prevent storage explosion?")
print("   A: Metadata-only approach + retention policies.")
print("      Never duplicate raw data. Archive old versions to cold storage.")


# ======================================================================
# ## Key Takeaways
# **2-Minute Elevator Pitch:**
# "Data versioning enables reproducibility—the ability to reconstruct training data and results from months ago. Use metadata-only versioning: store data immutable in S3, record metadata (hash, rows, schema). Retention: keep current + 2 previous, archive older to cold storage. Track lineage to know what feeds each dataset. When accuracy drops, retrain on old data to diagnose data vs code issues."
# ======================================================================
