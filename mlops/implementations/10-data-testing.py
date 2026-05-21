"""
Auto-generated from 10-data-testing.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Data Testing: Validating Input Quality for Production ML
# ## Learning Objectives
# - Implement schema and completeness validation
# - Detect distribution shifts in data
# ======================================================================

# ======================================================================
# ## Basic Implementation: Schema & Completeness Validation
# ======================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

@dataclass
class ColumnValidator:
    """Validates a single column"""
    name: str
    dtype: type
    required: bool = True  # Can be null?
    allowed_range: Tuple = None  # (min, max)
    allowed_values: List = None  # Categories

class DataQualityValidator:
    """Basic data validation framework"""
    
    def __init__(self, validators: List[ColumnValidator]):
        self.validators = {v.name: v for v in validators}
        self.issues = []
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """Run all validation checks"""
        print("\n=== DATA VALIDATION REPORT ===")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print()
        
        results = {'passed': 0, 'warnings': 0, 'errors': 0}
        
        for col_name, validator in self.validators.items():
            if col_name not in df.columns:
                print(f"❌ MISSING COLUMN: {col_name}")
                results['errors'] += 1
                continue
            
            col = df[col_name]
            
            # Check 1: Type
            print(f"\n{col_name}:")
            if col.dtype != validator.dtype:
                print(f"  ⚠ Type mismatch: {col.dtype} vs expected {validator.dtype}")
                results['warnings'] += 1
            else:
                print(f"  ✓ Type OK ({validator.dtype})")
            
            # Check 2: Nulls
            null_pct = col.isnull().sum() / len(col) * 100
            if validator.required and null_pct > 0:
                print(f"  ❌ Required field has {null_pct:.2f}% nulls")
                results['errors'] += 1
            elif null_pct > 5:
                print(f"  ⚠ {null_pct:.2f}% nulls (watch)")
                results['warnings'] += 1
            else:
                print(f"  ✓ Completeness OK ({100-null_pct:.2f}%)")
            
            # Check 3: Range
            if validator.allowed_range:
                min_val, max_val = validator.allowed_range
                col_clean = col.dropna()
                out_of_range = ((col_clean < min_val) | (col_clean > max_val)).sum()
                if out_of_range > 0:
                    pct = out_of_range / len(col_clean) * 100
                    print(f"  ⚠ {out_of_range} values out of range [{min_val}, {max_val}] ({pct:.2f}%)")
                    results['warnings'] += 1
                else:
                    print(f"  ✓ Range OK [{min_val}, {max_val}]")
            
            # Check 4: Categories
            if validator.allowed_values:
                col_clean = col.dropna()
                invalid = ~col_clean.isin(validator.allowed_values)
                if invalid.sum() > 0:
                    pct = invalid.sum() / len(col_clean) * 100
                    print(f"  ⚠ {invalid.sum()} invalid categories ({pct:.2f}%)")
                    print(f"    Unique values: {col_clean.unique()[:5]}")
                    results['warnings'] += 1
                else:
                    print(f"  ✓ Categories OK")
        
        return results

# Example: Airbnb listings
data = {
    'listing_id': [1, 2, 3, 4, 5],
    'price': [150.0, 200.0, None, 250.0, -50.0],  # Missing and negative
    'country': ['USA', 'USA', 'France', 'France', 'Canada'],
    'rating': [4.5, 5.0, 3.8, 4.2, 5.5],  # 5.5 is invalid
    'reviews': [10, 20, 5, 15, 30]
}
df = pd.DataFrame(data)

# Define validators
validators = [
    ColumnValidator('listing_id', dtype=np.int64, required=True),
    ColumnValidator('price', dtype=np.float64, required=True, allowed_range=(0, 50000)),
    ColumnValidator('country', dtype=object, required=True, allowed_values=['USA', 'France', 'Canada']),
    ColumnValidator('rating', dtype=np.float64, allowed_range=(1, 5)),
    ColumnValidator('reviews', dtype=np.int64, allowed_range=(0, None))
]

# Validate
validator = DataQualityValidator(validators)
results = validator.validate(df)

print(f"\nSUMMARY: {results['passed']} passed, {results['warnings']} warnings, {results['errors']} errors")


# ======================================================================
# ## Advanced Implementation: Distribution Monitoring & Drift Detection
# ======================================================================

from scipy.stats import ks_2samp, chi2_contingency

class DistributionMonitor:
    """Monitor data distributions and detect drift"""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistics on dataset"""
        stats = {}
        for col in df.columns:
            col_data = df[col].dropna()
            if col_data.dtype in [np.float64, np.int64]:
                stats[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                    'p95': col_data.quantile(0.95)
                }
            else:
                stats[col] = {'categories': col_data.unique().tolist()}
        return stats
    
    def detect_drift(self, production_data: pd.DataFrame, alert_threshold: float = 0.05) -> Dict:
        """Detect distribution shifts using KS test"""
        print("\n=== DISTRIBUTION DRIFT DETECTION ===")
        drifts = {}
        
        for col in self.reference.columns:
            ref_col = self.reference[col].dropna()
            prod_col = production_data[col].dropna()
            
            if ref_col.dtype in [np.float64, np.int64]:
                # Use KS test for numerical columns
                statistic, p_value = ks_2samp(ref_col, prod_col)
                has_drift = p_value < alert_threshold
                
                print(f"\n{col}:")
                print(f"  Reference mean: {ref_col.mean():.2f}")
                print(f"  Production mean: {prod_col.mean():.2f}")
                print(f"  KS statistic: {statistic:.3f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Drift detected: {'YES ⚠' if has_drift else 'NO ✓'}")
                
                drifts[col] = {
                    'has_drift': has_drift,
                    'p_value': p_value,
                    'statistic': statistic,
                    'ref_mean': ref_col.mean(),
                    'prod_mean': prod_col.mean()
                }
        
        return drifts
    
    def check_data_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Check completeness and outliers"""
        print("\n=== DATA QUALITY METRICS ===")
        metrics = {}
        
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            metrics[col] = {'null_pct': null_pct}
            
            # Outlier detection (values outside 3 std from mean)
            col_data = df[col].dropna()
            if col_data.dtype in [np.float64, np.int64]:
                mean = col_data.mean()
                std = col_data.std()
                outliers = ((col_data < mean - 3*std) | (col_data > mean + 3*std)).sum()
                outlier_pct = outliers / len(col_data) * 100
                metrics[col]['outlier_pct'] = outlier_pct
                
                print(f"{col}:")
                print(f"  Null: {null_pct:.2f}%")
                print(f"  Outliers (>3σ): {outlier_pct:.2f}%")
        
        return metrics

# Simulate: training data vs production data
np.random.seed(42)

# Training data (reference)
train_data = pd.DataFrame({
    'price': np.random.normal(150, 30, 1000),  # mean=150, std=30
    'rating': np.random.normal(4.2, 0.5, 1000),
    'reviews': np.random.poisson(15, 1000)
})

# Production data (slightly different distribution)
prod_data = pd.DataFrame({
    'price': np.random.normal(180, 35, 1000),  # shifted to higher prices
    'rating': np.random.normal(4.0, 0.6, 1000),  # slightly lower
    'reviews': np.random.poisson(15, 1000)
})

monitor = DistributionMonitor(train_data)

# Detect drift
drifts = monitor.detect_drift(prod_data)

# Check quality metrics
quality = monitor.check_data_quality_metrics(prod_data)

print("\n=== DRIFT SUMMARY ===")
for col, drift in drifts.items():
    if drift['has_drift']:
        print(f"⚠ {col}: Distribution shifted (p={drift['p_value']:.4f})")


# ======================================================================
# ## Real-World Example 1: Stripe Transaction Validation
# ======================================================================

import pandas as pd
import numpy as np

def stripe_transaction_validation():
    """Validate 500M+ transactions/day"""

    print("STRIPE: Transaction Data Validation")
    print("=" * 60)

    # Simulate transactions
    np.random.seed(42)
    n_txns = 100_000  # sample of 500M/day

    # Clean transactions (99%)
    clean_txns = pd.DataFrame({
        'amount': np.random.lognormal(4.5, 1.2, int(n_txns * 0.99)),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], int(n_txns * 0.99)),
        'merchant_id': np.random.randint(1, 100000, int(n_txns * 0.99)),
    })

    # Corrupted transactions (1%)
    corrupt_txns = pd.DataFrame({
        'amount': np.concatenate([
            [-50] * 250,  # negative amounts
            [999999] * 250,  # outlier amounts
            [0] * 250,  # zero amounts
        ]),
        'currency': ['INVALID'] * 750,
        'merchant_id': [0] * 750,
    })

    # Combine
    df = pd.concat([clean_txns, corrupt_txns], ignore_index=True)

    print(f"Total transactions: {len(df):,}")
    print()

    # Validation 1: Amount range
    valid_amount = (df['amount'] > 0) & (df['amount'] < 999999)
    print(f"VALIDATION 1 - Amount Range (0, 999999):")
    print(f"  Valid: {valid_amount.sum():,}")
    print(f"  Invalid: {(~valid_amount).sum():,}")
    print()

    # Validation 2: Currency
    valid_currencies = df['currency'].isin(['USD', 'EUR', 'GBP'])
    print(f"VALIDATION 2 - Currency (USD, EUR, GBP):")
    print(f"  Valid: {valid_currencies.sum():,}")
    print(f"  Invalid: {(~valid_currencies).sum():,}")
    if (~valid_currencies).sum() > 0:
        print(f"  Found: {df[~valid_currencies]['currency'].unique()}")
    print()

    # Validation 3: Merchant ID
    valid_merchant = df['merchant_id'] > 0
    print(f"VALIDATION 3 - Merchant ID (> 0):")
    print(f"  Valid: {valid_merchant.sum():,}")
    print(f"  Invalid: {(~valid_merchant).sum():,}")
    print()

    # Final clean dataset
    clean = df[valid_amount & valid_currencies & valid_merchant]
    print(f"FINAL RESULT:")
    print(f"  Clean transactions: {len(clean):,}")
    print(f"  Removed: {len(df) - len(clean):,} ({(len(df)-len(clean))/len(df)*100:.2f}%)")
    print(f"  Status: Ready for fraud model training ✓")

stripe_transaction_validation()



# ======================================================================
# ## Real-World Example 2: Netflix Content Data Validation
# ======================================================================

def netflix_validation():
    print("NETFLIX: Content Data Validation (1000+ features)\n")
    
    print("1. Challenge:")
    print("   - 7000+ titles in catalog")
    print("   - 1000+ metadata fields per title")
    print("   - Constantly adding new titles (needs immediate metadata)")
    print()
    
    print("2. Common issues caught by validation:")
    print()
    print("   Issue 1: Missing regional metadata")
    print("   - Title available in US but metadata incomplete for India")
    print("   - Result: Recommendation model can't personalize for India")
    print("   - Fix: Require complete metadata for all available regions")
    print()
    
    print("   Issue 2: Corrupted ratings")
    print("   - Old title had rating=9.5 in source system")
    print("   - ETL didn't validate, loaded as-is")
    print("   - Model confused (ratings should be 1-10 decimal accuracy)")
    print("   - Fix: Schema validation (rating in [0, 10], must be numeric)")
    print()
    
    print("   Issue 3: Availability drift")
    print("   - Title marked 'available in 100 countries' but actual availability 92")
    print("   - Recommendation sent to unavailable users (bad experience)")
    print("   - Fix: Validate availability list matches actual availability")
    print()
    
    print("3. Validation approach:")
    print("   - Daily: schema checks (required fields, types)")
    print("   - Daily: range checks (ratings 1-10, budget > 0)")
    print("   - Weekly: sample manual review (1% of titles)")
    print("   - Monthly: deep audit (pick 5 titles, manually verify completeness)")
    print()
    
    print("4. Result:")
    print("   Catches issues before they reach 250M+ users")
    print("   Maintains data quality that enables good recommendations")

netflix_validation()


# ======================================================================
# ## Real-World Example 3: Uber Demand Data Distribution Shift
# ======================================================================

def uber_distribution_shift():
    print("UBER: Demand Prediction - Distribution Shift Detection\n")
    
    print("1. Scenario:")
    print("   - Demand model trained on 2 years of historical data")
    print("   - Predicts: demand by location, hour, day of week")
    print()
    
    print("2. Expected distributions (from training):")
    print("   - Peak hour demand: median = 100 rides/hour")
    print("   - Off-peak demand: median = 30 rides/hour")
    print("   - Weekend multiplier: 1.2x (weekends busier)")
    print()
    
    print("3. Observed in production (last 7 days):")
    print("   - Median demand: 95 rides/hour (slight decrease, expected)")
    print("   - KS test p-value: 0.15 (no significant drift)")
    print("   - Distribution shape looks normal → Continue running")
    print()
    
    print("4. Another week later:")
    print("   - Median demand: 45 rides/hour (huge drop!)")
    print("   - KS test p-value: 0.001 (significant drift)")
    print("   - Weekend multiplier: 1.1x (weekends no longer busier)")
    print()
    
    print("5. Investigation:")
    print("   - Check: did competitor launch? Yes, new competitor")
    print("   - Market share: Uber dropped from 80% to 60%")
    print("   - Real-world change (not data issue)")
    print()
    
    print("6. Action:")
    print("   - Retrain on recent 3 months data")
    print("   - Incorporate competitor data")
    print("   - New model captures correct demand patterns")
    print("   - Accuracy restored to 87%")
    print()
    
    print("Key insight: Distribution shift can be data quality OR real-world change.")
    print("Validation catches both, but action depends on cause.")

uber_distribution_shift()


# ======================================================================
# ## Interview Case Study: Airbnb Listing Data Validation
# ======================================================================

print("CASE STUDY: AIRBNB LISTING DATA VALIDATION")
print()
print("SCENARIO:")
print("  Build price prediction model")
print("  7M+ listings, training data: historical booking data + listing attributes")
print("  Need to validate training data before training")
print()

print("SOLUTION: Multi-layer validation")
print()

print("1. SCHEMA VALIDATION")
print()
print("   Required columns:")
print("   - listing_id (int): unique identifier")
print("   - price (float): nightly rate")
print("   - bedrooms (int): number of bedrooms")
print("   - country (string): country code")
print("   - rating (float): guest rating 1-5")
print()
print("   Check: All columns present? All required fields?")
print("   Result: All present, no missing columns ✓")
print()

print("2. COMPLETENESS VALIDATION")
print()
print("   Threshold: <1% null for critical fields, <5% for optional")
print()
print("   Column nulls:")
print("   - listing_id: 0% ✓")
print("   - price: 0% ✓")
print("   - bedrooms: 0% ✓")
print("   - country: 0% ✓")
print("   - rating: 2% (within 5% threshold) ✓")
print()
print("   Decision: Data is complete enough, proceed")
print()

print("3. RANGE VALIDATION")
print()
print("   Valid ranges:")
print("   - price: (0, $50000) per night")
print("   - bedrooms: (0, 20)")
print("   - rating: [1, 5]")
print()
print("   Issues found:")
print("   - 0.1% listings have price < $0 (data corruption)")
print("   - 0.05% listings have 100+ bedrooms (data entry errors)")
print("   - 0.2% have rating > 5 (system bug from 6 months ago)")
print()
print("   Action: Remove 0.35% of listings (acceptable loss)")
print()

print("4. CATEGORICAL VALIDATION")
print()
print("   Country codes must be valid ISO 3166")
print()
print("   Issues found:")
print("   - 0.05% have invalid country codes (e.g., 'USA' instead of 'US')")
print()
print("   Action: Standardize (map 'USA' → 'US') or remove")
print()

print("5. STATISTICAL VALIDATION")
print()
print("   Distribution baseline (from last 12 months):")
print("   - Price median: $120/night")
print("   - Price p95: $450/night")
print()
print("   Current month distribution:")
print("   - Price median: $118/night (stable)")
print("   - Price p95: $440/night (stable)")
print("   - KS test p-value: 0.23 (no drift) ✓")
print()
print("   Seasonal check:")
print("   - This is December (holiday season)")
print("   - Expected: 20-30% higher prices")
print("   - Observed: 15% higher (slight lower than expected)")
print("   - Action: Note, watch for early booking trend")
print()

print("6. TEMPORAL VALIDATION")
print()
print("   Listing creation dates should be in past")
print()
print("   Issue found:")
print("   - 50 listings have creation_date in future (impossible)")
print("   - Root cause: timezone bug in date logging")
print()
print("   Action: Fix upstream, remove 50 listings")
print()

print("7. RELATIONSHIP VALIDATION")
print()
print("   Logical checks:")
print("   - listing_date <= first_booking_date (listing must exist before booking)")
print("   - avg_rating >= min_rating across all reviews")
print()
print("   Issues: None found")
print()

print("SUMMARY OF VALIDATION")
print()
print("✓ Schema: complete")
print("✓ Completeness: acceptable")
print("⚠ Range: removed 0.35% outliers")
print("⚠ Categories: standardized 0.05% invalid codes")
print("✓ Statistical: no drift")
print("⚠ Temporal: removed 50 future-dated listings")
print("✓ Relationships: valid")
print()
print("Total cleaned: 0.4% of data removed")
print("Final dataset: 6.97M listings (was 7.0M)")
print("Ready for training ✓")
print()

print("STRONG ANSWER:")
print()
print("'I'd validate in layers:")
print()
print("1. Schema: all required columns present, correct types")
print()
print("2. Completeness: nulls <1% critical fields, <5% optional")
print()
print("3. Ranges: price > 0, bedrooms 1-20, rating 1-5")
print()
print("4. Categories: country codes valid ISO standards")
print()
print("5. Statistical: distribution matches baseline, KS test p > 0.05")
print()
print("6. Temporal: dates in valid range, no future dates")
print()
print("7. Relationships: logical consistency (listing before booking)")
print()
print("Result: Clean 99.6% of data for training.'")


# ======================================================================
# ## Key Takeaways
# **Data quality is foundational.** Clean models can't save dirty data. Bad data produces confident wrong predictions.
# **Validation pyramid:**
# ======================================================================
