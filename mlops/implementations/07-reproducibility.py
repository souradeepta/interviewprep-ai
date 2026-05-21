"""
Auto-generated from 07-reproducibility.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Reproducibility: Exact Replication of ML Results
# ## Learning Objectives
# - Control randomness in ML training
# - Version code, data, environment, and dependencies
# ======================================================================

# ======================================================================
# ## Basic Implementation: Seed Control
# ======================================================================

import numpy as np
import random
from typing import Any

def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # If using PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Demonstrate randomness with/without seed
print("Without seed control:")
random_vals_1 = [random.random() for _ in range(3)]
random_vals_2 = [random.random() for _ in range(3)]
print(f"  Run 1: {random_vals_1}")
print(f"  Run 2: {random_vals_2}")
print(f"  Different? {random_vals_1 != random_vals_2} ✗")

print("\nWith seed control:")
set_seed(42)
random_vals_1 = [random.random() for _ in range(3)]
set_seed(42)
random_vals_2 = [random.random() for _ in range(3)]
print(f"  Run 1: {random_vals_1}")
print(f"  Run 2: {random_vals_2}")
print(f"  Same? {random_vals_1 == random_vals_2} ✓")


# ======================================================================
# ## Advanced Implementation: Complete Reproducibility
# ======================================================================

import hashlib
from pathlib import Path
from typing import Dict, Tuple
import json

class ReproducibilityManager:
    """Manages all reproducibility components"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.metadata = {}
    
    def set_seed(self):
        """Set all random seeds"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.metadata['seed'] = self.seed
    
    def get_code_version(self) -> Dict[str, str]:
        """Get git commit hash (code version)"""
        # In real scenario: git rev-parse HEAD
        return {
            'commit': 'abc123def456',
            'branch': 'main',
            'dirty': False,  # uncommitted changes
        }
    
    def get_data_version(self, data_path: str) -> str:
        """Get hash of dataset (data version)"""
        # In real scenario: compute SHA256 of data file
        return 'data_hash_abc123'
    
    def get_environment(self) -> Dict[str, str]:
        """Get environment details"""
        import sys
        return {
            'python_version': sys.version.split()[0],
            'platform': 'linux',
        }
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get pinned library versions"""
        return {
            'numpy': '1.24.0',
            'pandas': '1.5.3',
            'torch': '2.0.0',
            'transformers': '4.30.0',
        }
    
    def capture_metadata(self, data_path: str, hyperparams: Dict) -> Dict:
        """Capture all reproducibility information"""
        self.metadata = {
            'timestamp': '2026-05-17T10:30:00Z',
            'seed': self.seed,
            'code': self.get_code_version(),
            'data': {'path': data_path, 'hash': self.get_data_version(data_path)},
            'environment': self.get_environment(),
            'dependencies': self.get_dependencies(),
            'hyperparameters': hyperparams,
        }
        return self.metadata
    
    def save_metadata(self, output_path: str):
        """Save metadata for future reproduction"""
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def validate_environment(self, saved_metadata: Dict) -> bool:
        """Check if current environment matches saved metadata"""
        current_deps = self.get_dependencies()
        saved_deps = saved_metadata.get('dependencies', {})
        
        mismatches = []
        for lib, version in saved_deps.items():
            if current_deps.get(lib) != version:
                mismatches.append(f"{lib}: {current_deps.get(lib)} vs {version}")
        
        if mismatches:
            print("⚠ Environment mismatch:")
            for m in mismatches:
                print(f"  {m}")
            return False
        return True

# Usage
repro = ReproducibilityManager(seed=42)

# Capture metadata
metadata = repro.capture_metadata(
    data_path='data/fraud_training_v3.parquet',
    hyperparams={'learning_rate': 0.01, 'batch_size': 64}
)

print("✓ Reproducibility metadata captured:")
print(f"  Seed: {metadata['seed']}")
print(f"  Code: {metadata['code']['commit']}")
print(f"  Data: {metadata['data']['hash']}")
print(f"  Python: {metadata['environment']['python_version']}")
print(f"  Libraries: {list(metadata['dependencies'].keys())}")


# ======================================================================
# ## Real-World Example 1: Debugging Accuracy Differences
# ======================================================================

import numpy as np

def netflix_reproducibility():
    """Ensure exact model reproduction"""

    print("NETFLIX: Model Reproducibility Framework")
    print("=" * 60)

    # Capture all reproduction metadata
    metadata = {
        'model_id': 'ranking_v5',
        'trained': '2026-05-16T10:30:00Z',
        'reproducibility': {
            'numpy_seed': 42,
            'python_seed': 42,
            'tf_seed': 42,
            'torch_seed': 42,
            'cuda_deterministic': True,
        },
        'environment': {
            'python_version': '3.9.12',
            'tensorflow': '2.11.0',
            'torch': '2.0.0',
            'numpy': '1.23.5',
            'pandas': '1.5.1',
        },
        'data': {
            'training_set': 'netflix_events_v2.0',
            'version': 'snapshot_2026-05-15',
            'size': '500GB',
            'rows': '10B events',
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 10,
            'early_stopping': 'patience=2',
            'optimizer': 'adam',
        },
        'git': {
            'commit': 'abc123def456',
            'branch': 'main',
            'tag': 'v5.0.0',
        }
    }

    print("\nREPRODUCIBILITY METADATA:")
    for section, details in metadata.items():
        if isinstance(details, dict):
            print(f"\n{section.upper()}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {details}")

    print("\n\nTO REPRODUCE:")
    print("  1. git checkout abc123def456")
    print("  2. pip install -r requirements-v5.0.0.txt")
    print("  3. python train.py --seed 42 --data snapshot_2026-05-15")
    print("  4. Expected: accuracy = 0.92345 (exact)")

    print("\n✓ Reproducible with metadata snapshot")

netflix_reproducibility()



# ======================================================================
# ## Real-World Example 2: Environment Reproducibility
# ======================================================================

def environment_reproducibility():
    print("Environment Reproducibility")
    print()
    print("Problem: Code works on laptop, fails on server")
    print()
    
    print("Causes:")
    print("  1. Python version: 3.9 vs 3.8 (API differences)")
    print("  2. Library versions: PyTorch 1.10 vs 2.0 (breaking changes)")
    print("  3. CUDA version: 11.0 vs 12.0 (GPU rounding differences)")
    print("  4. System libraries: different Linux distros")
    print()
    
    print("Solution: Docker containerization")
    print()
    print("Dockerfile:")
    print("  FROM python:3.9.0")
    print("  RUN pip install torch==2.0.0 transformers==4.30.0")
    print("  COPY train.py /app/")
    print("  CMD python train.py --seed 42")
    print()
    
    print("Usage:")
    print("  # Build container with pinned versions")
    print("  docker build -t model:v1 .")
    print()
    print("  # Run on laptop")
    print("  docker run model:v1")
    print()
    print("  # Run on server (same container, same results)")
    print("  docker run model:v1")
    print()
    print("Result: Same results across laptop, server, CI/CD ✓")

environment_reproducibility()


# ======================================================================
# ## Real-World Example 3: Netflix Reproducibility at Scale
# ======================================================================

def netflix_reproducibility():
    print("Netflix: Reproducibility for 100+ Models")
    print()
    print("Challenge: 100+ models, each trained by different team.")
    print("Need: every model reproducible after 1 year.")
    print()
    
    print("Solution:")
    print()
    print("1. Version everything:")
    print("   - Code: git commit stored with model")
    print("   - Data: DVC or data hashes stored")
    print("   - Dependencies: requirements.txt pinned")
    print("   - Seed: captured in metadata")
    print()
    
    print("2. Containerize:")
    print("   - Dockerfile per training pipeline")
    print("   - Pin Python, system libraries, GPU drivers")
    print()
    
    print("3. Verify reproducibility:")
    print("   - Monthly: retrain old models from 1 year ago")
    print("   - Compare: accuracy should match (within 0.1%)")
    print("   - Alert: if reproducibility fails")
    print()
    
    print("4. Document:")
    print("   - README: 'To reproduce: git checkout X, docker build, ...")
    print("   - Metadata: stored with each model")
    print()
    
    print("Result: Any model from any date can be retrained identically ✓")

netflix_reproducibility()


# ======================================================================
# ## Interview Case Study: Making Pipeline Reproducible
# ======================================================================

print("CASE STUDY: REPRODUCIBLE ML PIPELINE")
print()
print("SCENARIO:")
print("  Model achieved 95% accuracy 3 months ago.")
print("  Today: need to reproduce it. How?")
print()

print("SOLUTION:")
print()
print("1. Fetch metadata (saved with model):")
print("   {")
print("     'code': {'commit': 'abc123', 'branch': 'main'},")
print("     'data': {'version': 'fraud_training_v10', 'hash': 'def456'},")
print("     'seed': 42,")
print("     'dependencies': {")
print("       'python': '3.9.0',")
print("       'torch': '2.0.0',")
print("       'numpy': '1.24.0'")
print("     }")
print("   }")
print()

print("2. Step-by-step reproduction:")
print("   a) Fetch code:")
print("      git checkout abc123")
print()
print("   b) Setup environment:")
print("      docker run -v /data:/data my_training_image:2.0.0")
print()
print("   c) Load data:")
print("      Load fraud_training_v10, verify hash = def456")
print()
print("   d) Run training:")
print("      python train.py --seed 42 --data fraud_training_v10")
print()

print("3. Verify:")
print("   accuracy = 95.0% ✓ (matches original, within floating-point tolerance)")
print()

print("STRONG ANSWER:")
print("  'Store metadata with every model: code commit, data version, seed,")
print("  library versions. To reproduce: fetch metadata, git checkout code,")
print("  docker run with pinned image, load data using hash, train with seed.")
print("  Verify: accuracy matches original (within 0.1%).")
print("  Enables: debugging accuracy changes, compliance, model audits.'")


# ======================================================================
# ## Key Takeaways
# **Reproducibility Components:**
# 1. Code: git commit hash
# 2. Data: content hash, exact version
# ======================================================================
