"""
Auto-generated from 06-model-versioning.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Model Versioning & Registry: Managing Models in Production
# ## Learning Objectives
# - Implement model registry with versioning and governance
# - Manage model lifecycle: train → staging → production → archive
# ======================================================================

# ======================================================================
# ## Basic Implementation: Simple Model Registry
# ======================================================================

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import json

class ModelStatus(Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ModelRegistry:
    """Simple model registry with versioning"""
    
    def __init__(self):
        self.models = {}  # model_name -> [versions]
        self.production = {}  # model_name -> current production version
    
    def register_model(self, model_name: str, version: str, 
                      accuracy: float, precision: float, recall: float,
                      model_path: str, data_version: str, code_commit: str) -> Dict:
        """Register a trained model"""
        model_info = {
            'name': model_name,
            'version': version,
            'status': ModelStatus.STAGING.value,
            'created_at': datetime.now().isoformat(),
            'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall},
            'model_path': model_path,
            'data_version': data_version,
            'code_commit': code_commit,
            'approved_by': None,
            'deployed_at': None,
        }
        
        if model_name not in self.models:
            self.models[model_name] = []
        
        self.models[model_name].append(model_info)
        return model_info
    
    def approve_model(self, model_name: str, version: str, approved_by: str):
        """Approve model for production deployment"""
        model = self.get_model(model_name, version)
        if model:
            model['approved_by'] = approved_by
            model['status'] = ModelStatus.STAGING.value  # Ready for deployment
            return True
        return False
    
    def deploy_model(self, model_name: str, version: str) -> bool:
        """Deploy approved model to production"""
        model = self.get_model(model_name, version)
        if model and model['approved_by']:
            model['status'] = ModelStatus.PRODUCTION.value
            model['deployed_at'] = datetime.now().isoformat()
            self.production[model_name] = version
            return True
        return False
    
    def get_model(self, model_name: str, version: str) -> Optional[Dict]:
        """Get specific model version"""
        if model_name in self.models:
            for m in self.models[model_name]:
                if m['version'] == version:
                    return m
        return None
    
    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Get current production version"""
        if model_name in self.production:
            version = self.production[model_name]
            return self.get_model(model_name, version)
        return None
    
    def rollback(self, model_name: str) -> bool:
        """Rollback to previous version"""
        if model_name in self.models and len(self.models[model_name]) > 1:
            versions = self.models[model_name]
            # Find most recent non-current production version
            current_version = self.production.get(model_name)
            for model in reversed(versions[:-1]):
                if model['version'] != current_version:
                    self.deploy_model(model_name, model['version'])
                    return True
        return False

# Usage
registry = ModelRegistry()

# Register models
registry.register_model(
    'fraud_detector', 'v1',
    accuracy=0.94, precision=0.97, recall=0.90,
    model_path='s3://models/fraud_v1.pkl',
    data_version='fraud_training_v3',
    code_commit='abc123'
)

registry.register_model(
    'fraud_detector', 'v2',
    accuracy=0.96, precision=0.98, recall=0.92,
    model_path='s3://models/fraud_v2.pkl',
    data_version='fraud_training_v4',
    code_commit='def456'
)

# Approve and deploy v2
registry.approve_model('fraud_detector', 'v2', 'data_scientist_alice')
registry.deploy_model('fraud_detector', 'v2')

# Check production
prod = registry.get_production_model('fraud_detector')
print(f"✓ Production model: {prod['name']} {prod['version']}")
print(f"  Accuracy: {prod['metrics']['accuracy']}")
print(f"  Deployed by: {prod['approved_by']}")


# ======================================================================
# ## Advanced Implementation: Registry with Governance
# ======================================================================

from dataclasses import dataclass
from typing import List

@dataclass
class AuditLog:
    """Track all actions on models"""
    timestamp: str
    action: str
    model: str
    version: str
    actor: str
    details: str

class GovernedModelRegistry:
    """Production registry with governance and audit trail"""
    
    def __init__(self):
        self.models = {}
        self.audit_log = []
        self.approvers = {}  # model -> list of approved versions
    
    def _log_action(self, action: str, model: str, version: str, actor: str, details: str):
        """Log all actions for compliance"""
        log = AuditLog(
            timestamp=datetime.now().isoformat(),
            action=action,
            model=model,
            version=version,
            actor=actor,
            details=details
        )
        self.audit_log.append(log)
    
    def register_model(self, model_name: str, version: str,
                      metrics: Dict, data_version: str, code_commit: str,
                      owner: str) -> Dict:
        """Register model with metadata"""
        if model_name not in self.models:
            self.models[model_name] = []
        
        model_info = {
            'version': version,
            'status': 'staging',
            'metrics': metrics,
            'data_version': data_version,
            'code_commit': code_commit,
            'owner': owner,
            'registered_at': datetime.now().isoformat(),
            'approvals': [],  # List of approvers
        }
        
        self.models[model_name].append(model_info)
        self._log_action('register', model_name, version, owner, f'Registered {version}')
        return model_info
    
    def request_approval(self, model_name: str, version: str,
                        requester: str, reason: str) -> bool:
        """Request approval from governance committee"""
        model = self._get_model(model_name, version)
        if model:
            self._log_action('approval_requested', model_name, version, requester, reason)
            return True
        return False
    
    def approve_model(self, model_name: str, version: str,
                     approver: str, notes: str) -> bool:
        """Approve model for production"""
        model = self._get_model(model_name, version)
        if model:
            model['approvals'].append({'approver': approver, 'timestamp': datetime.now().isoformat()})
            # Require 2 approvals for production
            if len(model['approvals']) >= 2:
                model['status'] = 'approved'
            self._log_action('approved', model_name, version, approver, notes)
            return True
        return False
    
    def deploy(self, model_name: str, version: str, deployer: str) -> bool:
        """Deploy approved model to production"""
        model = self._get_model(model_name, version)
        if model and model['status'] == 'approved':
            # Archive previous production
            for m in self.models[model_name]:
                if m['status'] == 'production':
                    m['status'] = 'archived'
                    self._log_action('archived', model_name, m['version'], deployer, 'Previous production archived')
            
            model['status'] = 'production'
            self._log_action('deployed', model_name, version, deployer, 'Deployed to production')
            return True
        return False
    
    def get_audit_trail(self, model_name: str, version: str) -> List[AuditLog]:
        """Get audit trail for compliance"""
        return [log for log in self.audit_log 
                if log.model == model_name and log.version == version]
    
    def _get_model(self, model_name: str, version: str) -> Optional[Dict]:
        if model_name in self.models:
            for m in self.models[model_name]:
                if m['version'] == version:
                    return m
        return None

# Usage
reg = GovernedModelRegistry()

# Register
reg.register_model('payment_fraud', 'v5',
                  metrics={'accuracy': 0.95, 'precision': 0.98},
                  data_version='fraud_training_v10',
                  code_commit='xyz789',
                  owner='fraud_team')

# Request approval
reg.request_approval('payment_fraud', 'v5', 'alice', 'Better accuracy on holdout set')

# Approve (need 2 approvals)
reg.approve_model('payment_fraud', 'v5', 'bob', 'Looks good')
reg.approve_model('payment_fraud', 'v5', 'carol', 'Approved')

# Deploy
reg.deploy('payment_fraud', 'v5', 'alice')

# Audit trail
trail = reg.get_audit_trail('payment_fraud', 'v5')
print(f"✓ Audit trail for payment_fraud v5:")
for log in trail:
    print(f"  {log.timestamp}: {log.action} by {log.actor}")


# ======================================================================
# ## Real-World Example 1: Netflix Model Registry
# ======================================================================

import pandas as pd

def netflix_model_registry():
    """Manage model versions across production, staging, etc"""

    print("NETFLIX: Model Registry & Lifecycle")
    print("=" * 60)

    models = pd.DataFrame({
        'model_id': ['rec_v1', 'rec_v2', 'rec_v3', 'rec_v4'],
        'status': ['archived', 'staging', 'production', 'development'],
        'accuracy': [0.88, 0.91, 0.92, 0.93],
        'trained': ['2026-02-01', '2026-03-15', '2026-04-20', '2026-05-10'],
        'traffic': ['0%', '5%', '90%', '0%'],
    })

    print("\nMODEL VERSIONS:")
    print(models.to_string(index=False))

    print("\nAPPROVAL WORKFLOW:")
    print("  1. Development: rec_v4 trains on latest data, accuracy=0.93")
    print("  2. Staging: rec_v4 deployed to staging cluster")
    print("  3. Validation: A/B test on 5% traffic (rec_v2)")
    print("  4. Production: If metrics improve, promote to 100% traffic")
    print("  5. Archive: Old versions kept for 30 days (rollback safety)")

    print("\nPROMOTION RULES:")
    print("  Development → Staging: Any new model version")
    print("  Staging → Production: accuracy >= prev + 0.5%, latency < 50ms")
    print("  Production: auto-rollback if metrics degrade >1%")

    print("\nROLLBACK PROCEDURE:")
    print("  Detected issue: accuracy drops to 0.90")
    print("  Action: Revert to production model rec_v3 (0.92)")
    print("  Time to rollback: <5 minutes")
    print("  Traffic impact: zero (instant switch)")

netflix_model_registry()



# ======================================================================
# ## Real-World Example 2: Stripe Fraud Model Registry
# ======================================================================

import pandas as pd

def stripe_approval_gates():
    """Model approval workflow with quality gates"""

    print("STRIPE: Fraud Model Approval Gates")
    print("=" * 60)

    checks = pd.DataFrame({
        'check': [
            'Code review',
            'Unit tests',
            'Integration tests',
            'Accuracy (test set)',
            'Fairness (demographic parity)',
            'Latency (<50ms)',
            'Rollback plan',
        ],
        'required': [True, True, True, True, True, True, True],
        'status': ['✓ Pass', '✓ Pass', '✓ Pass', '✓ Pass', '⚠ Warning', '✓ Pass', '✓ Pass'],
        'details': [
            '2 approvals received',
            '95 tests, 0 failures',
            '1000 test cases, all pass',
            'Accuracy 94.5% vs 93.2% baseline',
            'Precision varies 2% across demographics',
            'p99 latency: 35ms',
            'Rollback to v3 tested and working'
        ]
    })

    print("\nAPPROVAL CHECKLIST:")
    for _, row in checks.iterrows():
        status_symbol = '✓' if '✓' in row['status'] else '⚠'
        print(f"  {status_symbol} {row['check']:30s} {row['status']}")
        print(f"    → {row['details']}")

    print("\nDECISION:")
    warnings = checks[~checks['status'].str.contains('✓')].shape[0]
    if warnings == 0:
        print("  ✓ All checks passed - approve for staging")
    else:
        print(f"  ⚠ {warnings} warning(s) - investigate before promotion")

stripe_approval_gates()



# ======================================================================
# ## Real-World Example 3: Uber Multi-Model Registry
# ======================================================================

def uber_model_registry():
    print("Uber: Multi-Team Model Registry")
    print()
    print("1. Organization:")
    print("   - 3 domains: pricing, matching, ETA")
    print("   - Each domain has 10+ models")
    print("   - Central registry for all models")
    print()
    print("2. Team Access:")
    print("   - Pricing team: owns pricing models, can deploy")
    print("   - Matching team: owns matching models, can deploy")
    print("   - Cross-team: can see all models, read-only")
    print()
    print("3. Shared Infrastructure:")
    print("   - Central dashboard: see all models, search by metric")
    print("   - Deployment: one-click deploy across all regions")
    print("   - Rollback: one-click revert to previous version")
    print()
    print("4. Deployment Strategy:")
    print("   - Rolling: deploy to one region, then next")
    print("   - Canary: 5% traffic, monitor, 50%, 100%")
    print("   - Shadow: run new model in parallel, don't use predictions")
    print()
    print("5. Results:")
    print("   - 2-3% metric improvement per model")
    print("   - Deploy weekly across all models")
    print("   - Rollback in minutes if issues")

uber_model_registry()


# ======================================================================
# ## Interview Case Study: Registry for 50 Models Across 5 Teams
# ======================================================================

print("INTERVIEW CASE STUDY: MULTI-TEAM MODEL REGISTRY")
print()
print("SCENARIO:")
print("  Company has 5 ML teams, 50 models in production")
print("  Need: centralized registry, governance, easy deployment")
print()

print("DESIGN:")
print()
print("1. REGISTRY STRUCTURE:")
print("   Model: fraud_detector")
print("   ├─ v1 (archived): accuracy=0.93")
print("   ├─ v2 (production): accuracy=0.95")
print("   └─ v3 (staging): accuracy=0.96 (awaiting approval)")
print()

print("2. METADATA PER VERSION:")
print("   - Metrics: accuracy, precision, latency")
print("   - Code: git commit, branch")
print("   - Data: training set version, date")
print("   - Dependencies: library versions (pinned)")
print("   - Owner: which team owns this")
print()

print("3. GOVERNANCE WORKFLOW:")
print("   Register → Staging → Approval → Canary → Production")
print("   - Approval: 2-factor (code + business)")
print("   - Audit: log all actions with timestamps")
print()

print("4. DEPLOYMENT:")
print("   - Canary: 5% traffic for 24h")
print("   - Monitor: automated metrics check")
print("   - Decision: if metrics better, gradually increase traffic")
print("   - Rollback: auto if metrics degrade by >2%")
print()

print("5. SEARCH & DISCOVERY:")
print("   - Find: all models with accuracy >0.90")
print("   - Find: all fraud models owned by team_a")
print("   - Find: all models trained on 2026-05 data")
print()

print("STRONG ANSWER:")
print("  'Central registry with rich metadata (metrics, code, data, owner).")
print("  Governance: 2-approval workflow, audit trail, compliance.")
print("  Deployment: canary → monitor → rollback if needed.")
print("  Access: teams own their models, can deploy; others read-only.")
print("  Search: filter by metric, owner, date. Fast discovery.")
print("  Results: enables 50 teams, 50 models, fast iteration.'")


# ======================================================================
# ## Key Takeaways
# **2-Minute Elevator Pitch:**
# "Model registry centralizes model management: versioning, governance, deployment, and audit trails. Enables teams to safely deploy models to production, rollback if metrics degrade, and comply with regulations. Essential for managing 10+ models in production."
# ======================================================================
