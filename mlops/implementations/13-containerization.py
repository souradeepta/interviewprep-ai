"""
Auto-generated from 13-containerization.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Containerization & Docker: Packaging ML Systems
# ## Learning Objectives
# - Build Docker images with reproducible environments
# - Optimize image size and build times
# ======================================================================

# ======================================================================
# ## Basic: Dockerfile Structure & Layer Caching
# ======================================================================

# Example Dockerfile for ML model training
dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

# Copy dependencies first (slow to install, change rarely)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code last (changes frequently)
COPY train.py .
COPY data/ data/

ENV SEED=42
ENV MODEL_PATH=/app/model.pkl

ENTRYPOINT ["python", "train.py"]
'''

print("DOCKERFILE BEST PRACTICES")
print()
print(dockerfile_content)
print()
print("Why this order?")
print("1. Base image (Python 3.9-slim): 100MB, never changes")
print("2. Dependencies: 500MB, change occasionally")
print("3. Code: 10MB, changes frequently")
print()
print("Docker caching:")
print("  - If code changes: reuse cached layers 1-2, rebuild layer 3")
print("  - If dependencies change: rebuild layers 2-3")
print()
print("Build times:")
print("  - First build: 5 minutes (install everything)")
print("  - Code change: 10 seconds (reuse cached dependencies)")
print("  - Without optimization: 5 minutes (rebuild everything)")


# ======================================================================
# ## Advanced: Multi-Stage Build for Size Optimization
# ======================================================================

multi_stage = '''
# Stage 1: Builder (large, contains build tools)
FROM python:3.9 as builder

WORKDIR /build
COPY requirements.txt .

# Install + compile wheels
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (small, only runtime dependencies)
FROM python:3.9-slim

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY train.py .

ENTRYPOINT ["python", "train.py"]
'''

print("MULTI-STAGE BUILD")
print()
print(multi_stage)
print()
print("Size comparison:")
print("  - Full image (stage 1): 1.2GB")
print("    - Python 3.9: 900MB")
print("    - Build tools (gcc, make): 300MB")
print("    - Packages: 500MB")
print()
print("  - Multi-stage image (stage 2): 500MB")
print("    - Python 3.9-slim: 100MB")
print("    - Packages: 400MB")
print()
print("  - Reduction: 1.2GB → 500MB (58% smaller!)")
print("  - Faster push/pull to registry")


# ======================================================================
# ## Real-World Examples: Netflix, Stripe, Uber
# ======================================================================

def netflix_containerization():
    """Build optimized Docker image for recommendation model"""

    print("NETFLIX: Docker Container Optimization")
    print("=" * 60)

    print("\nMULTI-STAGE BUILD:")
    print("  Stage 1: Builder (install deps)")
    print("    Size: 800+ MB (includes build tools)")
    print("  Stage 2: Runtime (copy only binaries)")
    print("    Size: 200 MB (final image)")
    print("  Reduction: 75% smaller")

    print("\nLAYER CACHING:")
    print("  Base (python:3.9-slim): 125 MB (cached)")
    print("  System deps: +45 MB (cached)")
    print("  Python libs: +150 MB (cached if no changes)")
    print("  App code: +50 MB (rebuilt on changes)")

    print("\nBUILD TIMES:")
    print("  First build: 5 minutes")
    print("  Incremental (code change): 1 minute (layers cached)")
    print("  Cold build (all changed): 5 minutes")

def stripe_containerization():
    """Build fraud detection model image"""

    print("\nSTRIPE: Fraud Model Containerization")
    print("=" * 60)

    print("\nDOCKERFILE:")
    print("  FROM python:3.9-slim")
    print("  COPY requirements.txt .")
    print("  RUN pip install --no-cache-dir -r requirements.txt")
    print("  COPY src/ /app")
    print("  WORKDIR /app")
    print("  EXPOSE 8000")
    print("  CMD ['python', 'app.py']")

    print("\nIMAGE REGISTRY:")
    print("  stripe-fraud:v3.1.0-sha256abc123")
    print("  Size: 420 MB")
    print("  Build: 2026-05-16")
    print("  Registry: private ECR")

    print("\nDEPLOYMENT:")
    print("  docker run -p 8000:8000 stripe-fraud:v3.1.0")
    print("  Latency: 50ms fraud scoring")
    print("  Throughput: 10K req/sec per instance")

def uber_containerization():
    """Build matching service image"""

    print("\nUBER: Matching Service Containerization")
    print("=" * 60)

    print("\nPERFORMANCE:")
    print("  Cold start: 2 seconds")
    print("  Warm start: <100ms")
    print("  Model load: 1.5 GB (GPU memory)")
    print("  Inference: 30ms per match")

    print("\nHEALTH CHECKS:")
    print("  Readiness: /health → model loaded? (fast)")
    print("  Liveness: /alive → responding? (tcp)")
    print("  Metrics: /metrics → Prometheus")

    print("\nRESOURCE LIMITS:")
    print("  CPU: 4 cores")
    print("  Memory: 8 GB (including model)")
    print("  GPU: 1 × V100 (16GB VRAM)")
    print("  Storage: 50 GB cache")

netflix_containerization()
stripe_containerization()
uber_containerization()



# ======================================================================
# ## Interview Case Study: Containerizing ML Pipeline
# ======================================================================

print("CASE STUDY: CONTAINERIZE FRAUD MODEL TRAINING")
print()
print("Dockerfile:")
print("  FROM python:3.9-slim")
print("  RUN pip install --no-cache-dir pandas scikit-learn xgboost")
print("  COPY train.py /app/")
print("  ENV SEED=42")
print("  ENTRYPOINT ['python', 'train.py']")
print()
print("Benefits:")
print("  ✓ Reproducibility: same dependencies everywhere")
print("  ✓ Portability: runs on laptop, CI/CD, Kubernetes")
print("  ✓ Version control: git commit → docker image version")
print("  ✓ Rollback: previous image available instantly")
print()
print("Workflow:")
print("  1. git commit -m 'improve fraud model'")
print("  2. docker build -t fraud:abc123 .  (tag with commit hash)")
print("  3. docker run fraud:abc123 (train locally, verify)")
print("  4. docker push myregistry/fraud:abc123")
print("  5. kubernetes deploy fraud:abc123")
print("  6. Monitor metrics")
print("  7. If issues: kubernetes deploy fraud:def456 (previous version)")


# ======================================================================
# ## Key Takeaways
# **Docker enables reproducibility:** Same image = same behavior everywhere.
# **Layer caching speeds development:** Code changes rebuild in seconds, not minutes.
# ======================================================================
