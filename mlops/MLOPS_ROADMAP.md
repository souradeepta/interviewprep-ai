# ML Ops Interview Prep Roadmap

Choose your learning path based on your role and time availability.

## Quick Reference

| Path | Duration | Focus | Best For |
|------|----------|-------|----------|
| **System Design** | 1-2 weeks | Concepts 13-24 | ML System Design interviews |
| **Full Stack** | 3-4 weeks | Concepts 1-24 | Senior ML Engineer roles |
| **Data Engineer** | 1-2 weeks | Concepts 1-4, 19-20, 22-24 | Data Engineer roles |
| **ML Engineer** | 1-2 weeks | Concepts 5-12, 13-16 | ML Engineer roles |
| **Quick Prep** | 3-5 days | Concepts 13-20 | Urgent interview prep |

---

## Path 1: System Design Interview (1-2 weeks)

**Target:** ML System Design rounds at FAANG companies  
**Focus:** Designing systems at scale (latency, cost, reliability)  
**Prerequisites:** Basic understanding of ML training and inference  

### Study Order (2 hours per concept)

**Week 1: Serving & Deployment**
- Concept 13: Containerization & Docker (2h)
- Concept 14: Model Serving Frameworks (2h)
- Concept 16: Deployment Strategies (2h)
- Concept 15: Model Registry & CI/CD (2h)

**Week 2: Monitoring, Reliability & Governance**
- Concept 17: Model Monitoring and Alert Design (2h)
- Concept 18: Data and Concept Drift Detection (2h)
- Concept 19: Pipeline Orchestration and Backfills (2h)
- Concept 20: Online/Offline Feature Parity (2h)
- Concept 21: ML Incident Response and Rollback (2h)
- Concept 22: ML Governance, Lineage and Auditability (2h)
- Concept 23: ML Security and Privacy Controls (2h)
- Concept 24: Cost, Capacity and Reliability Planning (2h)

### Practice & Interview Prep

**Case Studies:** 5-10 of the most common system design scenarios
1. Stripe: Fraud detection at 1M events/day
2. Netflix: Feature store for 1B events/day
3. Uber: Model monitoring across 100+ models
4. Google: ML Ops infrastructure for 1000s of models
5. Amazon: Safe model deployment with canary rollouts

**Interview Questions:** 30-40 system design questions
- Practice with follow-ups
- Time yourself (20-30 minutes per question)
- Explain your design out loud

**Success Criteria:**
- ✅ Can design a model serving system for 1M requests/sec
- ✅ Understand latency vs cost trade-offs
- ✅ Know deployment strategies and when to use each
- ✅ Can explain monitoring strategy for production models
- ✅ Familiar with Kubernetes/Airflow basics

---

## Path 2: Full ML Engineering (2-3 weeks)

**Target:** Senior ML Engineer or ML Systems Engineer roles  
**Focus:** Complete ML Ops lifecycle from data to production  
**Prerequisites:** Python, basic ML knowledge  

### Study Order (2 hours per concept, 24 concepts total)

**Week 1: Data & Feature Engineering**
- Concept 01: Data Pipelines (2h)
- Concept 02: Feature Stores (2h)
- Concept 03: Data Validation (2h)
- Concept 04: Data Versioning (2h)

**Week 2: Model Development**
- Concept 05: Experiment Tracking (2h)
- Concept 06: Model Versioning & Registry (2h)
- Concept 07: Reproducibility (2h)
- Concept 08: Hyperparameter Optimization (2h)

**Week 3: Testing & Deployment**
- Concept 09: Model Testing (2h)
- Concept 10: Data Testing (2h)
- Concept 11: A/B Testing & Experimentation (2h)
- Concept 12: Evaluation Metrics (2h)
- Concept 13: Containerization (2h)
- Concept 14: Model Serving (2h)

**Week 4: Production Reliability**
- Concept 15: Model Registry & CI/CD (2h)
- Concept 16: Deployment Strategies (2h)
- Concept 17: Monitoring and Alert Design (2h)
- Concept 18: Data and Concept Drift (2h)
- Concept 21: Incident Response and Rollback (2h)
- Concept 20: Online/Offline Feature Parity (2h)

**Week 5: Governance and Scale (optional depth)**
- Concept 19: Pipeline Orchestration and Backfills (2h)
- Concept 22: Governance, Lineage and Auditability (2h)
- Concept 23: Security and Privacy Controls (2h)
- Concept 24: Cost, Capacity and Reliability Planning (2h)

**Total Study Time:** 48 hours for concepts 1–24 at two hours each.

### Practice & Interview Prep

**Case Studies:** All 8+ case studies
1. Stripe: Fraud detection
2. Netflix: Feature pipeline
3. Uber: Model monitoring
4. Google: ML Ops at scale
5. Meta: Experiment tracking
6. Amazon: Safe deployment
7. Tesla: Continuous training
8. OpenAI: Multi-modal serving

**Interview Questions:** All 150+ questions
- Practice 5-10 per day
- Group by concept first, then mix them
- Do mock interviews with other candidates

**Coding Practice:** Implement small versions of:
- Experiment tracking system
- Feature store
- Model serving API
- Monitoring system

**Success Criteria:**
- ✅ Understand full ML Ops lifecycle
- ✅ Can design and implement production systems
- ✅ Know trade-offs across all stages
- ✅ Can answer 90%+ of interview questions
- ✅ Can teach these concepts to others

---

## Path 3: Data Engineer (1-2 weeks)

**Target:** Data Engineer roles working on ML infrastructure  
**Focus:** Data pipelines, feature engineering, infrastructure  
**Prerequisites:** Python, SQL, distributed systems basics  

### Study Order (2 hours per concept)

**Week 1: Data Fundamentals**
- Concept 01: Data Pipelines (2h)
- Concept 02: Feature Stores (2h)
- Concept 03: Data Validation (2h)
- Concept 04: Data Versioning (2h)

**Week 2: Reliable Data and Operations**
- Concept 19: Pipeline Orchestration and Backfills (2h)
- Concept 20: Online/Offline Feature Parity (2h)
- Concept 22: Governance, Lineage and Auditability (2h)
- Concept 23: Security and Privacy Controls (2h)
- Concept 24: Cost, Capacity and Reliability Planning (2h)

### Practice & Interview Prep

**Case Studies:** Focus on data-heavy scenarios
1. Netflix: Feature pipeline for 1B events/day
2. Stripe: Data pipeline for fraud detection
3. Uber: Real-time feature computation

**Interview Questions:** 20-30 focused on data pipelines and infrastructure
- How to design pipeline for X throughput
- How to handle failures and retries
- Cost optimization for data processing

**Coding Practice:**
- Implement simple Airflow DAG
- Build feature store prototype
- Create data validation pipeline

**Success Criteria:**
- ✅ Can design data pipelines at scale
- ✅ Understand feature store concepts
- ✅ Know Airflow/Kubernetes basics
- ✅ Can discuss cost and latency trade-offs
- ✅ Comfortable with distributed systems

---

## Path 4: ML Engineer (1-2 weeks)

**Target:** ML Engineer roles focused on models and training  
**Focus:** Model development, serving, monitoring  
**Prerequisites:** Python, ML algorithms, TensorFlow/PyTorch  

### Study Order (2 hours per concept)

**Week 1: Model Development**
- Concept 05: Experiment Tracking (2h)
- Concept 06: Model Versioning (2h)
- Concept 07: Reproducibility (2h)
- Concept 08: Hyperparameter Optimization (2h)

**Week 2: Testing & Production**
- Concept 09: Model Testing (2h)
- Concept 11: A/B Testing (2h)
- Concept 12: Evaluation Metrics (2h)
- Concept 13: Containerization (2h)

**Week 3: Serving & Monitoring**
- Concept 14: Model Serving (2h)
- Concept 17: Monitoring and Alert Design (2h)
- Concept 18: Data and Concept Drift (2h)
- Concept 20: Online/Offline Feature Parity (2h)
- Concept 21: Incident Response and Rollback (2h)

### Practice & Interview Prep

**Case Studies:** Model-focused scenarios
1. Stripe: Fraud detection model
2. Netflix: Recommendation ranking
3. Uber: ETA prediction at scale

**Interview Questions:** 20-30 focused on models, serving, monitoring
- How to train and serve models at scale
- Detect and respond to model degradation
- Handle model versioning and rollback

**Coding Practice:**
- Implement experiment tracking system
- Build model serving API
- Create monitoring/alerting system

**Success Criteria:**
- ✅ Understand full model lifecycle
- ✅ Know how to monitor model performance
- ✅ Comfortable with serving frameworks
- ✅ Can design reliable inference systems
- ✅ Understand production constraints

---

## Path 5: Quick Prep (3-5 days)

**Target:** Last-minute interview prep  
**Focus:** Most common system design concepts  
**Prerequisites:** Basic ML knowledge  

### Minimum Study (4-6 hours)

**Priority 1 (Essential, 3 hours):**
- Concept 13: Containerization (1h)
- Concept 14: Model Serving (1h)
- Concept 16: Deployment Strategies (1h)

**Priority 2 (Highly Important, 2-3 hours):**
- Concept 17: Monitoring and Alert Design (1h)
- Concept 18: Drift Detection (1h)
- Concept 01: Data Pipelines (1h)

### Practice (2-3 hours)

- Study 2 case studies (Stripe fraud, Netflix features)
- Practice 10-15 interview questions with follow-ups
- Do 1-2 mock interviews

### Success Criteria

- ✅ Can discuss basic system design concepts
- ✅ Know key trade-offs (latency, cost, reliability)
- ✅ Familiar with common tools and patterns
- ✅ Can handle follow-up questions

---

## Study Tips

### Maximize Retention
1. **Read actively:** Take notes, don't just read
2. **Code alongside:** Type out code examples yourself
3. **Explain out loud:** Teach concepts to yourself or others
4. **Make flashcards:** Key concepts and trade-offs
5. **Repeat:** Review 1-2 concepts daily from previous weeks

### Prepare for Interviews
1. **Practice write:** Write out your answers, then verbalize
2. **Time yourself:** 20-30 minutes per system design question
3. **Get feedback:** Have someone critique your approach
4. **Cover edge cases:** Think about failure modes
5. **Know your tools:** Be able to name 2-3 options for each role

### Stay Motivated
1. **Progress tracking:** Mark concepts as you complete them
2. **Mix topics:** Don't study same topic 2 days in a row
3. **Practice with others:** Pair interview prep with a friend
4. **Real-world connection:** Follow companies' ML Ops blogs
5. **Celebrate wins:** Review case studies you now understand

---

## Interview Success Checklist

Before your interview:
- [ ] All concepts in your path studied and reviewed
- [ ] All case studies read and understood
- [ ] 20+ interview questions practiced
- [ ] 3-5 mock interviews completed
- [ ] Can explain each concept in 2 min and 10 min versions
- [ ] Familiar with 2-3 tools in your domain
- [ ] Know common trade-offs and pitfalls
- [ ] Can discuss real-world examples

---

## After the Interview

1. **Document what you learned:** Add new questions if you got stuck
2. **Review case studies:** Did you follow the pattern of strong answers?
3. **Update your understanding:** Were there surprises?
4. **Help others:** Share what you learned with your study group

---

**Get started:** Choose your path above and begin with the first concept!
