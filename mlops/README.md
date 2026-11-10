# ML Ops for Interview Prep & Learning

Comprehensive ML Ops learning materials for FAANG interview preparation and practical implementation.

## What's Inside

- **24 Core ML Ops Concepts** — Data pipelines, model development, deployment, monitoring, governance, security, and reliability
- **25 Jupyter Notebooks** — One concept map and one notebook for each numbered concept
- **62 Interview Questions** — Three banks covering the original concepts and production decisions
- **2 Case Studies** — Full interview walkthroughs with strong/weak answer patterns
- **Interview Prep Learning Paths** — 1-3 week focused learning for different roles

## Organization

The numbered concept track contains 24 topics organized by ML lifecycle stage:
1. **Data & Feature Engineering** (1-4) — Pipelines, feature stores, validation, versioning
2. **Model Development** (5-8) — Experiment tracking, versioning, reproducibility, optimization
3. **Model Evaluation & Testing** (9-12) — Model testing, data testing, A/B testing, metrics
4. **Deployment & Serving** (13-16) — Containerization, serving, registry, deployment strategies

5. **Production Operations** (17-24) — [Monitoring](concepts/17-monitoring-alert-design.md), [drift detection](concepts/18-drift-detection.md), [pipeline orchestration and backfills](concepts/19-pipeline-orchestration-backfills.md), [feature parity](concepts/20-feature-parity.md), [incident response](concepts/21-incident-response-rollback.md), [governance](concepts/22-governance-lineage-auditability.md), [security and privacy](concepts/23-security-privacy-controls.md), and [cost and reliability](concepts/24-cost-capacity-reliability.md).

## Quick Start

**Choose your learning path:**
- **System Design Interview** (1-2 weeks) → See [MLOPS_ROADMAP.md](MLOPS_ROADMAP.md) Path 1
- **Full ML Engineering** (2-3 weeks) → See [MLOPS_ROADMAP.md](MLOPS_ROADMAP.md) Path 2
- **Data Engineer** (1-2 weeks) → See [MLOPS_ROADMAP.md](MLOPS_ROADMAP.md) Path 3
- **ML Engineer** (1-2 weeks) → See [MLOPS_ROADMAP.md](MLOPS_ROADMAP.md) Path 4

**Start with concepts 1-4 (Data & Feature Engineering):**
- Read: `concepts/01-data-pipelines.md`
- Practice: `notebooks/01-data-pipelines.ipynb`
- Interview prep: `interview-questions/questions.json` (filter by concept)
- Structured practice: [MLOps Interview Questions](interview-prep/mlops-interview-questions.md)

## Repository Structure

```
mlops/
├── concepts/              # 24 markdown files with theory and interview prep
├── notebooks/             # 25 Jupyter notebooks with production code examples
├── interview-questions/   # FAANG interview questions organized by concept
├── interview-prep/        # Scenario, debugging, and system-design question bank
├── case-studies/          # Full interview case studies with solutions
├── README.md              # This file
├── MLOPS_ROADMAP.md       # Learning paths for different roles
└── CONTRIBUTING.md        # How to add concepts or questions
```

## Each Concept Includes

**Markdown File** (`concepts/{NN}-{name}.md`):
- Comprehensive overview (4-5 paragraphs)
- How it works (architecture and flow)
- Tool comparisons (2-3 tools with trade-offs)
- Interview Q&A (5-8 judgment-focused questions)
- Best practices (5+ production patterns)
- Real-world case studies (2-3 FAANG examples)
- Sample interview questions (3-5 from FAANG)
- Full interview case study (complete scenario)
- Common answer patterns (strong vs weak)

**Jupyter Notebook** (`notebooks/{NN}-{name}.ipynb`):
- Introduction & learning objectives
- Basic implementation (20-40 lines)
- Advanced implementation (60-100 lines, production patterns)
- 3 real-world examples (40-60 lines each)
- Interview scenario (annotated code)
- Key takeaways for interview prep

## Quality Standards

✅ All imports are real (MLflow, Airflow, Feast, FastAPI, etc.)  
✅ Production patterns throughout (error handling, logging, monitoring)  
✅ Code is tested and runnable  
✅ Multi-tool approach (show 2-3 implementations with trade-offs)  
✅ Interview-focused Q&A (judgment calls, not memorization)  
✅ Based on real FAANG interview questions and system designs  

## How to Use This Material

### For FAANG Interview Prep
1. Choose your role (Data Engineer, ML Engineer, Systems Engineer)
2. Follow the learning path: 1-3 weeks, 2 hours/day
3. Read concept markdown → Code notebook → Practice interview questions
4. Study case studies → Understand strong answer patterns
5. Practice explaining in 2-minute and 10-minute versions

### For Production Learning
1. Start with concepts relevant to your role (data pipeline work? Start at 1)
2. Read markdown for theory and best practices
3. Code notebook to understand implementation
4. Reference case studies for real-world patterns
5. Refer back when making production decisions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to:
- Add new concepts
- Improve existing concepts
- Add interview questions
- Add case studies

## Resources & References

Each concept links to original papers, tools, and documentation:
- Research papers (if applicable)
- Official documentation
- Production case studies
- Interview resources

## Learning Goals

After studying this material, you will:
- ✅ Understand ML Ops lifecycle from data to production
- ✅ Know production patterns and trade-offs
- ✅ Ace ML systems design interviews
- ✅ Make informed tool choices in real projects
- ✅ Recognize pitfalls and failure modes
- ✅ Explain concepts clearly to interviewers

## Contact & Updates

This material is continuously updated with new concepts, interview questions, and case studies. Check back regularly for new content.

---

**Get started:** Read [Concept 1: Data Pipelines](concepts/01-data-pipelines.md)
