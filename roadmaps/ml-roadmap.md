# Machine Learning Roadmap

## Who This Is For
Engineers who want to deeply understand classical ML, neural networks, and deep learning — and be
ready for ML theory and coding interviews at any top tech company. You should be comfortable
with Python and have basic linear algebra / calculus knowledge.

---

## Phase 1 — Foundations (Beginner)
**Goal:** Explain and implement core ML algorithms from scratch. Pass easy/medium ML theory questions.
**Estimated time:** 3–4 weeks at 10 hrs/week

- [ ] [Supervised Learning](../ml/concepts/supervised-learning.md)
- [ ] [Probability & Statistics](../ml/concepts/probability-statistics.md)
- [ ] [Evaluation Metrics](../ml/concepts/evaluation-metrics.md)
- [ ] [Optimization](../ml/concepts/optimization.md)
- [ ] [Regularization](../ml/concepts/regularization.md)
- [ ] Implement: [Linear Regression](../ml/implementations/linear-regression.ipynb)
- [ ] Implement: [Logistic Regression](../ml/implementations/logistic-regression.ipynb)
- [ ] Practice: [ML Theory Questions](../ml/interview-prep/ml-theory-questions.md) — Q1–Q20

**Phase 1 exit check:**
- Can you derive the gradient descent update rule from scratch?
- Can you explain bias-variance trade-off with a diagram?
- Can you implement linear regression in NumPy without looking it up?

---

## Phase 2 — Core Depth (Intermediate)
**Goal:** Cover tree models, unsupervised learning, neural networks, and backprop. Pass medium/hard ML theory questions.
**Estimated time:** 3–4 weeks at 10 hrs/week

- [ ] [Unsupervised Learning](../ml/concepts/unsupervised-learning.md)
- [ ] [Feature Engineering](../ml/concepts/feature-engineering.md)
- [ ] [Ensemble Methods](../ml/concepts/ensemble-methods.md)
- [ ] [Neural Networks](../ml/concepts/neural-networks.md)
- [ ] [Deep Learning — CNNs](../ml/concepts/deep-learning/cnns.md)
- [ ] [Deep Learning — RNNs & LSTMs](../ml/concepts/deep-learning/rnns-lstms.md)
- [ ] Implement: [Decision Tree](../ml/implementations/decision-tree.ipynb)
- [ ] Implement: [Random Forest](../ml/implementations/random-forest.ipynb)
- [ ] Implement: [K-Means From Scratch](../ml/implementations/kmeans-from-scratch.ipynb)
- [ ] Implement: [Neural Net From Scratch](../ml/implementations/neural-net-from-scratch.ipynb)
- [ ] Implement: [Backpropagation](../ml/implementations/backpropagation.ipynb)
- [ ] Practice: [ML Theory Questions](../ml/interview-prep/ml-theory-questions.md) — Q21–Q50
- [ ] Practice: [ML Coding Questions](../ml/interview-prep/ml-coding-questions.md) — Q1–Q10

**Phase 2 exit check:**
- Can you explain how a random forest reduces variance without increasing bias?
- Can you implement backprop for a 2-layer network with no libraries?
- Can you describe when gradient boosting beats random forest and why?

---

## Phase 3 — Advanced + Production (Advanced)
**Goal:** Cover transformers, distributed training, MLOps. Ready for senior ML engineer interviews.
**Estimated time:** 2–4 weeks at 10 hrs/week

- [ ] [Deep Learning — Attention Mechanism](../ml/concepts/deep-learning/attention-mechanism.md)
- [ ] [Deep Learning — Transformers](../ml/concepts/deep-learning/transformers.md)
- [ ] Implement: [CNN Image Classifier](../ml/implementations/cnn-image-classifier.ipynb)
- [ ] Implement: [Backpropagation](../ml/implementations/backpropagation.ipynb) — verify gradient correctness
- [ ] Practice: [ML Coding Questions](../ml/interview-prep/ml-coding-questions.md) — Q11–Q20
- [ ] Practice: [ML Case Studies](../ml/interview-prep/case-studies.md)
- [ ] [System Design — MLOps Overview](../system-design/patterns/mlops-overview.md)

**Phase 3 exit check:**
- Can you explain the attention mechanism and why it replaced RNNs?
- Can you design an end-to-end training pipeline for a large model?
- Can you answer: "How would you deploy a model that needs to serve 10k QPS?"

---

## Interview Readiness Checklist
- [ ] Implemented linear/logistic regression, decision tree, neural net from scratch (NumPy only)
- [ ] Can explain gradient descent, backprop, attention without notes
- [ ] Completed at least 20 ML theory questions in simulation format
- [ ] Completed at least 5 ML coding questions from scratch
- [ ] Done one full case study mock (recommendation or ranking system)

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | Supervised learning + probability | supervised-learning.md, probability-statistics.md |
| 2 | Evaluation + optimization + regularization | evaluation-metrics.md, optimization.md, regularization.md |
| 3 | Implement linear + logistic regression | linear-regression.ipynb, logistic-regression.ipynb |
| 4 | Theory practice + unsupervised | ml-theory-questions.md Q1–20, unsupervised-learning.md |
| 5 | Ensemble methods + trees | ensemble-methods.md, decision-tree.ipynb, random-forest.ipynb |
| 6 | Neural networks + backprop | neural-networks.md, neural-net-from-scratch.ipynb, backpropagation.ipynb |
| 7 | Deep learning (CNN, RNN) | cnns.md, rnns-lstms.md, cnn-image-classifier.ipynb |
| 8 | Attention + transformers + coding Qs | attention-mechanism.md, transformers.md, ml-coding-questions.md |
