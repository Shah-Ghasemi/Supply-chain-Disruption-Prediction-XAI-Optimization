# ML-Driven Disruption Risk Prediction with Explainable AI (XAI)
## for Resilient Supply Chain Network Optimization

![Pipeline](https://img.shields.io/badge/Pipeline-ML%20→%20XAI%20→%20Optimization-blueviolet)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.49-green)
![Optimization](https://img.shields.io/badge/Optimization-Linear%20Programming-red)
![Dataset](https://img.shields.io/badge/Dataset-DataCo%20Supply%20Chain-informational)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project implements an end-to-end **ML → XAI → Optimization** 
pipeline for supply chain disruption risk prediction and 
logistics decision support.

Given a supply chain order, the pipeline:
1. **Predicts** the probability of delivery disruption (XGBoost)
2. **Explains** which factors drive the prediction (SHAP)
3. **Optimizes** shipping mode assignments to minimize the 
   weighted sum of operational cost and disruption risk (Linear Programming)

> *"How can interpretable machine learning insights be leveraged 
> to optimize logistics decisions — specifically shipping mode 
> selection — in order to achieve a balanced trade-off between 
> operational costs and disruption risk?"*

---

## Pipeline Architecture

```
Raw Data (DataCo Supply Chain)
         │
         ▼
┌─────────────────────┐
│  EDA & Data Quality │  ← Structural anomaly detection
│  Audit              │     Shipping mode analysis
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing &    │  ← Leakage-free feature selection
│  Leakage Prevention │     Temporal feature engineering
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Baseline Model     │  ← Logistic Regression
│  (Logistic Reg.)    │     AUC benchmark
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  XGBoost +          │  ← RandomizedSearchCV tuning
│  Hyperparameter     │     Leakage validation protocol
│  Tuning             │     Threshold optimization
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  SHAP Explainability│  ← Global: Summary + Importance
│  (XAI)             │     Feature-level: Dependence plots
│                     │     Local: Per-order explanations
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LP Optimization    │  ← Pareto frontier analysis
│                     │     Scenario analysis (α sweep)
│                     │     Managerial decision support
└─────────────────────┘
```

---

## Key Results

| Stage | Metric | Value |
|---|---|---|
| **ML** | AUC-ROC (leakage-free) | 0.756 |
| **ML** | Recall — Disruption class | 71.5% |
| **ML** | Precision — Disruption class | 71.3% |
| **XAI** | Top controllable feature | Shipping Mode |
| **Optimization** | Cost reduction (α=0.5) | ~11% |
| **Optimization** | Risk reduction (α=0.5) | ~20% |
| **Optimization** | Orders optimized | ~25% |

---

## Dataset

**DataCo Smart Supply Chain Dataset**  
Source: [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

- 180,519 supply chain transactions
- Global retail supply chain (orders, shipping, customers, products)
- Key variables: Shipping Mode, Order Region, Scheduled vs Actual 
  delivery days, Product Category, Customer Segment

### Data Quality Note

A structural anomaly was identified during EDA:

| Shipping Mode | Scheduled Days | Actual Days | Disruption Rate |
|---|---|---|---|
| Same Day | 0 | 0.48 | 47.8% |
| First Class | 1 | 2.00 | 100.0% |
| Second Class | 2 | 3.99 | 79.7% |
| Standard Class | 4 | 4.00 | 39.8% |

First Class and Same Day have unrealistic scheduled windows, 
producing artificially inflated disruption rates. 
**Optimization scope is limited to Standard Class and Second 
Class** — the two modes with reliable and comparable 
disruption baselines. This is a data quality decision, 
not a modeling limitation.

---

## Data Leakage Prevention

A rigorous three-test leakage validation protocol was applied:

```
Test A — Full feature set       : AUC = 0.756
Test B — Remove scheduled days  : AUC = 0.757  (↑ slightly, no dependency)
Test C — Safe features only     : AUC = 0.503  (≈ random → no hidden leakage)
```

**Excluded features (leakage risk):**

| Feature | Reason |
|---|---|
| `shipping date` | Post-order information — not available at prediction time |
| `planned_lead_time` | Derived from shipping date → temporal leakage |
| `Days for shipping (real)` | The outcome itself → target leakage |
| `Late_delivery_risk` | Source of target variable |

---

## Methodology

### 1. Disruption Risk Prediction

- **Baseline:** Logistic Regression with `class_weight='balanced'`
- **Main model:** XGBoost with `scale_pos_weight` for class imbalance
- **Tuning:** RandomizedSearchCV (30 iterations, 3-fold CV, AUC scoring)
- **Threshold:** 0.34 (optimized via Precision-Recall curve)
  - Default 0.5 threshold penalizes false negatives and false positives equally
  - In supply chain, missing a disruption is costlier than a false alarm
  - Threshold 0.34 achieves Precision ≈ Recall ≈ 0.71 for disruption class

### 2. Explainability (SHAP)

Three levels of explanation:

```
Global  → Summary plot: which features matter most across all orders?
Feature → Dependence plots: how does each feature affect risk?
Local   → Force plots: why is THIS specific order high-risk?
```

SHAP confirms Shipping Mode as the top controllable risk driver — 
providing the theoretical justification for optimization intervention.

### 3. LP Optimization

**Decision variable:** Shipping mode assignment per order  
**Objective:** Minimize α × risk + (1−α) × normalized_cost  
**Constraints:**
- Each order assigned to exactly one mode
- Average cost ≤ 110% of current average (budget constraint)

**Solver:** HiGHS via `scipy.optimize.linprog`

The tradeoff parameter α is swept across [0, 1] to generate 
the full Pareto frontier, enabling stakeholders to select 
their preferred risk-cost operating point.

---

## Project Structure

```
📁 repository/
│
├── 📓 notebook.ipynb           ← Main notebook (all sections)
│
├── 📁 figures/                 ← Generated visualizations
│   ├── eda_shipping_mode.png
│   ├── eda_temporal.png
│   ├── eda_financial.png
│   ├── baseline_evaluation.png
│   ├── baseline_coefficients.png
│   ├── model_comparison.png
│   ├── threshold_analysis.png
│   ├── threshold_comparison.png
│   ├── shap_summary.png
│   ├── shap_importance_bar.png
│   ├── shap_dependence.png
│   ├── shap_local.png
│   ├── shap_shipping_bridge.png
│   ├── optimization_results.png
│   └── pipeline_summary.png
│
├── 📄 README.md
└── 📄 requirements.txt
```

---

## Notebook Sections

| Section | Title | Key Output |
|---|---|---|
| 1 | Problem Framing | Research questions, pipeline diagram |
| 2 | EDA | Shipping mode analysis, temporal patterns |
| 3 | Preprocessing & Leakage Prevention | Leakage audit, clean feature set |
| 4 | Baseline Model | Logistic Regression, AUC = 0.711 |
| 5 | XGBoost + Tuning | Leakage validation, AUC = 0.756 |
| 6 | Threshold Optimization | Precision-Recall analysis, t = 0.34 |
| 7 | SHAP Explainability | 3-level XAI, shipping mode bridge |
| 8 | LP Optimization | Pareto frontier, managerial recommendations |

---

## Connection to Doctoral Research

This mini-project serves as a **proof-of-concept** for the 
following doctoral research agenda:

> *"Data-Driven Robust Optimization for Resilient Circular 
> Supply Chain Network Design: A Machine Learning-Enhanced 
> Approach under Deep Uncertainty"*

| Dimension | This Project | Dissertation |
|---|---|---|
| Decision level | Order assignment | Network design |
| Uncertainty | Deterministic | Deep uncertainty (ambiguity sets) |
| Sustainability | Not modeled | Circular economy constraints |
| Optimization | LP (operational) | Robust LP / Stochastic Programming |

---

## Limitations & Future Work

- **Optimization scope:** Two shipping modes only (data quality constraint)
- **Cost model:** Normalized relative costs; real deployment requires 
  actual logistics cost data
- **LP relaxation:** Continuous relaxation of binary assignment; 
  Integer Programming gives exact binary solutions
- **Static predictions:** Fixed disruption probabilities; 
  online learning would enable dynamic risk updating
- **Network level:** Current work operates at order level; 
  dissertation extends to strategic facility/supplier network design

---

## Author

**Shahrokh Ghasemi Dehcheshmeh**  
PhD Candidate — Supply Chain Management & Operations Research  

*This project was developed as part of doctoral research 
preparation in ML-enhanced supply chain optimization.*

---

## Development Note
This project was developed over three weeks of iterative 
experimentation. AI coding assistants (Claude) were used 
for code documentation and refactoring in the final stage. 
All analytical decisions, model design choices, leakage 
detection, and result interpretation were performed by 
the author.

---

## License

MIT License — free to use, cite, and build upon with attribution.
