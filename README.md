# Workforce Analytics using Applied Machine Learning

## Overview
This project demonstrates an **applied machine learning pipeline for workforce analytics**, with a focus on **employee attrition risk prediction** and **data‑driven workforce planning**.

The goal is not only to build predictive models, but to show how **AI‑driven insights can support organisational decision‑making** in real‑world HR and business contexts.

---

## Decision Problem
Organisations face increasing challenges in:
- Identifying early indicators of employee attrition
- Understanding workforce risk factors
- Planning retention and hiring strategies proactively

Traditional reporting often fails to provide **forward‑looking insights**. This project addresses that gap by applying machine learning to workforce data to support **evidence‑based decisions**.

---

## Data
- Synthetic HR dataset representing employee demographics, performance, tenure, and engagement indicators
- Synthetic data is used to demonstrate methodology while preserving privacy

Key features include:
- Tenure and role level
- Performance ratings
- Absenteeism and engagement indicators
- Compensation‑related variables

## Data Note

This project uses a synthetically generated workforce dataset created within the repository to reflect realistic organisational patterns. Synthetic data is used to demonstrate methodology, reproducibility, and decision‑making workflows while preserving privacy and ethical standards.

---

## Approach
The project follows an end‑to‑end applied analytics pipeline:

1. Data preprocessing and feature engineering
2. Exploratory data analysis to identify workforce patterns
3. Supervised machine learning models for attrition prediction
4. Model evaluation using standard classification metrics
5. Interpretation of key drivers influencing attrition risk

Models implemented include baseline and tree‑based classifiers using `scikit‑learn`.

---

## Results Summary

The models were evaluated using stratified 5‑fold cross‑validation and a held‑out test set to ensure robustness and generalisability.

| Model | CV ROC‑AUC (mean ± std) | Test ROC‑AUC | Test PR‑AUC |
|---|---|---|---|
| Logistic Regression | 0.764 ± 0.039 | 0.766 | 0.629 |
| Random Forest       | 0.806 ± 0.027 | 0.849 | 0.702 |

The Random Forest model demonstrates stronger non‑linear modelling capability, while Logistic Regression provides interpretability and stability.

Key predictive drivers observed:
- Employee tenure
- Performance rating trends
- Absenteeism frequency
- Role level and progression

---

## Decision & Business Use
This analytics pipeline can support:
- Early identification of high‑risk attrition segments
- Targeted retention and engagement strategies
- Workforce planning and scenario analysis
- HR leaders and managers in making **data‑driven people decisions**

The emphasis is on **practical deployment of AI**, not academic modelling alone.

---

## Author Note
This project reflects my focus on applying machine learning to real‑world organisational decision‑making, bridging analytics, business context, and responsible AI practices.
