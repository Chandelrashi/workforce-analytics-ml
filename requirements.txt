# Workforce Analytics using Applied Machine Learning

## Overview
This project demonstrates an **applied machine learning pipeline for workforce analytics**, with a focus on **employee attrition risk prediction** and **data-driven workforce planning**.

The goal is not only to build predictive models, but to show how **AI-driven insights can support organisational decision-making** in real-world HR and business contexts.

---

## Decision Problem
Organisations face increasing challenges in:
- Identifying early indicators of employee attrition
- Understanding workforce risk factors
- Planning retention and hiring strategies proactively

Traditional reporting often fails to provide **forward-looking insights**. This project addresses that gap by applying machine learning to workforce data to support **evidence-based decisions**.

---

## Data
- Synthetic HR dataset representing employee demographics, performance, tenure, and engagement indicators  
- Synthetic data is used to demonstrate methodology while preserving privacy

Key features include:
- Tenure and role level
- Performance ratings
- Absenteeism and engagement indicators
- Compensation-related variables

---

## Approach
The project follows an end-to-end applied analytics pipeline:

1. Data preprocessing and feature engineering  
2. Exploratory data analysis to identify workforce patterns  
3. Supervised machine learning models for attrition prediction  
4. Model evaluation using standard classification metrics  
5. Interpretation of key drivers influencing attrition risk  

Models implemented include baseline and tree-based classifiers using `scikit-learn`.

---

## Results (Baseline)
- Model type: Classification (e.g. Logistic Regression / Random Forest)
- Evaluation metrics:
  - Accuracy: _to be updated_
  - F1-score: _to be updated_
  - ROC-AUC: _to be updated_

Key predictive drivers observed:
- Employee tenure
- Performance rating trends
- Absenteeism frequency
- Role level and progression

---

## Decision & Business Use
This analytics pipeline can support:
- Early identification of high-risk attrition segments
- Targeted retention and engagement strategies
- Workforce planning and scenario analysis
- HR leaders and managers in making **data-driven people decisions**

The emphasis is on **practical deployment of AI**, not academic modelling alone.

---

## Project Structure
workforce-analytics-ml/
│

├── src/ # Core Python scripts

├── README.md # Project documentation

├── requirements.txt # Python dependencies

└── assets/ # Visuals (charts, model outputs)

---

## How to Run
1. Clone the repository:
git clone https://github.com/Chandelrashi/workforce-analytics-ml.git

3. Install dependencies:
pip install -r requirements.txt

4. Run the main pipeline:
python -m src.main

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (for analysis and visualisation)

---

## Author & Context
This project forms part of my broader work in **Applied AI and Data Science**, focused on bridging the gap between machine learning models and real-world organisational decision-making.

Related work includes:
- Book: *Applied AI for Data-Driven Decision Making*
- Research and applied analytics publications

