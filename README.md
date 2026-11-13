# Telecom Customer Churn Prediction

An end-to-end Machine Learning classification pipeline and interactive Streamlit web application that predicts whether a telecom customer is at risk of churning.

## Problem Statement
Acquiring a new telecom customer costs significantly more than retaining an existing one. This project identifies high-risk subscribers early using billing, tenure, contract type, and support interaction signals so retention teams can intervene proactively.

## Architecture
`Data Generation & Cleaning` $\rightarrow$ `Feature Engineering (Charge-to-Tenure Ratio)` $\rightarrow$ `Balanced Random Forest Classifier` $\rightarrow$ `Joblib Serialization` $\rightarrow$ `Streamlit Web App`

## Key Highlights & Metrics
* **Class Imbalance Handling:** Used `class_weight="balanced"` and stratified train-test splitting to prevent majority-class bias.
* **Evaluation Focus:** Evaluated using **ROC-AUC (~0.94)**, **Precision**, **Recall**, and **F1-Score** rather than raw accuracy alone.
* **Interactive Inference:** Built a self-contained Streamlit dashboard allowing real-time risk simulation.

## Quickstart
```bash
git clone [https://github.com/kirtan3ai-code/customer-churn-prediction.git](https://github.com/kirtan3ai-code/customer-churn-prediction.git)
cd customer-churn-prediction
pip install -r requirements.txt
python train.py
streamlit run app.py
