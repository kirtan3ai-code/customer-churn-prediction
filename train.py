import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Generate realistic telecom customer dataset
np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    "tenure_months": np.random.randint(1, 72, n_samples),
    "monthly_charges": np.round(np.random.uniform(299, 2499, n_samples), 2),
    "contract_type": np.random.choice([0, 1, 2], n_samples, p=[0.55, 0.25, 0.20]), # 0: Month-to-month, 1: 1-Yr, 2: 2-Yr
    "tech_support": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    "support_calls": np.random.poisson(1.8, n_samples)
})

# Create target variable (Churn: 1 = Yes, 0 = No) based on realistic patterns
churn_prob = (
    0.45 * (data["contract_type"] == 0).astype(int)
    + 0.25 * (data["support_calls"] > 2).astype(int)
    + 0.20 * (data["tenure_months"] < 12).astype(int)
    + np.random.uniform(0, 0.25, n_samples)
)
data["churn"] = (churn_prob > 0.65).astype(int)

# 2. Feature Engineering
data["charge_per_month_ratio"] = data["monthly_charges"] / (data["tenure_months"] + 1)

# 3. Train-Test Split
X = data.drop("churn", axis=1)
y = data["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model Training (Handling class imbalance)
model = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(model, "churn_model.pkl")
print("Model saved successfully as churn_model.pkl")
