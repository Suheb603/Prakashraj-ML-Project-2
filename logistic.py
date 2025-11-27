# ============================================================
# Founder Retention Prediction – Logistic Regression (sklearn)
# Uses uploaded CSV files
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------------
# 1. Load Data
# --------------------------

train_path = "train_1.csv"
test_path  = "test_1.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# --------------------------
# 2. Process Target
# --------------------------

target_col = "retention_status"
mapping = {"Stayed": 1, "Left": 0}
train[target_col] = train[target_col].map(mapping)

# --------------------------
# 3. Extract Features
# --------------------------

test_ids = test["founder_id"]

train_features = train.drop(columns=["founder_id", target_col])
test_features = test.drop(columns=["founder_id"])

# --------------------------
# 4. Encode Categorical Columns
# --------------------------

categorical_cols = train_features.select_dtypes(include="object").columns

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_features[col], test_features[col]], axis=0).astype(str)
    le.fit(combined)
    train_features[col] = le.transform(train_features[col].astype(str))
    test_features[col] = le.transform(test_features[col].astype(str))
    encoders[col] = le

# Replace missing values
train_features = train_features.fillna(-1)
test_features = test_features.fillna(-1)

# --------------------------
# 5. Scaling
# --------------------------

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# --------------------------
# 6. Train/Val Split
# --------------------------

X_train, X_val, y_train, y_val = train_test_split(
    train_scaled,
    train[target_col],
    test_size=0.5,
    random_state=42,
    stratify=train[target_col]
)

# --------------------------
# 7. Logistic Regression Model
# --------------------------

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=50,
    C=1.0,           
    class_weight="balanced"  
)

print("Training Logistic Regression...")
model.fit(X_train, y_train)

# --------------------------
# 8. Validation AUC
# --------------------------

val_prob = model.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_prob))

# --------------------------
# 9. Predict Test Data
# --------------------------

test_prob = model.predict_proba(test_scaled)[:, 1]
test_pred_label = np.where(test_prob >= 0.5, "Stayed", "Left")

# --------------------------
# 10. Submission
# --------------------------

submission = pd.DataFrame({
    "founder_id": test_ids,
    "retention_status": test_pred_label
})

submission.to_csv("submission_logreg.csv", index=False)

print("Saved: submission_logreg.csv")
submission.head()
