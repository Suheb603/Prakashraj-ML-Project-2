# ============================================================
# FAST Founder Retention Prediction – Optimized SVM (NO GRID)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# --------------------------
# 1. Load Data
# --------------------------

train = pd.read_csv("train_1.csv")
test  = pd.read_csv("test_1.csv")

target_col = "retention_status"
mapping = {"Stayed": 1, "Left": 0}
train[target_col] = train[target_col].map(mapping)

test_ids = test["founder_id"]

train_features = train.drop(columns=["founder_id", target_col])
test_features  = test.drop(columns=["founder_id"])

# --------------------------
# 2. Encode Categorical Columns
# --------------------------

categorical_cols = train_features.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_features[col], test_features[col]], axis=0).astype(str)
    le.fit(combined)
    train_features[col] = le.transform(train_features[col].astype(str))
    test_features[col] = le.transform(test_features[col].astype(str))

train_features = train_features.fillna(-1)
test_features  = test_features.fillna(-1)

# --------------------------
# 3. Fast Robust Scaling
# --------------------------

scaler = RobustScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled  = scaler.transform(test_features)

# --------------------------
# 4. Train/Val Split
# --------------------------

X_train, X_val, y_train, y_val = train_test_split(
    train_scaled,
    train[target_col],
    test_size=0.2,
    random_state=42,
    stratify=train[target_col]
)

# --------------------------
# 5. FAST SVM (Carefully chosen params)
# --------------------------

model = SVC(
    C=8,               
    gamma=0.03,       
    kernel="rbf",
    probability=True,
    class_weight="balanced",
)

print("Training FAST SVM...")
model.fit(X_train, y_train)

# --------------------------
# 6. Validation AUC
# --------------------------

val_prob = model.predict_proba(X_val)[:, 1]
print("Validation AUC:", roc_auc_score(y_val, val_prob))

# --------------------------
# 7. Final Train on FULL DATA
# --------------------------

final_svm = SVC(
    C=8,
    gamma=0.03,
    kernel="rbf",
    probability=True,
    class_weight="balanced"
)
final_svm.fit(train_scaled, train[target_col])

# --------------------------
# 8. Predict Test
# --------------------------

test_prob = final_svm.predict_proba(test_scaled)[:, 1]
test_pred_label = np.where(test_prob >= 0.5, "Stayed", "Left")

# --------------------------
# 9. Save Submission
# --------------------------

submission = pd.DataFrame({
    "founder_id": test_ids,
    "retention_status": test_pred_label
})

submission.to_csv("submission_fast_svm.csv", index=False)
print("Saved submission_fast_svm.csv")
