import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, r"WEEK 2\DAY 2")
from Pipe import TelcoFeatureEngineer


df = pd.read_csv(r"cleaned_data\Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
X = df.drop("Churn", axis=1)
y = (df["Churn"] == "Yes").astype(int)

numeric_columns = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "charges_per_tenure", "total_services",
]
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "tenure_bucket", "PaymentMethod",
]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_columns),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_columns),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
}

# Task 1: compare the two baseline models using stratified 5-fold CV.
# False negatives are the actual churners that the model failed to catch.
baseline_probabilities = {}
baseline_rows = []
for name, model in models.items():
    pipeline = ImbPipeline([
        ("features", TelcoFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model),
    ])
    probabilities = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba"
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    baseline_probabilities[name] = probabilities
    baseline_rows.append({
        "Model": name,
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1": f1_score(y, predictions),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })
    
    print(baseline_rows[-1])



# PR curves matter more than ROC curves here because ROC-AUC can look good
# when the majority class dominates, while PR shows minority-class precision.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, probabilities in baseline_probabilities.items():
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, probabilities):.3f})")
    axes[1].plot(
        recall, precision,
        label=f"{name} (AP={average_precision_score(y, probabilities):.3f})",
    )
axes[0].plot([0, 1], [0, 1], "--", color="grey")
axes[0].set(title="ROC curves", xlabel="False positive rate", ylabel="Recall")
axes[1].axhline(y.mean(), linestyle="--", color="grey", label="Churn prevalence")
axes[1].set(
    title="Precision-Recall curves", xlabel="Recall", ylabel="Precision"
)
axes[0].legend()
axes[1].legend()
fig.tight_layout()
fig.savefig(r"WEEK 2\DAY 4\roc_pr_curves.png", dpi=160)
plt.close(fig)

# Task 1: confusion matrices for the two baseline models.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for axis, row in zip(axes, baseline_rows):
    matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
    axis.imshow(matrix, cmap="Blues")
    for (row_number, column_number), value in np.ndenumerate(matrix):
        axis.text(column_number, row_number, str(value), ha="center", va="center")
    axis.set(
        title=row["Model"],
        xticks=[0, 1], xticklabels=["No churn", "Churn"],
        yticks=[0, 1], yticklabels=["No churn", "Churn"],
        xlabel="Predicted", ylabel="Actual",
    )
fig.tight_layout()
fig.savefig(r"WEEK 2\DAY 4\confusion_matrices.png", dpi=160)
plt.close(fig)

# Task 2: compare imbalance strategies on Logistic Regression only.
# SMOTE is inside the CV pipeline, so synthetic points are created only from
# each training fold and never leak into that fold's validation data.
imbalance_models = {
    "Baseline": LogisticRegression(max_iter=1000, random_state=42),
    "class_weight='balanced'": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "SMOTE": SMOTE(random_state=42),
}
imbalance_rows = []
for name, strategy in imbalance_models.items():
    steps = [
        ("features", TelcoFeatureEngineer()),
        ("preprocessing", preprocessor),
    ]
    if name == "SMOTE":
        steps.append(("smote", strategy))
        steps.append(("model", LogisticRegression(max_iter=1000, random_state=42)))
    else:
        steps.append(("model", strategy))
    probabilities = cross_val_predict(
        ImbPipeline(steps), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    imbalance_rows.append({
        "Approach": name,
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1": f1_score(y, predictions),
    })

imbalance_results = pd.DataFrame(imbalance_rows)
print("\nLogistic Regression imbalance comparison")
print(imbalance_results.to_string(index=False))

# Task 3: lower the threshold because a missed churner costs more than a
# retention offer sent to a customer who would not have churned.
baseline_probabilities = baseline_probabilities["Logistic Regression"]
threshold_rows = []
for threshold in np.arange(0.1, 1.0, 0.1):
    predictions = (baseline_probabilities >= threshold).astype(int)
    threshold_rows.append({
        "Threshold": threshold,
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
    })
threshold_results = pd.DataFrame(threshold_rows)
print("\nPrecision and recall by threshold")
print(threshold_results.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.plot(threshold_results["Threshold"], threshold_results["Precision"],
         marker="o", label="Precision")
plt.plot(threshold_results["Threshold"], threshold_results["Recall"],
         marker="o", label="Recall")
plt.xlabel("Probability threshold")
plt.ylabel("Score")
plt.title("Logistic Regression threshold trade-off")
plt.xticks(np.arange(0.1, 1.0, 0.1))
plt.legend()
plt.tight_layout()
plt.savefig(r"WEEK 2\DAY 4\threshold_curves.png", dpi=160)
plt.close()

print("\nClassification report at the default 0.5 threshold")
print(classification_report(y, (baseline_probabilities >= 0.5).astype(int)))
print("\nChosen threshold: 0.3, because catching more churners matters more than")
print("avoiding every unnecessary retention offer.")
