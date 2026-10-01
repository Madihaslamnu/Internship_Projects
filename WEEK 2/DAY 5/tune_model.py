from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.visualization.matplotlib import plot_optimization_history
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, r"WEEK 2\DAY 2")
from Pipe import TelcoFeatureEngineer


DATA_PATH = Path(r"cleaned_data\Telco-Customer-Churn.csv")
OUTPUT_DIR = Path(r"WEEK 2\DAY 5")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
X = df.drop(columns="Churn")
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


def make_pipeline(c, penalty, class_weight):
    return Pipeline([
        ("features", TelcoFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(
            C=c,
            penalty=penalty,
            solver="liblinear",
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )),
    ])


def objective(trial):
    pipeline = make_pipeline(
        trial.suggest_float("C", 1e-3, 10.0, log=True),
        trial.suggest_categorical("penalty", ["l1", "l2"]),
        trial.suggest_categorical("class_weight", [None, "balanced"]),
    )
    return cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1
    ).mean()


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=30)

best_pipeline = make_pipeline(
    study.best_params["C"],
    study.best_params["penalty"],
    study.best_params["class_weight"],
)
best_pipeline.fit(X, y)
joblib.dump(best_pipeline, OUTPUT_DIR / "tuned_pipeline.joblib")

plt.figure()
plot_optimization_history(study)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "optimization_history.png", dpi=160)
plt.close()

probabilities = cross_val_predict(
    best_pipeline, X, y, cv=cv, method="predict_proba"
)[:, 1]
predictions = (probabilities >= 0.3).astype(int)
print(f"Trials completed: {len(study.trials)}")
print(f"Best F1 CV score: {study.best_value:.3f}")
print(f"Best parameters: {study.best_params}")
print(f"Threshold 0.30 F1: {f1_score(y, predictions):.3f}")
print(f"Threshold 0.30 ROC-AUC: {roc_auc_score(y, probabilities):.3f}")
