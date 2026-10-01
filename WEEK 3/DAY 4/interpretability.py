from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.inspection import PartialDependenceDisplay

RAW_FEATURES = ["store_nbr", "family", "year", "month", "day_of_week", "is_holiday", "onpromotion"]
CAT_FEATURES = [0, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("WEEK 3/DAY 1/feature_table_v1.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("WEEK 3/DAY 4"))
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--val-rows", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input).sort_values("date").reset_index(drop=True)
    cutoff = df["date"].quantile(0.8)
    train = df[df["date"] <= cutoff].sample(n=args.max_train_rows, random_state=args.random_state)
    validation = df[df["date"] > cutoff].sample(n=args.val_rows, random_state=args.random_state)
    X_train = train[RAW_FEATURES].copy()
    X_val = validation[RAW_FEATURES].copy()
    for data in (X_train, X_val):
        data["store_nbr"] = data["store_nbr"].astype(str)
        data["family"] = data["family"].astype(str)
    y_train = train["sales"].to_numpy()
    y_val = validation["sales"].to_numpy()

    model = CatBoostRegressor(iterations=100, verbose=0, random_seed=args.random_state, allow_writing_files=False)
    model.fit(X_train, y_train, cat_features=CAT_FEATURES)
    predictions = model.predict(X_val)
    errors = np.abs(y_val - predictions)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    importance = pd.DataFrame({"feature": RAW_FEATURES, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, show=False, max_display=len(RAW_FEATURES))
    plt.tight_layout()
    plt.savefig(args.output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    selected = [int(np.argmin(errors)), int(np.argmax(errors)), int(np.argmax(predictions))]
    labels = ["typical_low_error", "highest_error", "highest_prediction"]
    force_rows = []
    for label, index in zip(labels, selected):
        force = shap.force_plot(explainer.expected_value, shap_values[index], X_val.iloc[index], matplotlib=False)
        shap.save_html(str(args.output_dir / f"shap_force_{label}.html"), force)
        force_rows.append({
            "case": label,
            "validation_index": index,
            "date": validation.iloc[index]["date"].date().isoformat(),
            "store_nbr": validation.iloc[index]["store_nbr"],
            "family": validation.iloc[index]["family"],
            "actual_sales": y_val[index],
            "predicted_sales": predictions[index],
            "absolute_error": errors[index],
        })

    top_three = importance.head(3)["feature"].tolist()
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    PartialDependenceDisplay.from_estimator(
        model, X_val, top_three, ax=axes, kind="average", grid_resolution=50, categorical_features=CAT_FEATURES
    )
    figure.suptitle("Partial dependence for the three most important SHAP features")
    figure.tight_layout()
    figure.savefig(args.output_dir / "partial_dependence_top3.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(importance.to_string(index=False))
    print(pd.DataFrame(force_rows).to_string(index=False))


if __name__ == "__main__":
    main()


