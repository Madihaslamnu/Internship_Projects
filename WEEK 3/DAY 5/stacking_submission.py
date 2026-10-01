from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

FEATURES = ["store_nbr", "family", "year", "month", "day_of_week", "is_holiday", "onpromotion"]
CAT_COLS = [0, 1]
MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["year"] = result["date"].dt.year.astype(int)
    result["month"] = result["date"].dt.month.astype(int)
    result["day_of_week"] = result["date"].dt.dayofweek.astype(int)
    ec = holidays.country_holidays("EC", years=range(result["year"].min(), result["year"].max() + 1))
    result["is_holiday"] = result["date"].dt.date.map(lambda d: int(d in ec))
    result["onpromotion"] = result["onpromotion"].fillna(0).astype(float)
    return result


def model_factory(name: str, seed: int):
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.08, tree_method="hist", n_jobs=4, random_state=seed)
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=100, learning_rate=0.08, verbosity=-1, n_jobs=4, random_state=seed)
    return CatBoostRegressor(iterations=100, verbose=0, allow_writing_files=False, random_seed=seed)


def prepare_x(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    x = frame[FEATURES].copy()
    x["store_nbr"] = x["store_nbr"].astype(str)
    x["family"] = x["family"].astype(str)
    return x


def fit_model(model, name, x, y):
    if name == "CatBoost":
        model.fit(x, y, cat_features=CAT_COLS)
    else:
        model.fit(x if name == "CatBoost" else pd.get_dummies(x, columns=["store_nbr", "family"], dtype=float), y)
    return model


def numeric_features(train_x, val_x, test_x):
    combined = pd.concat([train_x, val_x, test_x], axis=0)
    numeric = pd.get_dummies(combined, columns=["store_nbr", "family"], dtype=float)
    numeric.columns = [f"feature_{i}" for i in range(numeric.shape[1])]
    return numeric.iloc[:len(train_x)], numeric.iloc[len(train_x):len(train_x)+len(val_x)], numeric.iloc[len(train_x)+len(val_x):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("WEEK 3/DAY 1/feature_table_v1.parquet"))
    parser.add_argument("--test", type=Path, default=Path("store-sales-time-series-forecasting/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("WEEK 3/DAY 5"))
    parser.add_argument("--max-rows", type=int, default=150_000)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.features).sort_values("date").reset_index(drop=True)
    train = train.iloc[:min(args.max_rows, len(train))].copy()
    test = add_calendar(pd.read_csv(args.test))
    train = add_calendar(train)
    y = train["sales"].to_numpy(dtype=float)
    x_raw = prepare_x(train, "CatBoost")
    test_raw = prepare_x(test, "CatBoost")
    x_numeric, _, test_numeric = numeric_features(x_raw, x_raw.iloc[:0], test_raw)

    splitter = TimeSeriesSplit(n_splits=args.splits)
    oof = np.full((len(train), len(MODEL_NAMES)), np.nan)
    test_fold_predictions = {name: [] for name in MODEL_NAMES}
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(train), start=1):
        for col, name in enumerate(MODEL_NAMES):
            if name == "CatBoost":
                model = model_factory(name, args.seed + fold)
                model.fit(x_raw.iloc[fit_idx], y[fit_idx], cat_features=CAT_COLS)
                oof[val_idx, col] = model.predict(x_raw.iloc[val_idx])
                test_fold_predictions[name].append(model.predict(test_raw))
            else:
                model = model_factory(name, args.seed + fold)
                model.fit(x_numeric.iloc[fit_idx], y[fit_idx])
                oof[val_idx, col] = model.predict(x_numeric.iloc[val_idx])
                test_fold_predictions[name].append(model.predict(test_numeric))

    valid = ~np.isnan(oof).any(axis=1)
    metrics = []
    for col, name in enumerate(MODEL_NAMES):
        metrics.append({"model": name, "rmse": np.sqrt(mean_squared_error(y[valid], oof[valid, col])), "mae": mean_absolute_error(y[valid], oof[valid, col])})
    metrics_df = pd.DataFrame(metrics)
    weights = 1 / metrics_df["rmse"].to_numpy()
    weights = weights / weights.sum()
    weighted_oof = oof[valid] @ weights
    weighted_test = sum(weights[i] * np.mean(test_fold_predictions[name], axis=0) for i, name in enumerate(MODEL_NAMES))
    ridge = Ridge(alpha=10.0)
    ridge.fit(oof[valid], y[valid])
    ridge_oof = ridge.predict(oof[valid])
    ridge_test = ridge.predict(np.column_stack([np.mean(test_fold_predictions[name], axis=0) for name in MODEL_NAMES]))
    metrics_df = pd.concat([metrics_df, pd.DataFrame([
        {"model": "Weighted_average", "rmse": np.sqrt(mean_squared_error(y[valid], weighted_oof)), "mae": mean_absolute_error(y[valid], weighted_oof)},
        {"model": "Ridge_stack", "rmse": np.sqrt(mean_squared_error(y[valid], ridge_oof)), "mae": mean_absolute_error(y[valid], ridge_oof)},
    ])], ignore_index=True)
    metrics_df.to_csv(args.output_dir / "stacking_cv_comparison.csv", index=False)
    submission = pd.DataFrame({"id": test["id"], "sales": np.maximum(ridge_test, 0)})
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    print(metrics_df.to_string(index=False))
    print(f"Saved {args.output_dir / 'submission.csv'} with {len(submission)} rows")


if __name__ == "__main__":
    main()

