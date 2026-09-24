from __future__ import annotations
import argparse
import time
import threading
import os
import math
from pathlib import Path

import psutil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


def measure_fit(model, fit_fn, *args, **kwargs):
    """Run fit_fn(model, *args, **kwargs) while recording peak RSS (MB) and elapsed time."""
    proc = psutil.Process(os.getpid())
    peak = 0
    start = time.perf_counter()
    finished = threading.Event()

    def runner():
        try:
            fit_fn(*args, **kwargs)
        finally:
            finished.set()

    t = threading.Thread(target=runner)
    t.start()
    while not finished.is_set():
        try:
            rss = proc.memory_info().rss
            if rss > peak:
                peak = rss
        except Exception:
            pass
        time.sleep(0.2)
    t.join()
    elapsed = time.perf_counter() - start
    return elapsed, peak / (1024 * 1024)


def train_and_eval(model_name, model, X_train, y_train, X_val, y_val, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}

    def fit_fn():
        model.fit(X_train, y_train, **({} if not fit_kwargs else fit_kwargs))

    elapsed, peak_mb = measure_fit(model, fit_fn)
    preds = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    return dict(model=model_name, rmse=float(rmse), mae=float(mae), train_time_s=float(elapsed), peak_memory_mb=float(peak_mb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("WEEK 3/DAY 1/feature_table_v1.parquet"))
    parser.add_argument("--output", type=Path, default=Path("WEEK 3/DAY 2/model_comparison.csv"))
    parser.add_argument("--max-train-rows", type=int, default=200_000, help="Max number of training rows (train portion)")
    parser.add_argument("--val-rows", type=int, default=50_000, help="Max number of validation rows (validation portion)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    # consistent split: earliest 80% dates as training portion, rest as validation
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = df["date"].quantile(0.8)
    train_df = df[df["date"] <= cutoff].copy()
    val_df = df[df["date"] > cutoff].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Train/val split empty — check input file and date column")

    # Subsample for speed but keep reproducible
    train_df = train_df.sample(n=min(args.max_train_rows, len(train_df)), random_state=args.random_state)
    val_df = val_df.sample(n=min(args.val_rows, len(val_df)), random_state=args.random_state)

    target = "sales"
    # Features chosen: frequency encodings, date decompositions, holiday flags, promotion
    encoded_features = [
        "store_nbr_frequency",
        "family_frequency",
        "store_family_frequency",
        "year",
        "month",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_holiday",
        "holiday_x_day_of_week",
        "onpromotion",
        "onpromotion_sq",
    ]

    # Prepare numeric X for RF/XGB/LGB
    X_train_enc = train_df[encoded_features].fillna(0).astype(float)
    X_val_enc = val_df[encoded_features].fillna(0).astype(float)
    y_train = train_df[target].values
    y_val = val_df[target].values

    results = []

    # Random Forest baseline
    print("Training Random Forest")
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=args.random_state)
    results.append(train_and_eval("RandomForest", rf, X_train_enc, y_train, X_val_enc, y_val))

    # XGBoost
    if xgb is not None:
        print("Training XGBoost")
        xgb_reg = xgb.XGBRegressor(n_estimators=100, tree_method="hist", random_state=args.random_state, n_jobs=4)
        results.append(train_and_eval("XGBoost", xgb_reg, X_train_enc, y_train, X_val_enc, y_val))
    else:
        print("xgboost not installed — skipping")

    # LightGBM
    if lgb is not None:
        print("Training LightGBM")
        lgb_reg = lgb.LGBMRegressor(n_estimators=100, random_state=args.random_state, n_jobs=4)
        results.append(train_and_eval("LightGBM", lgb_reg, X_train_enc, y_train, X_val_enc, y_val))
    else:
        print("lightgbm not installed — skipping")

    # CatBoost on encoded features
    if CatBoostRegressor is not None:
        print("Training CatBoost (encoded features)")
        cat_enc = CatBoostRegressor(iterations=100, verbose=0, random_state=args.random_state)
        results.append(train_and_eval("CatBoost_encoded", cat_enc, X_train_enc, y_train, X_val_enc, y_val))

        # CatBoost on raw categoricals: use raw store_nbr and family
        print("Training CatBoost (raw categoricals)")
        raw_features = ["store_nbr", "family", "year", "month", "day_of_week", "is_holiday", "onpromotion"]
        X_train_raw = train_df[raw_features].copy()
        X_val_raw = val_df[raw_features].copy()
        # Convert store_nbr to string to be categorical
        X_train_raw["store_nbr"] = X_train_raw["store_nbr"].astype(str)
        X_val_raw["store_nbr"] = X_val_raw["store_nbr"].astype(str)
        # CatBoost can accept object/string columns as categorical; provide column indices
        cat_cols = [0, 1]  # store_nbr, family
        cat_raw = CatBoostRegressor(iterations=100, verbose=0, random_state=args.random_state)
        results.append(train_and_eval("CatBoost_raw", cat_raw, X_train_raw, y_train, X_val_raw, y_val, fit_kwargs={"cat_features": cat_cols}))
    else:
        print("catboost not installed — skipping CatBoost")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df = res_df[["model", "rmse", "mae", "train_time_s", "peak_memory_mb"]]
    res_df.to_csv(args.output, index=False)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
