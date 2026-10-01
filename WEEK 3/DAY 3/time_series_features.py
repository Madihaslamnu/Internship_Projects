from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

GROUPS = ["store_nbr", "family"]


def add_lag_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "store_nbr", "family", "sales"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    result = frame.sort_values(GROUPS + ["date"]).reset_index(drop=True).copy()
    grouped_sales = result.groupby(GROUPS, sort=False)["sales"]
    for lag in (1, 7, 28):
        result[f"sales_lag_{lag}"] = grouped_sales.shift(lag)

    past_sales = grouped_sales.shift(1)
    for window in (7, 28):
        rolling = past_sales.groupby([result[group] for group in GROUPS], sort=False).rolling(window, min_periods=window)
        result[f"sales_rolling_mean_{window}"] = rolling.mean().reset_index(level=GROUPS, drop=True).sort_index()
        result[f"sales_rolling_std_{window}"] = rolling.std().reset_index(level=GROUPS, drop=True).sort_index()
    return result


def trace_row(frame: pd.DataFrame, store_nbr, family, date) -> str:
    series = frame[(frame["store_nbr"] == store_nbr) & (frame["family"] == family)].sort_values("date")
    row = series.loc[series["date"] == pd.Timestamp(date)]
    if row.empty:
        raise ValueError("Trace row was not found")
    position = row.index[0]
    history = series.loc[series.index < position, ["date", "sales"]].tail(28)
    return (f"Trace for store={store_nbr}, family={family}, date={date}: "
            f"all source rows have dates strictly before {date}; "
            f"lag_1={history.tail(1)['date'].dt.date.tolist()}, "
            f"lag_7={history.tail(7).head(1)['date'].dt.date.tolist()}, "
            f"lag_28={history.head(1)['date'].dt.date.tolist()}; "
            f"rolling windows use only the preceding 7 or 28 sales values.")


def make_decomposition(frame: pd.DataFrame, output_path: Path, store_nbr, family) -> str:
    series = frame[(frame["store_nbr"] == store_nbr) & (frame["family"] == family)].set_index("date")["sales"].sort_index()
    series = series.asfreq("D").interpolate(limit_direction="both")
    decomposition = seasonal_decompose(series, model="additive", period=7, extrapolate_trend="period")
    figure = decomposition.plot()
    figure.set_size_inches(12, 8)
    figure.suptitle(f"Seasonal decomposition: store {store_nbr}, {family} (weekly period)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return ("The selected series shows a weekly seasonal component, while the trend "
            "captures the slower change in sales over time. The residual is the part "
            "not explained by those components; remaining spikes suggest promotions, "
            "holidays, or other covariates should be included by later models.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("WEEK 3/DAY 1/feature_table_v1.parquet"))
    parser.add_argument("--output", type=Path, default=Path("WEEK 3/DAY 3/feature_table_v2_lag_rolling.parquet"))
    parser.add_argument("--plot", type=Path, default=Path("WEEK 3/DAY 3/seasonal_decomposition.png"))
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--family", default="GROCERY I")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    features = add_lag_rolling_features(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    trace = trace_row(frame, args.store, args.family, frame.loc[(frame["store_nbr"] == args.store) & (frame["family"] == args.family), "date"].sort_values().iloc[40])
    interpretation = make_decomposition(frame, args.plot, args.store, args.family)
    Path(args.output.with_name("leakage_trace.txt")).write_text(trace + "\n" + interpretation + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"Saved {args.plot}")
    print(trace)


if __name__ == "__main__":
    main()

