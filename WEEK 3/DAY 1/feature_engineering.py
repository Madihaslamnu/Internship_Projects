from __future__ import annotations
import argparse
from pathlib import Path
import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_DICTIONARY = [
    ("date", "Observation date", "date", "datetime"),
    ("store_nbr", "Store identifier retained for traceability", "store_nbr", "raw categorical"),
    ("family", "Product family retained for traceability", "family", "raw categorical"),
    ("store_nbr_frequency", "Share of training rows belonging to this store", "store_nbr", "training-only frequency encoding"),
    ("family_frequency", "Share of training rows belonging to this family", "family", "training-only frequency encoding"),
    ("store_family_frequency", "Share of training rows for this store-family pair", "store_nbr, family", "training-only frequency encoding"),
    ("year", "Calendar year", "date", "date decomposition"),
    ("month", "Calendar month", "date", "date decomposition"),
    ("day_of_week", "Monday=0 through Sunday=6", "date", "date decomposition"),
    ("day_of_month", "Day within the month", "date", "date decomposition"),
    ("week_of_year", "ISO week number", "date", "date decomposition"),
    ("is_weekend", "Whether the date is Saturday or Sunday", "date", "date decomposition"),
    ("is_month_start", "Whether the date is the first day of its month", "date", "date decomposition"),
    ("is_month_end", "Whether the date is the last day of its month", "date", "date decomposition"),
    ("is_holiday", "Whether the date is an Ecuador public holiday", "date", "holidays.Ecuador"),
    ("holiday_name", "Ecuador holiday name, or empty string", "date", "holidays.Ecuador"),
    ("holiday_x_day_of_week", "Holiday/week-day interaction", "is_holiday, day_of_week", "multiplication"),
    ("onpromotion", "Number of products on promotion", "onpromotion", "raw numeric"),
    ("onpromotion_sq", "Squared promotion count for diminishing/increasing effects", "onpromotion", "second-degree polynomial"),
    ("sales", "Forecast target", "sales", "raw target"),
]


def _frequency_map(values: pd.Series) -> dict[object, float]:
    return values.value_counts(normalize=True, dropna=False).to_dict()


def build_features(
    input_path: Path,
    output_path: Path,
    dictionary_path: Path,
    diagnostic_plot_path: Path | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(input_path, parse_dates=["date"])
    required = {"date", "store_nbr", "family", "onpromotion", "sales"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    raw = raw.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    cutoff = raw["date"].quantile(0.8)
    train_mask = raw["date"] <= cutoff
    result = raw[["date", "store_nbr", "family", "onpromotion", "sales"]].copy()
    result["onpromotion"] = result["onpromotion"].fillna(0).astype(float)

    pair = raw["store_nbr"].astype(str) + "::" + raw["family"].astype(str)
    for name, values in (
        ("store_nbr", raw["store_nbr"]),
        ("family", raw["family"]),
        ("store_family", pair),
    ):
        mapping = _frequency_map(values[train_mask])
        result[f"{name}_frequency"] = values.map(mapping).fillna(0.0).astype(float)

    dates = result["date"]
    result["year"] = dates.dt.year.astype("int16")
    result["month"] = dates.dt.month.astype("int8")
    result["day_of_week"] = dates.dt.dayofweek.astype("int8")
    result["day_of_month"] = dates.dt.day.astype("int8")
    result["week_of_year"] = dates.dt.isocalendar().week.astype("int8")
    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype("int8")
    result["is_month_start"] = dates.dt.is_month_start.astype("int8")
    result["is_month_end"] = dates.dt.is_month_end.astype("int8")

    ec_holidays = holidays.country_holidays("EC", years=range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1))
    result["holiday_name"] = dates.dt.date.map(lambda day: ec_holidays.get(day, ""))
    result["is_holiday"] = result["holiday_name"].ne("").astype("int8")
    result["holiday_x_day_of_week"] = result["is_holiday"] * result["day_of_week"]
    result["onpromotion_sq"] = np.square(result["onpromotion"])

    if diagnostic_plot_path is not None:
        diagnostic_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_data = raw.loc[train_mask, ["onpromotion", "sales"]].dropna()
        if len(plot_data) > 50_000:
            plot_data = plot_data.sample(50_000, random_state=42)
        grouped = (
            raw.loc[train_mask, ["onpromotion", "sales"]]
            .groupby("onpromotion", as_index=False)["sales"]
            .mean()
        )
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.scatter(plot_data["onpromotion"], plot_data["sales"], alpha=0.12, s=6, label="training observations")
        axis.plot(grouped["onpromotion"], grouped["sales"], color="crimson", linewidth=2, label="mean sales by promotion count")
        axis.set(title="Promotion count vs. sales", xlabel="onpromotion", ylabel="sales")
        axis.legend()
        figure.tight_layout()
        figure.savefig(diagnostic_plot_path, dpi=150)
        plt.close(figure)

    result.to_parquet(output_path, index=False)
    pd.DataFrame(FEATURE_DICTIONARY, columns=["feature_name", "description", "source_columns", "encoding_method"]).to_csv(
        dictionary_path, index=False
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to Kaggle train.csv")
    parser.add_argument("--output", type=Path, default=Path("feature_table_v1.parquet"))
    parser.add_argument("--dictionary", type=Path, default=Path("feature_dictionary.csv"))
    parser.add_argument(
        "--diagnostic-plot",
        type=Path,
        default=None,
        help="Optional path for the promotion/sales scatter diagnostic",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dictionary.parent.mkdir(parents=True, exist_ok=True)
    build_features(args.input, args.output, args.dictionary, args.diagnostic_plot)
    print(f"Saved {args.output}")
    print(f"Saved {args.dictionary}")


if __name__ == "__main__":
    main()
