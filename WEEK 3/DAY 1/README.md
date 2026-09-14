## Leakage and feature choices

The script uses the earliest 80% of dates as its training portion. Frequency
maps are fitted only on that portion; unseen stores, families, and pairs in
later dates receive `0.0`. This prevents future category frequencies from
leaking into the model.

The `store_family_frequency` feature represents the store-by-family
interaction without creating a large one-hot matrix. `holiday_x_day_of_week`
captures different holiday effects across weekdays.

`onpromotion_sq` is the only polynomial feature. The generated
`onpromotion_sales_scatter.png` shows a sample of training observations and
the mean sales at each promotion count. Its non-linear, saturating trend
supports including this single squared promotion feature rather than
expanding every numeric column.
