# Week 3 Day 3 - Time-series-specific features

The pipeline sorts each `store_nbr`/`family` series by date, then creates
`sales_lag_1`, `sales_lag_7`, and `sales_lag_28` with a grouped `shift`. Rolling
7-day and 28-day means and standard deviations are calculated from
`groupby(...).shift(1)` before rolling, so the current observation is excluded.
At prediction time for date D, no feature may use any data from date >= D.

Naive lag creation leaks when a global shift crosses from one store/product
series into another, because the previous row may belong to a different series.
A rolling calculation without `shift(1)` also includes the current day's sales
in its own average and standard deviation. Both mistakes make validation look
better than real forecasting because they expose information unavailable at
prediction time.

The generated `leakage_trace.txt` manually traces one store/family/date row and
records that every source date is strictly earlier than the prediction date.
The seasonal decomposition uses a seven-day period on store 1 / GROCERY I;
`seasonal_decomposition.png` contains trend, weekly seasonal, and residual
components. Remaining residual spikes indicate effects such as promotions and
holidays that decomposition alone does not explain.
