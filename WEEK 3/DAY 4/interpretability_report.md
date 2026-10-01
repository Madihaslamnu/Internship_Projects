# Week 3 Day 4 - Interpretability report

## Model and data used

The analysis uses the best model from Day 2: CatBoost trained on the raw
`store_nbr` and `family` categoricals plus `year`, `month`, `day_of_week`,
`is_holiday`, and `onpromotion`. The split is chronological: the first 80% of
dates are training data and the last 20% are validation data. As in Day 2, the
run uses a reproducible sample of 200,000 training rows and 50,000 validation
rows.

## Global SHAP findings

The global summary is in `shap_summary.png`. Mean absolute SHAP importance ranks
features as follows:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | `onpromotion` | 511.21 |
| 2 | `family` | 250.45 |
| 3 | `store_nbr` | 212.01 |
| 4 | `year` | 138.29 |
| 5 | `day_of_week` | 114.71 |
| 6 | `month` | 59.71 |
| 7 | `is_holiday` | 2.87 |

Promotion intensity is the strongest driver, which is domain-plausible because
promotions can create large sales changes. Family and store are also sensible:
different products have very different baseline demand, and stores have their
own scale and local demand patterns. The SHAP color/direction in the summary
plot should be read together with the raw categorical values: CatBoost is
handling those categories natively rather than treating their numeric labels as
ordered quantities.

## Three individual predictions

The three force plots are saved as interactive HTML files. The exact rows are described below.

1. **Typical low-error row** - Store 53, `MAGAZINES`, 2016-12-18. Actual sales
   were 10.00 and predicted sales were 9.99, for an absolute error of 0.01.
   The model's learned store/family baseline and the low-demand context combine
to keep the prediction near zero; no unusually large promotion signal pushed it
upward.

2. **Highest-error row** - Store 46, `GROCERY I`, 2017-04-01. Actual sales were
   24,394.00 but the prediction was 6,772.60, underestimating by 17,621.40.
   This is a useful failure case: the model recognizes the general family and
   store effects, but the available features do not fully explain an unusually
   large event-level spike. The missing signal could be a promotion detail,
   holiday/event effect, transaction volume, or another store-specific shock.

3. **Highest-prediction row** - Store 3, `PRODUCE`, 2016-12-14. Actual sales
   were 10,330.76 and predicted sales were 13,825.62. The strong product-family
   and store effects, together with the promotion/calendar context, pushed the
   estimate high; the model overshot this particular observation by 3,494.86.

## Partial dependence and sanity check

`partial_dependence_top3.png` contains scikit-learn partial-dependence plots
for the three highest-SHAP features: `onpromotion`, `family`, and `store_nbr`.
Promotion is the only numeric top-three feature, so its curve is the clearest
monotonicity check. The categorical family and store plots should be interpreted
as differences between learned category averages, not as ordered numeric
curves.

## Suspicious findings

No obvious leakage red flag was found. `family` and `store_nbr` are not random
IDs in this problem: they identify real demand segments and locations, so their
high importance is expected. `onpromotion` being the top feature is also
consistent with the business setting. `is_holiday` has very low global SHAP
importance (2.87), so the model is not relying on the holiday flag as a hidden
shortcut; holiday effects may be sparse or already partly represented by the
calendar/category features. The high-error example does show a modeling gap,
not evidence of leakage: the current feature set misses some event and
store-specific demand shocks.


