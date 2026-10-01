## Model comparison and stack result

The OOF run used 150,000 chronological training rows and three expanding
`TimeSeriesSplit` folds. The validation metrics were:

| Model | OOF RMSE | OOF MAE |
|---|---:|---:|
| Random Forest | 195.90 | 41.35 |
| XGBoost | 206.70 | 71.07 |
| LightGBM | 224.55 | 81.47 |
| CatBoost | 589.94 | 208.68 |
| Inverse-RMSE weighted average | 194.00 | 60.13 |
| Ridge stack | **183.42** | **42.99** |

The Ridge meta-learner was strongest on this chronological OOF evaluation. It
can learn that the base models make different errors instead of assigning one
fixed manual weight to every row.

## Leakage safety

The Day 1 categorical encodings were fitted using an earlier chronological
portion, and Day 3's lag/rolling pipeline uses grouped history per
`store_nbr`/`family`. For this stack, the fold splitter is chronological and
base models are fitted only on the earlier fold. OOF predictions are therefore
made on rows that were not used to train that fold's base model. The Ridge
model is trained only after all OOF predictions exist. This is the important
stacking rule: the meta-learner never receives in-sample base predictions.

## SHAP interpretability carried forward

Day 4's SHAP analysis found `onpromotion`, `family`, and `store_nbr` were the
largest drivers for the best Day 2 CatBoost model. Those findings remain
consistent with the stack: promotions, product family, and store identity are
real demand signals rather than random identifiers. The Day 4 report found no
obvious leakage red flag, and the Day 3 trace confirmed that lag and rolling
features use only dates before the prediction date.

## Submission and CV-vs-leaderboard gap

`submission.csv` contains 28,512 rows and exactly the required columns: `id`
and `sales`. Predictions are clipped at zero because sales cannot be negative.
The file was submitted successfully to Kaggle and received a public score of
**2.04443**.

The local OOF report uses RMSE and MAE, while the Store Sales competition uses
NWRMSLE (normalized weighted root mean squared logarithmic error). Therefore,
`2.04443 - 183.42` is not a meaningful metric gap: the units and definitions
differ. The submission confirms that the file format and prediction handoff
were accepted, but a valid CV-versus-leaderboard comparison requires computing
NWRMSLE on the same chronological OOF predictions used for the stack.

The leaderboard result should be treated as a useful warning about metric
alignment rather than automatically as evidence of leakage. The competition
test period is later than the training period, so distribution shift in
seasonality, promotions, holidays, and store demand is plausible. The local
validation protocol is leakage-aware because it uses chronological folds and
out-of-fold base predictions, but the current report did not optimize or record
NWRMSLE. A next iteration should train on `log1p(sales)` or evaluate NWRMSLE
directly, and should include the Day 3 lag/rolling features in the stack.
