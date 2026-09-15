# Day 3 - Ensemble methods comparison


## Results

| Model | RMSE | MAE | Training time (s) | Peak memory (MB) |
|---|---:|---:|---:|---:|
| Random Forest | 1264.75 | 575.08 | 18.86 | 684.75 |
| XGBoost | 1228.24 | 550.48 | 2.85 | 657.38 |
| LightGBM | 1121.60 | 523.82 | 1.83 | 613.17 |
| CatBoost encoded | 1168.73 | 533.44 | 3.64 | 609.86 |
| CatBoost raw categoricals | **972.78** | **419.36** | 10.49 | 684.39 |

CatBoost with raw `store_nbr` and `family` categoricals performed best in this
run. LightGBM was the fastest boosting model and used the least memory among
the main encoded-feature comparisons.

## Bagging versus boosting

Random Forest is a bagging method. It fits many decision trees independently,
using bootstrapped samples and random subsets of features, then averages their
predictions. Because the trees do not depend on one another, they can be trained
in parallel. Averaging makes the result less sensitive to noisy observations and
usually reduces variance, so Random Forest is a strong, low-maintenance
baseline.

XGBoost, LightGBM, and CatBoost are boosting methods. They build trees in
sequence rather than independently. Each new tree focuses more on the errors
left by the current ensemble, so later trees gradually improve the combined
prediction. This bias-correcting process often gives boosting the edge on
structured tabular data, especially when the features contain useful
interactions.

The trade-off is control. Boosting can overfit if it is allowed too many trees
or overly complex trees, so learning rate, tree depth, regularization, and
early stopping matter. Bagging is generally more forgiving and easier to
parallelize, while boosting usually reaches better accuracy when tuned with a
proper validation strategy.

`CatBoost_raw` is included separately because it receives the original
categorical columns directly; the other primary comparisons use the numeric
features produced on Day 1.
