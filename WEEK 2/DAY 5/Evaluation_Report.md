# Day 5 - Tuning and Packaging

Logistic Regression was selected because it had the strongest Day 4 baseline
ROC-AUC and PR-AUC. Optuna searched 30 parameter combinations using 5-fold
stratified cross-validation and F1 as the objective. The search varied `C`,
`penalty`, and `class_weight`; the best parameters and F1 score are printed by
`tune_model.py`.

The optimization history is saved as `optimization_history.png`. The final
pipeline is fitted on all available Telco data and saved as
`tuned_pipeline.joblib`.

`predict.py` exposes `predict_churn(customer_dict)`. It validates required
fields, types, and sensible ranges before prediction. It returns the churn
probability and a risk tier. The Day 4 threshold of 0.30 is used as the
business cutoff: below 0.30 is Low risk, 0.30 to below 0.60 is Medium risk,
and 0.60 or higher is High risk.

The script's deliberate negative-tenure test confirms that invalid input
produces a clear validation error.
