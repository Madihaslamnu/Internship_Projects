Olist Delivery Delay Prediction — Project README
Project objective

Predict whether an order placed on the Olist e-commerce platform will be delivered later than its estimated delivery date, using only information available at the time the order is placed. The goal is to give the business advance warning of likely delays so customer service or logistics teams could act proactively, rather than customers being surprised by a late delivery with no warning.

Dataset description

The Brazilian E-Commerce Public Dataset by Olist, made up of 9 linked CSV tables covering orders, order items, customers, products, sellers, payments, reviews, and geolocation data. After cleaning, the working dataset for modeling was built by joining orders, order items, customers, products, sellers, and reviews into a single table.

Target variable

is_late: a binary flag equal to 1 if the order's actual delivery date was later than its estimated delivery date, and 0 otherwise.

Feature engineering

Three features were used, chosen specifically because they are all known at the moment an order is placed, avoiding the leakage issue described below:

price: the item's price
freight_value: the shipping cost
promised_days: the number of days between purchase and the estimated delivery date (i.e. how many days Olist originally promised for delivery)

An earlier version of the model also included delivery_time_days (actual purchase-to-delivery time), but this was removed after identifying it as a data leakage issue: it's derived from the actual delivery date, which is only known after the order has already arrived, and is therefore not available at the time a real prediction would need to be made.

Preprocessing steps
Missing values were handled per column based on their likely cause (see Week 1 data cleaning notebook): delivery-date columns with missing values were flagged rather than imputed with fabricated dates, review comment fields were filled with constants and flagged, and product listing fields were filled using median values.
Exact duplicate rows were removed (found only in the geolocation table).
Inconsistent city name text was cleaned using lowercasing, whitespace stripping, and fuzzy matching (rapidfuzz) against the most common spelling per zip code.
Date columns were converted from text to proper datetime types; category-like columns were converted to pandas' category dtype.
Rows with missing values in the three modeling features or the target were dropped before training.
Models tested

Two classification models were trained and compared on the same train/test split:

Logistic Regression
Random Forest
Hyperparameter tuning approach

Grid search with 3-fold cross-validation was used on the Random Forest model, optimizing for F1 score (chosen over accuracy because the target classes are imbalanced — most orders are on time). The grid searched over n_estimators (100, 200), max_depth (5, 10, None), and min_samples_leaf (1, 5, 10).

Best parameters

{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 100} — best cross-validation F1 score: 0.189. These are the same values the model already used by default, indicating the grid search did not find an improvement over the baseline configuration within the searched range.

Selected final model

Random Forest, chosen over Logistic Regression because Logistic Regression predicted "on time" for every single test example (0.0 precision and recall on the "late" class), making it useless for the actual goal of catching late deliveries. Random Forest, while still weak, was at least able to identify some late orders.

Classification threshold

0.5 (the default). Given the model's low recall on the minority class, a lower threshold could be explored in future iterations to catch more true positives at the cost of more false alarms, since a missed late delivery is more costly to the business than an unnecessary warning.

Final test performance
Metric	Value
Accuracy	0.9322
Precision	0.4241
Recall	0.1494
F1-score	0.2210
ROC-AUC	0.6628
PR-AUC	0.2138
Brier score	0.0616
Important features

Feature importance (Random Forest): price (0.476), freight_value (0.411), promised_days (0.113). Price and freight_value dominate the model's decisions, while promised_days — which would logically be expected to matter most for predicting a delay — contributes comparatively little. This is discussed further under Limitations.

Known limitations
Weak recall on the minority class: the model misses roughly 85% of orders that are actually late (1,241 missed out of 1,459 in the test set), making it unreliable as a standalone early-warning system in its current form.
Limited feature set: only price, freight value, and promised delivery days were used. Stronger predictors likely exist but weren't available in this iteration — seller-to-customer distance, seller-level historical on-time rate, and seasonal/carrier effects were not included.
Possible confounding in feature importance: price and freight_value dominate the model's decisions, but manual error analysis (see Task 2 error analysis) showed the model performs worse specifically on higher-priced late orders — suggesting price may be acting as a rough proxy for some other factor (e.g. product type or shipping distance) rather than being a genuinely strong direct predictor.
Signs of overfitting: the learning curve shows a large, persistent gap between training F1 (staying around 0.90–0.98) and validation F1 (staying around 0.07–0.19) even as more training data is added, indicating the model is not generalizing well.
Poor probability calibration: the calibration curve shows the model's predicted probabilities are overconfident — for example, when the model predicts a 60–80% chance of lateness, the true rate is closer to 40–45%. Predicted probabilities should not be trusted at face value without recalibration.
Hyperparameter tuning found no improvement: the grid search converged on the same parameters used by default, suggesting either the feature set itself is the bottleneck, or a wider/different hyperparameter search would be needed.
How to reproduce training
Ensure the Poetry environment is installed: poetry install (from the project root, where pyproject.toml lives).
Ensure the Olist dataset CSVs are present in Olist_Data/ and cleaned data has been generated via clean_data.py.
Run the model training and validation script (Task 1 / Task 6 notebook or script) to reproduce the train/test split, model training, hyperparameter search, and evaluation.
The final model is saved to final_model.joblib via joblib.dump().
How to run inference
python
import joblib
import pandas as pd

model = joblib.load("final_model.joblib")

new_orders = pd.DataFrame({
    'price': [50, 300, 15],
    'freight_value': [15, 40, 8],
    'promised_days': [20, 15, 25],
})

probabilities = model.predict_proba(new_orders)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

No manual preprocessing is required beyond providing the three raw numeric features — the model was trained directly on these values without a separate preprocessing pipeline object.

Python / library versions
Python: >3.12,<4.0 (developed and tested on Python 3.14)
pandas: 3.0.5
numpy: 2.5.2
pyarrow: 25.0.1
rapidfuzz: 3.14.6
sqlalchemy: 2.0.52
psycopg2-binary: 2.9.12
seaborn: 0.13.2
plotly: 7.0.0
matplotlib: 3.11.1
joblib: 1.6.0
scikit-learn: 1.9.0
jupyter: 1.1.1
ipykernel: 7.3.0

Full dependency list with exact pinned versions is available in poetry.lock.