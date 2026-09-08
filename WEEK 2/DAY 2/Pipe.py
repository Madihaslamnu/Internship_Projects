import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from category_encoders import TargetEncoder


print(" FEATURE ENGINEERING & PREPROCESSING PIPELINE")

print("\nLoading and Cleaning Data")

file_path = r'C:\Users\CZ 3\OneDrive\Desktop\Internship_Project\cleaned_data\Telco-Customer-Churn.csv'

df = pd.read_csv(file_path)

print(f"Original dataset shape: {df.shape}")

# Convert TotalCharges from string to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

print(
    f"Missing TotalCharges after numeric conversion: "
    f"{df['TotalCharges'].isna().sum()}"
)

# Separate features and target
X = df.drop('Churn', axis=1)

y = df['Churn'].apply(
    lambda x: 1 if x == 'Yes' else 0
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

print("\nData leakage prevention:")
print(
    "TotalCharges median imputation will be performed INSIDE "
    "the pipeline so that the median is learned only from training data."
)

print("CUSTOM FEATURE ENGINEERING")

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X['tenure_bucket'] = pd.cut(
            X['tenure'],
            bins=[-1, 12, 24, 100],
            labels=[
                '0-12mo',
                '13-24mo',
                '25mo+'
            ]
        ).astype(str)

        service_cols = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]

        valid_services = [
            col for col in service_cols
            if col in X.columns
        ]

        if valid_services:
            X['total_services'] = (
                X[valid_services] == 'Yes'
            ).sum(axis=1)
        else:
            X['total_services'] = 0


        X['charges_per_tenure'] = np.where(
            X['tenure'] == 0,
            X['MonthlyCharges'],
            X['TotalCharges'] / X['tenure']
        )

        return X


feature_engineer = TelcoFeatureEngineer()

X_train_engineered = feature_engineer.fit_transform(X_train)

print("\nEngineered Features Created:")
print("1. tenure_bucket")
print("   Groups customers into 0-12mo, 13-24mo and 25mo+.")
print("   Purpose: capture possible non-linear churn patterns.")

print("\n2. total_services")
print("   Counts the number of subscribed add-on services.")
print("   Services counted:")
print("   OnlineSecurity, OnlineBackup, DeviceProtection,")
print("   TechSupport, StreamingTV and StreamingMovies.")
print("   Purpose: represent overall service adoption.")

print("\n3. charges_per_tenure")
print("   Calculates TotalCharges / tenure.")
print("   Purpose: represent average accumulated charge per month.")

print("\nTenure = 0 handling:")
print(
    "For customers with tenure = 0, charges_per_tenure "
    "uses MonthlyCharges instead of dividing by zero."
)


print("CATEGORICAL CARDINALITY ANALYSIS")

categorical_cols_original = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

print("\nUnique values in categorical columns:")

for col in categorical_cols_original:
    print(
        f"{col:20} -> "
        f"{X_train[col].nunique()} unique values"
    )

print("ENCODING & SCALING ARCHITECTURE")


numeric_cols = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'charges_per_tenure',
    'total_services'
]


low_card_cols = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'tenure_bucket'
]

high_card_cols = [
    'PaymentMethod'
]


print("\nEncoding decisions:")

print(
    "\nLOW-CARDINALITY CATEGORICAL FEATURES:"
)
print(
    "OneHotEncoder is used because these columns contain "
    "only a small number of categories."
)

print(
    "\nTARGET-ENCODED FEATURE:"
)
print(
    "PaymentMethod is technically low-cardinality (4 categories) "
    "in this Telco dataset."
)
print(
    "It is assigned to TargetEncoder specifically for practice."
)

print(
    "\nTARGET ENCODING LEAKAGE RISK:"
)
print(
    "TargetEncoder replaces each category with a statistic based "
    "on the target variable, such as the average churn rate."
)
print(
    "If TargetEncoder is fitted on the complete dataset before "
    "the train/test split, test-label information can leak into "
    "the feature values."
)
print(
    "This can make validation performance look artificially good."
)
print(
    "Therefore TargetEncoder must be fitted only on training "
    "data/folds."
)
print(
    "Because TargetEncoder is inside the sklearn Pipeline, it is "
    "fitted only when the training data is passed to .fit()."
)


print(
    "\nNUMERIC SCALING:"
)
print(
    "StandardScaler is used because Logistic Regression and "
    "k-NN are sensitive to feature scale."
)
print(
    "For example, TotalCharges can have values in the thousands, "
    "while binary features are represented as 0/1."
)
print(
    "Without scaling, large-valued features can have an "
    "unfair influence on distance or optimization."
)

print(
    "\nTREE-BASED MODELS:"
)
print(
    "Decision Trees and Gradient Boosting do not require "
    "feature scaling."
)
print(
    "Trees make threshold-based splits rather than calculating "
    "distances between observations."
)


numeric_pipeline = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='median')
    ),
    (
        'scaler',
        StandardScaler()
    )
])

preprocessor = ColumnTransformer(
    transformers=[

        (
            'num',
            numeric_pipeline,
            numeric_cols
        ),

        (
            'onehot',
            OneHotEncoder(
                drop='if_binary',
                handle_unknown='ignore'
            ),
            low_card_cols
        ),

        (
            'target',
            TargetEncoder(
                cols=high_card_cols
            ),
            high_card_cols
        )
    ]
)

print("BUILDING COMPLETE PIPELINE")


full_pipeline = Pipeline(
    steps=[
        (
            'feature_engineering',
            TelcoFeatureEngineer()
        ),

        (
            'preprocessing',
            preprocessor
        )
    ]
)


X_train_ready = full_pipeline.fit_transform(
    X_train,
    y_train
)

X_test_ready = full_pipeline.transform(
    X_test
)


feature_names = (
    full_pipeline
    .named_steps['preprocessing']
    .get_feature_names_out()
)


print(
    f"\nPipeline built successfully!"
)

print(
    f"Processed training shape: {X_train_ready.shape}"
)

print(
    f"Processed test shape:     {X_test_ready.shape}"
)

print(
    f"Total processed features: {len(feature_names)}"
)

print(
    "\nPipeline order:"
)

print(
    "Raw Data"
    " -> Feature Engineering"
    " -> ColumnTransformer"
    " -> Scaled/Encoded Features"
)


print("CORRELATION FILTER")


temp_numeric_df = X_train[
    [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]
].copy()


temp_numeric_df['charges_per_tenure'] = np.where(
    X_train['tenure'] == 0,
    X_train['MonthlyCharges'],
    X_train['TotalCharges'] / X_train['tenure']
)


temp_numeric_df['total_services'] = (
    X_train[
        [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]
    ] == 'Yes'
).sum(axis=1)


corr_matrix = temp_numeric_df.corr().abs()


print("\nAbsolute Numeric Feature Correlation Matrix:")
print(
    corr_matrix.round(3)
)

threshold = 0.9

high_corr_pairs = []

columns = corr_matrix.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        correlation = corr_matrix.iloc[i, j]

        if correlation > threshold:

            high_corr_pairs.append(
                (
                    columns[i],
                    columns[j],
                    correlation
                )
            )


print(
    f"\nHighly correlated pairs "
    f"(correlation > {threshold}):"
)


if high_corr_pairs:

    for feature1, feature2, correlation in high_corr_pairs:

        print(
            f"{feature1} <-> {feature2}: "
            f"{correlation:.3f}"
        )

else:

    print(
        "No feature pairs exceeded the correlation threshold."
    )


print("\nCorrelation Decision:")

if high_corr_pairs:

    print(
        "Highly correlated features have been FLAGGED rather "
        "than automatically dropped."
    )

    print(
        "This is intentional because correlation alone does "
        "not prove that one feature should always be removed."
    )

    print(
        "The flagged features can be reviewed before final "
        "model selection."
    )

else:

    print(
        "No features require correlation-based removal."
    )


print(
    "\nImportant observation:"
)

print(
    "MonthlyCharges and charges_per_tenure are expected to be "
    "highly correlated because charges_per_tenure is approximately "
    "a reconstruction of MonthlyCharges."
)


print("SELECT KBEST FEATURE SELECTION")

skb = SelectKBest(
    score_func=f_classif,
    k=10
)


skb.fit(
    X_train_ready,
    y_train
)


skb_scores = pd.Series(
    skb.scores_,
    index=feature_names
)


skb_scores = skb_scores.sort_values(
    ascending=False
)


print(
    "\nTop 10 features according to f_classif:"
)

print(
    skb_scores.head(10)
)


selected_mask = skb.get_support()

selected_features = feature_names[
    selected_mask
]


print(
    "\nFeatures selected by SelectKBest:"
)

for feature in selected_features:

    print(
        f"  {feature}"
    )


print(
    "\nWhat f_classif tells us:"
)

print(
    "f_classif measures whether the feature has a statistically "
    "significant linear/mean-based relationship with the class label."
)

print(
    "A higher F-score indicates a stronger relationship with churn."
)

print("MUTUAL INFORMATION")



mi_scores = mutual_info_classif(
    X_train_ready,
    y_train,
    random_state=42
)


mi_series = pd.Series(
    mi_scores,
    index=feature_names
)


mi_series = mi_series.sort_values(
    ascending=False
)


print(
    "\nTop 10 features according to Mutual Information:"
)

print(
    mi_series.head(10)
)


print(
    "\nWhat Mutual Information tells us:"
)

print(
    "Mutual information measures how much information a feature "
    "provides about the target."
)

print(
    "Unlike f_classif, it can detect more general and potentially "
    "non-linear relationships."
)


print("FEATURE RANKING COMPARISON")


comparison = pd.DataFrame({

    'f_classif_score': pd.Series(
        skb_scores,
        index=feature_names
    ),

    'f_classif_rank': pd.Series(
        skb_scores,
        index=feature_names
    ).rank(
        ascending=False
    ),

    'mi_score': pd.Series(
        mi_series,
        index=feature_names
    ),

    'mutual_info_rank': pd.Series(
        mi_series,
        index=feature_names
    ).rank(
        ascending=False
    )
})


comparison['rank_difference'] = (
    comparison['f_classif_rank']
    -
    comparison['mutual_info_rank']
).abs()


print(
    "\nTop 10 features by Mutual Information:"
)

print(
    comparison
    .sort_values(
        'mi_score',
        ascending=False
    )
    .head(10)
    [
        [
            'mi_score',
            'mutual_info_rank'
        ]
    ]
)


print(
    "\nLargest ranking disagreements:"
)

largest_disagreements = (
    comparison
    .sort_values(
        'rank_difference',
        ascending=False
    )
    .head(5)
)


print(
    largest_disagreements[
        [
            'f_classif_rank',
            'mutual_info_rank',
            'rank_difference'
        ]
    ]
)

print(
    "\nInterpretation of ranking differences:"
)

for feature, row in largest_disagreements.iterrows():

    print(
        f"\n{feature}:"
    )

    print(
        f"  f_classif rank = "
        f"{int(row['f_classif_rank'])}"
    )

    print(
        f"  Mutual Information rank = "
        f"{int(row['mutual_info_rank'])}"
    )

    print(
        f"  Difference = "
        f"{int(row['rank_difference'])}"
    )


print(
    "\nWhy can the rankings disagree?"
)

print(
    "f_classif focuses on a statistical relationship based on "
    "differences between class means."
)

print(
    "Mutual information can capture broader dependencies, "
    "including non-linear relationships."
)

print(
    "Therefore, a feature may receive a relatively low f_classif "
    "ranking but a higher mutual-information ranking."
)

print("ACTUAL RESULTS INTERPRETATION")


print(
    "\nBased on the current run:"
)

print(
    "• Contract_Month-to-month is the strongest feature "
    "according to Mutual Information."
)

print(
    "• tenure is also highly informative about churn."
)

print(
    "• TechSupport_No and OnlineSecurity_No rank highly, "
    "suggesting service-related behavior is associated with churn."
)

print(
    "• Contract_Two year also has a strong relationship with churn."
)

print(
    "• InternetService_Fiber optic appears among the strongest "
    "features."
)

print(
    "• MonthlyCharges ranks 7th by Mutual Information but "
    "21st by f_classif."
)

print(
    "This large difference suggests that the relationship between "
    "MonthlyCharges and churn may not be fully captured by the "
    "linear/mean-based f_classif test."
)




print(
    "\nTrain/test transformation:"
)

print(
    f"Training rows transformed: {X_train_ready.shape[0]}"
)

print(
    f"Testing rows transformed:  {X_test_ready.shape[0]}"
)

print(
    f"Same number of processed features: "
    f"{X_train_ready.shape[1] == X_test_ready.shape[1]}"
)




pipeline_path = 'pipeline.joblib'


joblib.dump(
    full_pipeline,
    pipeline_path
)


print(
    f"\nPipeline object saved successfully as: "
    f"{pipeline_path}"
)


print("\nTesting saved pipeline.")

loaded_pipeline = joblib.load(
    pipeline_path
)


X_test_loaded = loaded_pipeline.transform(
    X_test
)


if np.allclose(
    X_test_ready,
    X_test_loaded
):

    print(
        "Saved pipeline successfully reloaded."
    )

    print(
        "Reloaded pipeline produces the same transformed output."
    )

else:

    print(
        "WARNING: Reloaded pipeline output differs."
    )

