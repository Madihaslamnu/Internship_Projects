from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\CZ 3\OneDrive\Desktop\Internship_Project\WEEK 2\DAY 2')
from Pipe import TelcoFeatureEngineer, preprocessor, full_pipeline

file_path = r'C:\Users\CZ 3\OneDrive\Desktop\Internship_Project\cleaned_data\Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# TASK 1:
pipe_lr = Pipeline([
    ('feature_engineering', TelcoFeatureEngineer()),
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

pipe_dt = Pipeline([
    ('feature_engineering', TelcoFeatureEngineer()),
    ('preprocessing', preprocessor),   # trees don't need scaling
    ('model', DecisionTreeClassifier(random_state=42, max_depth=6))
])

pipe_knn = Pipeline([
    ('feature_engineering', TelcoFeatureEngineer()),
    ('preprocessing', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=15))
])

results = []
pipes={}

for name, pipe in [('Logistic Regression', pipe_lr), ('Decision Tree', pipe_dt), ('k-NN', pipe_knn)]:
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    
    preds = pipe.predict(X_test)
    acc=accuracy_score(y_test, preds)
    f1= f1_score(y_test, preds)
    results.append({'Model': name, 'Accuracy': acc, 'F1': f1, 'Train_time_sec': train_time})
    pipes[name] = pipe
    
results_df = pd.DataFrame(results)
print("\nCOMPARISON TABLE")
print(results_df.to_string(index=False))

print("""
Note: Logistic Regression got the best accuracy and F1 here. This makes sense
because most of the churn signal in this data (contract type, tenure, internet
service) has a fairly straightforward relationship with churn - Logistic
Regression is good at picking up on that kind of pattern. k-NN did slightly
worse, probably because it's more sensitive to having lots of features
(41 after encoding) and doesn't handle high-dimensional data as well.
""")


# TASK 2:    
feature_names = pipe_lr.named_steps['preprocessing'].get_feature_names_out()

lr_model = pipe_lr.named_steps['model']
coefs = pd.Series(lr_model.coef_[0], index=feature_names).sort_values(ascending=False)

print("\nLogistic Regression: Top 5 toward CHURN")
print(coefs.head(5))
print("\nLogistic Regression: Top 5 toward RETENTION")
print(coefs.tail(5))

dt_model = pipe_dt.named_steps['model']
importances = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=False)

print("\nDecision Tree: Top 10 Feature Importances")
print(importances.head(10))

plt.figure(figsize=(8, 5))
importances.head(10).sort_values().plot(kind='barh', color='#4C72B0')
plt.title('Top 10 Decision Tree Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('dt_feature_importance.png', bbox_inches='tight')
plt.show()

lr_top3 = coefs.head(3).index.tolist()
dt_top3 = importances.head(3).index.tolist()

print("\n=== TOP 3 COMPARISON ===")
print("Logistic Regression top 3 (toward churn):", lr_top3)
print("Decision Tree top 3:", dt_top3)

overlap = set(lr_top3) & set(dt_top3)
print(f"\nOverlap: {overlap if overlap else 'None'}")

print("""
Note: LR and the Decision Tree agree on 2 of the top 3 features -
Contract_Month-to-month and Fiber optic internet. Both models think these
are big churn drivers.

Where they disagree: PaymentMethod is LR's #1 feature but barely shows up
for the tree. This is probably because the tree used Contract_Month-to-month
so heavily for its first split that it didn't "need" PaymentMethod as much
afterward. LR doesn't work that way - it gives every feature its own weight
at the same time, so PaymentMethod's effect still shows up clearly there.
So this isn't really a disagreement about what matters, it's more about how
each model type distributes credit between correlated features.
""")

# TASK 3:
print("\nCLASS BALANCE CHECK")
print(df['Churn'].value_counts(normalize=True))

print("""
Note: Churn is imbalanced - only about 27% of customers churned. This matters
because a lazy model that just predicts "No churn" every single time would
still get ~73% accuracy, without catching a single real churner. That's why
F1 score (which cares about catching the minority class) is noticeably lower
than accuracy for all three models. """)

