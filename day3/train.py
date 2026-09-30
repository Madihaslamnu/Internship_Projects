
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from day3.data_utili import ID2LABEL, LABELS, get_split
from text_preprocessing import preprocess

warnings.filterwarnings("ignore")
ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)


def make_tfidf() -> TfidfVectorizer:
    # Preprocessing already tokenised + lowercased -> just split on spaces.
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, sublinear_tf=True)


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        **{f"f1_{l}": round(v, 4) for l, v in zip(LABELS, f1_score(y_true, y_pred, average=None, labels=LABELS))},
    }
def batch_preprocess(col):
  return [preprocess(t) for t in col]

def main() -> None:
    train, test = get_split()
    y_tr, y_te = train["airline_sentiment"], test["airline_sentiment"]
    print(f"train={len(train)}  test={len(test)}")

    t0 = time.time()
    X_tr = [preprocess(text) for text in train["text"]]
    X_te = [preprocess(text) for text in test["text"]]
    print(f"preprocessing: {time.time() - t0:.1f}s")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    ngrams = {"tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)], "tfidf__min_df": [2]}
    candidates = {
        "logreg": (LogisticRegression(max_iter=3000),
                   {"clf__C": [3, 10, 30], "clf__class_weight": [None, "balanced"]}),
        "linear_svm": (LinearSVC(),
                       {"clf__C": [0.1, 0.3, 1], "clf__class_weight": [None, "balanced"]}),
    }

    metrics: dict = {"n_train": len(train), "n_test": len(test), "models": {}}

    # --- reference points -------------------------------------------------
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    metrics["models"]["majority_class"] = {"test": scores(y_te, dummy.predict(X_te))}

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    comp = test["text"].map(lambda t: sia.polarity_scores(t)["compound"])
    vader = np.where(comp >= 0.05, "positive", np.where(comp <= -0.05, "negative", "neutral"))
    metrics["models"]["vader_lexicon"] = {"test": scores(y_te, vader)}

    # --- tuned baselines ---------------------------------------------------
    best_name, best_cv, best_params = None, -1.0, None
    for name, (clf, grid) in candidates.items():
        t0 = time.time()
        gs = GridSearchCV(
            Pipeline([("tfidf", make_tfidf()), ("clf", clf)]),
            {**ngrams, **grid}, cv=cv, scoring="f1_macro", n_jobs=-1,
        ).fit(X_tr, y_tr)
        pred = gs.predict(X_te)
        metrics["models"][name] = {
            "cv_macro_f1": round(gs.best_score_, 4),
            "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()},
            "test": scores(y_te, pred),
        }
        print(f"{name:11s} CV macro-F1={gs.best_score_:.4f}  test={metrics['models'][name]['test']}  "
              f"params={gs.best_params_}  ({time.time() - t0:.0f}s)", flush=True)
        if gs.best_score_ > best_cv:
            best_name, best_cv, best_params = name, gs.best_score_, gs.best_params_

    # --- choose winner by CV (never by test) --------------------------------
    metrics["selected_by_cv"] = best_name
    clf = candidates[best_name][0].set_params(
        **{k.split("__")[1]: v for k, v in best_params.items() if k.startswith("clf__")})
    tfidf = make_tfidf().set_params(**{k.split("__")[1]: v for k, v in best_params.items() if k.startswith("tfidf__")})

    # Out-of-fold predictions on TRAIN for error analysis (test stays untouched).
    oof = cross_val_predict(Pipeline([("tfidf", tfidf), ("clf", clf)]), X_tr, y_tr, cv=cv, n_jobs=-1)
    oof_df = train[["text", "airline", "airline_sentiment", "airline_sentiment_confidence"]].copy()
    oof_df["pred"] = oof
    oof_df.to_csv(ART / "oof_train_predictions.csv", index=True, index_label="row_id")
    metrics["oof_train"] = scores(y_tr, oof)


    final = Pipeline([
        ("prep", FunctionTransformer(batch_preprocess
                                     , validate=False)),
        ("tfidf", tfidf),
        ("clf", clf),
    ]).fit(train["text"], y_tr)
    test_pred = final.predict(test["text"])
    pd.DataFrame({"text": test["text"].values, "true": y_te.values, "pred": test_pred},
                 index=test.index).to_csv(ART / "test_predictions.csv", index_label="row_id")
    metrics["final_test"] = {"model": best_name, **scores(y_te, test_pred)}

    joblib.dump(final, ART / "baseline.joblib")
    (ART / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nselected:", best_name, "->", metrics["final_test"])


if __name__ == "__main__":
    main()