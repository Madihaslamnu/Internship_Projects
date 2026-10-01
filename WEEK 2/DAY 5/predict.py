from pathlib import Path
import numbers
import sys

import joblib
import pandas as pd


sys.path.insert(0, str(Path(__file__).parents[1] / "DAY 2"))
PIPELINE_PATH = Path(__file__).with_name("tuned_pipeline.joblib")
pipeline = joblib.load(PIPELINE_PATH)

REQUIRED_FIELDS = {
    "gender": str,
    "SeniorCitizen": int,
    "Partner": str,
    "Dependents": str,
    "tenure": numbers.Real,
    "PhoneService": str,
    "MultipleLines": str,
    "InternetService": str,
    "OnlineSecurity": str,
    "OnlineBackup": str,
    "DeviceProtection": str,
    "TechSupport": str,
    "StreamingTV": str,
    "StreamingMovies": str,
    "Contract": str,
    "PaperlessBilling": str,
    "PaymentMethod": str,
    "MonthlyCharges": numbers.Real,
    "TotalCharges": numbers.Real,
}


def predict_churn(customer_dict: dict) -> dict:
    # Return churn probability and a business friendly risk tier.
    missing = [field for field in REQUIRED_FIELDS if field not in customer_dict]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    for field, expected_type in REQUIRED_FIELDS.items():
        value = customer_dict[field]
        if expected_type is numbers.Real:
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError(
                    f"{field} must be a number; received {type(value).__name__}"
                )
        elif not isinstance(value, expected_type):
            raise TypeError(
                f"{field} must be {expected_type.__name__}; "
                f"received {type(value).__name__}"
            )

    if customer_dict["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1")
    if customer_dict["tenure"] < 0:
        raise ValueError("tenure cannot be negative")
    if customer_dict["tenure"] > 100:
        raise ValueError("tenure cannot be greater than 100 months")
    if customer_dict["MonthlyCharges"] < 0:
        raise ValueError("MonthlyCharges cannot be negative")
    if customer_dict["TotalCharges"] < 0:
        raise ValueError("TotalCharges cannot be negative")

    probability = float(
        pipeline.predict_proba(pd.DataFrame([customer_dict]))[0, 1]
    )
    if probability < 0.30:
        risk_tier = "Low"
    elif probability < 0.60:
        risk_tier = "Medium"
    else:
        risk_tier = "High"
    return {
        "churn_probability": round(probability, 4),
        "risk_tier": risk_tier,
    }


if __name__ == "__main__":
    example = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0, "TotalCharges": 170.0,
    }
    print(predict_churn(example))
    try:
        predict_churn({**example, "tenure": -1})
    except ValueError as error:
        print(f"Expected validation error: {error}")
