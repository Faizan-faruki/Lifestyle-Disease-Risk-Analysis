import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analysis import load_data

FEATURES = [
    "age_years", "gender", "height", "weight", "bmi",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]


def train_model():
    df = load_data()
    X = df[FEATURES]
    y = df["cardio"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, scaler, accuracy


def predict_risk(model, scaler, input_dict):
    """input_dict keys must match FEATURES order/content."""
    row = pd.DataFrame([input_dict])[FEATURES]
    row_scaled = scaler.transform(row)

    prob = model.predict_proba(row_scaled)[0][1]  # probability of cardio=1

    if prob < 0.33:
        category = "LOW"
    elif prob < 0.66:
        category = "MODERATE"
    else:
        category = "HIGH"

    return category, round(prob * 100, 1)