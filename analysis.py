import pandas as pd

DATA_PATH = "data/cardio_train.csv"

def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")

    df["age_years"] = (df["age"] / 365).astype(int)
    df["bmi"] = round(df["weight"] / ((df["height"] / 100) ** 2), 1)

    df = df[(df["ap_hi"] > 60) & (df["ap_hi"] < 240)]
    df = df[(df["ap_lo"] > 40) & (df["ap_lo"] < 160)]
    df = df[(df["height"] > 100) & (df["height"] < 220)]
    df = df[(df["weight"] > 30) & (df["weight"] < 200)]
    df = df[(df["bmi"] > 10) & (df["bmi"] < 60)]     # ← ye line hai ya nahi check karo

    df["risk_label"] = df["cardio"].map({0: "No Disease", 1: "Disease"})
    df["smoke_label"] = df["smoke"].map({0: "Non-Smoker", 1: "Smoker"})
    df["active_label"] = df["active"].map({0: "Inactive", 1: "Active"})

    return df


def basic_stats(df):
    return {
        "Total Records": len(df),
        "Average Age": round(df["age_years"].mean(), 1),
        "Average BMI": round(df["bmi"].mean(), 1),
        "Disease %": round(df["cardio"].mean() * 100, 1),
    }