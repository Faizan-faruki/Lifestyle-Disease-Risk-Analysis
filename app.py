import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import load_data, basic_stats
from model import train_model, predict_risk

# STEP 1: Basic page setup
st.set_page_config(page_title="Lifestyle Disease Risk", page_icon="🫀", layout="wide")

# STEP 2: Load the data
df = load_data()

# STEP 3: Sidebar navigation
st.sidebar.title("🫀 Lifestyle Disease Risk")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Dashboard", "🧮 Risk Calculator", "ℹ️ About Project"])

# ================= PAGE 1: HOME =================
if page == "🏠 Home":
    st.title("🫀 Lifestyle Disease Risk Analysis")

    st.write(
        "Heart disease is one of the leading causes of death worldwide, and many "
        "of its risk factors — like high BMI, high blood pressure, and high blood "
        "sugar — are tied to everyday lifestyle choices such as diet, exercise, "
        "smoking, and alcohol use. This project uses real patient data to explore "
        "these patterns and understand how different lifestyle and health factors "
        "connect to heart disease risk. It also includes a simple calculator that "
        "estimates an individual's risk level based on their own health and "
        "lifestyle details."
    )

    # ---------- Simple explanations of all terms used in this project ----------
    st.subheader("📖 Terms Used in This Project")
    st.write("New to these medical terms? Here's a little explanation.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎂 Age")
        st.write(
            "How old the person is, in years. Disease risk generally increases with "
            "age, since the heart and blood vessels wear down over time."
        )

        st.markdown("### ⚖️ BMI (Body Mass Index)")
        st.write(
            "A number calculated from height and weight (weight ÷ height²) that "
            "shows if someone is underweight, normal, overweight, or obese. Below "
            "18.5 is underweight, 18.5–24.9 is normal, 25–29.9 is overweight, and "
            "30+ is obese."
        )

        st.markdown("### 💓 Systolic Blood Pressure")
        st.write(
            "The top number in a BP reading (e.g. the '120' in 120/80). It measures "
            "pressure in your blood vessels when your heart beats. Normal is around "
            "90-120 — higher means your heart is working harder than it should."
        )

        st.markdown("### 💗 Diastolic Blood Pressure")
        st.write(
            "The bottom number in a BP reading (e.g. the '80' in 120/80). It "
            "measures pressure when your heart rests between beats. Normal is "
            "around 60-80."
        )

        st.markdown("### 🩸 Cholesterol")
        st.write(
            "A fatty substance in your blood that can build up and clog vessels. "
            "'Normal' is a healthy level, while 'Above Normal' and 'Well Above "
            "Normal' mean elevated levels found in a blood test."
        )

    with col2:
        st.markdown("### 🍬 Glucose")
        st.write(
            "Your blood sugar level, checked through a blood test. 'Normal' is "
            "healthy, while higher levels are often linked to pre-diabetes or "
            "diabetes."
        )

        st.markdown("### 🚬 Smoking")
        st.write(
            "Whether the person currently smokes tobacco. Smoking damages blood "
            "vessels and the heart, and is one of the biggest risk factors for "
            "heart disease."
        )

        st.markdown("### 🍷 Alcohol Intake")
        st.write(
            "Whether the person regularly drinks alcohol. Frequent or heavy "
            "drinking can raise blood pressure and strain the heart over time."
        )

        st.markdown("### 🏃 Physical Activity")
        st.write(
            "Whether the person gets regular exercise or movement in daily life. "
            "Being active strengthens the heart, while being inactive raises "
            "disease risk."
        )

        st.markdown("### 🫀 Cardiovascular Disease")
        st.write(
            "A general term for diseases affecting the heart and blood vessels, "
            "like heart attacks and strokes. This project estimates the likelihood "
            "of having this kind of disease."
        )

    st.markdown("---")


# ================= PAGE 2: DASHBOARD =================
# Each chart below follows the same simple pattern:
# 1. Show a dropdown filter
# 2. Filter the data based on the selection
# 3. Group values into ranges (bins) so each bar/slice represents enough people
# 4. Drop groups with too few people (unreliable percentage)
# 5. Draw the chart from that summary
elif page == "📊 Dashboard":
    st.title("📊 Dashboard")

    # ---------- Top KPI numbers ----------
    stats = basic_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{stats['Total Records']:,}")
    c2.metric("Average Age", stats["Average Age"])
    c3.metric("Average BMI", stats["Average BMI"])
    c4.metric("Disease %", f"{stats['Disease %']}%")

    st.markdown("---")

    MIN_GROUP_SIZE = 30  # ignore groups with fewer people than this (unreliable %)

    # ---------- Chart 1: Disease Risk % by Age ----------
    st.subheader("🎂 Disease Risk % by Age")
    age_gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female"], key="age_gender")

    filtered1 = df if age_gender_filter == "All" else df[df["gender"] == (2 if age_gender_filter == "Male" else 1)]

    age_risk = filtered1.groupby("age_years")["cardio"].agg(["mean", "count"]).reset_index()
    age_risk["risk_percent"] = (age_risk["mean"] * 100).round(1)
    age_risk = age_risk[age_risk["count"] >= MIN_GROUP_SIZE]

    fig1 = px.bar(
        age_risk, x="age_years", y="risk_percent",
        color="risk_percent", color_continuous_scale="RdYlGn_r",
        title="Disease Risk % at Each Age",
        template="plotly_dark",
    )
    fig1.update_layout(
        height=500, xaxis_title="Age (years)", yaxis_title="Risk %",
        coloraxis_showscale=False, bargap=0.05,
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ---------- Chart 2: Disease Risk % by BMI ----------
    st.subheader("⚖️ Disease Risk % by BMI")
    activity_filter = st.selectbox("Filter by Activity", ["All", "Active", "Inactive"], key="bmi_active")

    filtered2 = df if activity_filter == "All" else df[df["active_label"] == activity_filter]
    filtered2 = filtered2.copy()

    filtered2["bmi_group"] = (filtered2["bmi"] // 2) * 2

    bmi_risk = filtered2.groupby("bmi_group")["cardio"].agg(["mean", "count"]).reset_index()
    bmi_risk["risk_percent"] = (bmi_risk["mean"] * 100).round(1)
    bmi_risk = bmi_risk[bmi_risk["count"] >= MIN_GROUP_SIZE]

    fig2 = px.bar(
        bmi_risk, x="bmi_group", y="risk_percent",
        color="risk_percent", color_continuous_scale="RdYlGn_r",
        title="Disease Risk % at Each BMI Range",
        template="plotly_dark",
    )
    fig2.update_layout(
        height=500, xaxis_title="BMI (range)", yaxis_title="Risk %",
        coloraxis_showscale=False, bargap=0.1,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ---------- Chart 3: Disease Risk % by Glucose Level (Donut) ----------
    st.subheader("🩸 Disease Risk % by Glucose Level")
    gluc_gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female"], key="gluc_gender")

    filtered3 = df if gluc_gender_filter == "All" else df[df["gender"] == (2 if gluc_gender_filter == "Male" else 1)]

    gluc_risk = filtered3.groupby("gluc")["cardio"].agg(["mean", "count"]).reset_index()
    gluc_risk["risk_percent"] = (gluc_risk["mean"] * 100).round(1)
    gluc_risk["gluc_label"] = gluc_risk["gluc"].map({1: "Normal", 2: "Above Normal", 3: "Well Above Normal"})
    gluc_risk = gluc_risk[gluc_risk["count"] >= MIN_GROUP_SIZE]

    fig3 = px.pie(
        gluc_risk, names="gluc_label", values="risk_percent",
        hole=0.5, title="Disease Risk % by Glucose Level",
        color="gluc_label",
        color_discrete_map={
            "Normal": "#4CAF50",
            "Above Normal": "#FF9800",
            "Well Above Normal": "#E53935",
        },
        template="plotly_dark",
    )
    fig3.update_traces(
        textinfo="label+percent", textfont_size=14,
        marker=dict(line=dict(color="#0e1117", width=2)),
    )
    fig3.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ---------- Chart 4: Disease Risk % by Systolic BP ----------
    st.subheader("💓 Disease Risk % by Systolic Blood Pressure")
    smoke_filter = st.selectbox("Filter by Smoking", ["All", "Smoker", "Non-Smoker"], key="bp_smoke")

    filtered4 = df if smoke_filter == "All" else df[df["smoke_label"] == smoke_filter]
    filtered4 = filtered4[(filtered4["ap_hi"] >= 90) & (filtered4["ap_hi"] <= 180)]
    filtered4 = filtered4.copy()

    filtered4["bp_group"] = (filtered4["ap_hi"] // 10) * 10

    bp_risk = filtered4.groupby("bp_group")["cardio"].agg(["mean", "count"]).reset_index()
    bp_risk["risk_percent"] = (bp_risk["mean"] * 100).round(1)
    bp_risk = bp_risk[bp_risk["count"] >= 15]

    fig4 = px.bar(
        bp_risk, x="bp_group", y="risk_percent",
        color="risk_percent", color_continuous_scale="RdYlGn_r",
        title="Disease Risk % at Each Systolic BP Range",
        template="plotly_dark",
    )
    fig4.update_layout(
        height=500, xaxis_title="Systolic BP (range)", yaxis_title="Risk %",
        coloraxis_showscale=False, bargap=0.1,
    )
    st.plotly_chart(fig4, use_container_width=True)


# ================= PAGE 3: RISK CALCULATOR =================
elif page == "🧮 Risk Calculator":
    st.title("🧮 Risk Calculator")
    st.write("Enter your details to get an estimated cardiovascular disease risk.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 18, 100, 40)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", 100, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        ap_hi = st.number_input("Systolic BP", 80, 240, 120)
        ap_lo = st.number_input("Diastolic BP", 50, 160, 80)

    with col2:
        cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
        gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
        smoke = st.selectbox("Smoking", ["No", "Yes"])
        alco = st.selectbox("Alcohol Intake", ["No", "Yes"])
        active = st.selectbox("Physically Active", ["Yes", "No"])

    bmi = round(weight / ((height / 100) ** 2), 1)
    st.info(f"Calculated BMI: **{bmi}**")

    if st.button("Predict Risk", type="primary"):
        model, scaler, metrics = train_model()

        input_dict = {
            "age_years": age,
            "gender": 2 if gender == "Male" else 1,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": ["Normal", "Above Normal", "Well Above Normal"].index(cholesterol) + 1,
            "gluc": ["Normal", "Above Normal", "Well Above Normal"].index(gluc) + 1,
            "smoke": 1 if smoke == "Yes" else 0,
            "alco": 1 if alco == "Yes" else 0,
            "active": 1 if active == "Yes" else 0,
        }

        category, prob = predict_risk(model, scaler, input_dict)

        color = {"LOW": "#4CAF50", "MODERATE": "#FF9800", "HIGH": "#E53935"}[category]
        st.markdown(
            f"<h2 style='color:{color};'>Estimated Risk: {category} ({prob}%)</h2>",
            unsafe_allow_html=True,
        )

        factors = []
        if bmi >= 25:
            factors.append("Higher BMI")
        if ap_hi >= 130 or ap_lo >= 85:
            factors.append("Elevated Blood Pressure")
        if smoke == "Yes":
            factors.append("Smoking")
        if active == "No":
            factors.append("Low Physical Activity")
        if cholesterol != "Normal":
            factors.append("High Cholesterol")
        if gluc != "Normal":
            factors.append("High Glucose")

        if factors:
            st.write("**Main associated factors:**")
            for f in factors:
                st.write(f"• {f}")
        else:
            st.write("No major risk factors detected — keep it up!")

        st.write("**Suggested awareness:**")
        st.write("• Increase physical activity")
        st.write("• Maintain a healthy weight")
        st.write("• Avoid tobacco and limit alcohol")
        st.write("• Get BP/glucose/cholesterol checked regularly")

        st.caption(
    f"Model performance — "
    f"Accuracy: {round(metrics['accuracy']*100, 1)}% | "
    f"Precision: {round(metrics['precision']*100, 1)}% | "
    f"Recall: {round(metrics['recall']*100, 1)}% | "
    f"ROC-AUC: {round(metrics['roc_auc'], 3)}"
)


# ================= PAGE 4: ABOUT PROJECT =================
else:
    st.title(" About This Project")

    st.markdown("""
### 🫀 Lifestyle Disease Risk Analysis

This project studies how everyday lifestyle and biometric factors — such as
age, weight, blood pressure, cholesterol, blood sugar, smoking, alcohol use,
and physical activity — relate to a person's risk of cardiovascular (heart)
disease. It combines data analysis, an interactive dashboard, and a machine
learning based risk calculator into one simple tool.

### 📂 Dataset

**Cardiovascular Disease Dataset** — Kaggle, by sulianova
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

The dataset contains **70,000 real patient records** with the following information:

| Column | Meaning |
|---|---|
| age | Age of the patient (in days, converted to years) |
| gender | Male or Female |
| height, weight | Used to calculate BMI |
| ap_hi, ap_lo | Systolic and Diastolic blood pressure |
| cholesterol | Normal / Above Normal / Well Above Normal |
| gluc | Blood glucose level: Normal / Above Normal / Well Above Normal |
| smoke | Whether the person smokes |
| alco | Whether the person drinks alcohol |
| active | Whether the person is physically active |
| cardio | Target label — whether the person has cardiovascular disease (0 = No, 1 = Yes) |

Before analysis, the data is cleaned to remove impossible values — for
example, unrealistic heights, weights, blood pressure readings, and extreme
BMI outliers that would otherwise distort the charts.

### 🤖 Machine Learning Model

**Model used:** Logistic Regression

Logistic Regression was chosen because this is a **binary classification
problem** — the model only needs to predict one of two outcomes: *disease*
or *no disease*. It's simple, fast to train, and its predictions are easy to
interpret compared to more complex models, which matters for a health-related
tool where explainability is important.

**How it works, step by step:**
1. All 12 input features (age, gender, height, weight, BMI, systolic BP,
   diastolic BP, cholesterol, glucose, smoking, alcohol, activity) are
   collected for each patient.
2. The features are **scaled** using `StandardScaler`, so that features with
   larger numeric ranges (like blood pressure) don't unfairly dominate
   features with smaller ranges (like smoking, which is just 0 or 1).
3. The dataset is split — **80% for training** the model, **20% for testing**
   how well it performs on data it hasn't seen before.
4. Logistic Regression learns a weight for each feature, indicating how
   strongly it pushes the prediction toward "disease" or "no disease."
5. For a new person's input, the model outputs a **probability** (0–100%) of
   having cardiovascular disease. This probability is converted into three
   simple risk bands:
   - **Below 33% → LOW risk**
   - **33-66% → MODERATE risk**
   - **Above 66% → HIGH risk**

### Model Performance

To check how good the model actually is, four different scores are used:

- **Accuracy (72.2%)** — Out of all predictions the model made, how many were correct overall.
- **Precision (74.9%)** — When the model predicts that someone has the disease, how often is it actually right. A higher precision means fewer false alarms.
- **Recall (65.7%)** — Out of all the people who actually have the disease, how many the model successfully identified. A higher recall means fewer missed cases.
- **ROC-AUC (0.788)** — A single score (between 0 and 1) that shows how well the model can tell the difference between "disease" and "no disease" across all possible thresholds. 0.5 means random guessing; 0.788 means the model performs meaningfully better than chance.

### 🛠 Technologies Used

- **Python** — main programming language
- **Pandas** — data cleaning and analysis
- **Scikit-learn** — Logistic Regression model, scaling, train/test splitting
- **Plotly** — interactive charts on the dashboard
- **Streamlit** — the web app/dashboard itself

### 🎯 Main Objectives

- Analyze how age, BMI, blood pressure, and glucose relate to disease risk
- Present findings through clear, interactive visual charts
- Build a working ML model that predicts individual risk
- Make medical terms and results understandable to non-experts
- Encourage awareness of preventable, lifestyle-linked risk factors

### 💡 Why This Project Is Useful

A large share of cardiovascular disease risk comes from modifiable lifestyle
factors rather than genetics alone. By turning raw medical data into an
interactive, easy-to-understand tool, this project helps people see which
habits matter most and get a rough, personalized sense of their own risk —
without needing to understand statistics or machine learning themselves.

### 📈 Key Benefits

- Clear visual breakdown of major risk factors
- Instant, personalized risk estimation
- Simple explanations of medical terms for non-technical users
- A lightweight, fully open-source project built entirely in Python
""")
