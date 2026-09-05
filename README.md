# Lifestyle Disease Risk Analysis

Simple Python data science project analyzing cardiovascular disease risk
based on lifestyle and biometric factors, with an interactive dashboard
and individual risk calculator.

## Dataset
[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
by sulianova (70,000 records). Download `cardio_train.csv` and place it
inside `data/`.

## Structure
- `analysis.py` — data loading, cleaning, stats, graphs
- `model.py` — Logistic Regression model
- `app.py` — Streamlit dashboard + risk calculator
- `requirements.txt` — dependencies

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```