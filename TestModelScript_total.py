import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# Load test CSV
# file_path = r"C:\Users\oscar\hackathonv2\hackathon\data\challenge_3\test\forecast_total.csv"
df = pd.read_csv("data/test/forecast_total.csv", parse_dates=["date"])

# Feature engineering
cols = ["date", "is_weekend", "is_holiday", "expected_volume"]
df = df[cols]
df["day"] = df.date.dt.day
df["month"] = df.date.dt.month
df["year"] = df.date.dt.year
df["weekday"] = df.date.dt.weekday

# Encode categorical variables using LabelEncoder (should match training encoding)
cat_features = ["is_weekend", "is_holiday"]
for col in cat_features:
    df[col] = LabelEncoder().fit_transform(df[col])  # Replace with trained encoder if available

# Lag features (shift introduces NaNs that need handling)
for lag in [1, 7, 14, 28]:
    df[f'expected_volume_lag_{lag}'] = df['expected_volume'].shift(lag)

df['expected_volume_roll_mean_7'] = df['expected_volume'].rolling(7).mean()
df['expected_volume_roll_mean_14'] = df['expected_volume'].rolling(14).mean()
df['expected_volume_roll_mean_28'] = df['expected_volume'].rolling(28).mean()

# Drop initial rows with NaNs from lag/rolling
df = df.dropna().reset_index(drop=True)

# Define feature columns
FEATURES = [
    "is_weekend", "is_holiday", "day", "month", "year", "weekday", "expected_volume",
    "expected_volume_lag_1", "expected_volume_lag_7", "expected_volume_lag_14", "expected_volume_lag_28",
    "expected_volume_roll_mean_7", "expected_volume_roll_mean_14", "expected_volume_roll_mean_28"
]

X_test = df[FEATURES]

# Load model
model_path = os.path.join(os.getcwd(), "xgb_model_forecast_total.json")
model = xgb.Booster()
model.load_model(model_path)

# Predict
dtest = xgb.DMatrix(X_test, enable_categorical=True)
y_pred = model.predict(dtest)

# Add predictions to dataframe and export
output = df[["date"]].copy()
output["actual_volume"] = y_pred
output.to_csv("challenge3_total_submission.csv", index=False)
print("Predictions saved to forecast_total_predictions.csv")