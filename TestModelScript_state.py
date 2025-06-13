import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# Load test CSV
test_path = "data/test/forecast_state.csv"
df_test = pd.read_csv(test_path, parse_dates=["date"])

# Select and engineer features (same as training)
cols = ["date", "state", "is_weekend", "is_holiday", "expected_volume"]
df_test = df_test[cols].copy()

df_test["day"] = df_test.date.dt.day
df_test["month"] = df_test.date.dt.month
df_test["year"] = df_test.date.dt.year
df_test["weekday"] = df_test.date.dt.weekday

# Encode categorical variables (re-use logic from training)
cat_features = ["state", "is_weekend", "is_holiday"]
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df_test[col] = le.fit_transform(df_test[col])  # Warning: assumes test has all categories seen in train
    label_encoders[col] = le

# Create lag features (pad with NaNs)
for lag in [1, 7, 14, 28]:
    df_test[f'expected_volume_lag_{lag}'] = df_test['expected_volume'].shift(lag)

df_test['expected_volume_roll_mean_7'] = df_test['expected_volume'].rolling(7).mean()
df_test['expected_volume_roll_mean_14'] = df_test['expected_volume'].rolling(14).mean()
df_test['expected_volume_roll_mean_28'] = df_test['expected_volume'].rolling(28).mean()

# Drop rows with NaNs due to lag/rolling
df_test = df_test.dropna().reset_index(drop=True)

# Features used in training
FEATURES = ["state", "is_weekend", "is_holiday", "day", "month", "year", "weekday", "expected_volume",
            "expected_volume_lag_1", "expected_volume_lag_7", "expected_volume_lag_14", "expected_volume_lag_28",
            "expected_volume_roll_mean_7", "expected_volume_roll_mean_14", "expected_volume_roll_mean_28"]

X_test = df_test[FEATURES]

# Load model
model_path = os.path.join(os.getcwd(), "xgb_model_forecast_state.json")
booster = xgb.Booster()
booster.load_model(model_path)

# Predict
dtest = xgb.DMatrix(X_test, enable_categorical=True)
y_pred = booster.predict(dtest)

# Save predictions
df_test["actual_volume"] = y_pred
output_path = "challenge3_state_submission.csv"
df_test.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")