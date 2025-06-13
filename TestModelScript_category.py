import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# Load test data
file_path = "data/test/forecast_category.csv"
df_test = pd.read_csv(file_path, parse_dates=["date"])

# Ensure required columns are present
required_columns = ["date", "category", "expected_volume"]
if not all(col in df_test.columns for col in required_columns):
    raise ValueError(f"Missing required columns in test data: {required_columns}")

# Feature engineering: date components
df_test["day"] = df_test.date.dt.day
df_test["month"] = df_test.date.dt.month
df_test["year"] = df_test.date.dt.year
df_test["weekday"] = df_test.date.dt.weekday

# Encode categorical variables (must match training encoders)
label_encoders = {}  # Replace this with saved encoders if available
le = LabelEncoder()
df_test["category"] = le.fit_transform(df_test["category"])
label_encoders["category"] = le

# Generate lag and rolling features (with caution for edge rows)
for lag in [1, 7, 14, 28]:
    df_test[f'expected_volume_lag_{lag}'] = df_test['expected_volume'].shift(lag)

df_test['expected_volume_roll_mean_7'] = df_test['expected_volume'].rolling(7).mean()
df_test['expected_volume_roll_mean_14'] = df_test['expected_volume'].rolling(14).mean()
df_test['expected_volume_roll_mean_28'] = df_test['expected_volume'].rolling(28).mean()

# Drop rows with NaN from lag/rolling features
df_test.dropna(inplace=True)

# Define features used in training
FEATURES = [
    "category", "expected_volume", "day", "month", "year", "weekday",
    "expected_volume_lag_1", "expected_volume_lag_7", "expected_volume_lag_14",
    "expected_volume_lag_28", "expected_volume_roll_mean_7",
    "expected_volume_roll_mean_14", "expected_volume_roll_mean_28"
]

# Prepare feature matrix for prediction
X_test = df_test[FEATURES]
dtest = xgb.DMatrix(X_test)

# Load trained model
model_path = os.path.join(os.getcwd(), "xgb_model_forecast_category.json")
loaded_model = xgb.Booster()
loaded_model.load_model(model_path)

# Predict
y_pred = loaded_model.predict(dtest)
df_test["actual_volume"] = y_pred

# Save predictions
output_path = os.path.join(os.getcwd(), "challenge3_category_submission.csv")
df_test[["date", "category", "expected_volume", "actual_volume"]].to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")


