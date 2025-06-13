# --- Prediction Phase: Apply the trained model to new test data ---
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# Load test data
new_test_path = "data/test/forecast_full.csv"  # <-- Update path to your new test data
df_test = pd.read_csv(new_test_path, parse_dates=["date"])

# Apply same preprocessing as training data
cat_features = ["category", "state", "age_group", "gender", "is_weekend", "is_holiday"]
for col in cat_features:
    if col in df_test.columns:
        le = LabelEncoder()
        df_test[col] = le.fit_transform(df_test[col].astype(str))  # use .astype(str) to handle NaNs safely

# Feature engineering: date components
df_test["day"] = df_test.date.dt.day
df_test["month"] = df_test.date.dt.month
df_test["year"] = df_test.date.dt.year
df_test["weekday"] = df_test.date.dt.weekday

# Define the same FEATURES list used during training
FEATURES = ["category", "state", "age_group", "gender", "expected_volume", "day", "month", "year", "weekday", "is_weekend", "is_holiday",
            "expected_volume_lag_1", "expected_volume_lag_7", "expected_volume_lag_14", "expected_volume_lag_28", "expected_volume_roll_mean_7",
"expected_volume_roll_mean_14", "expected_volume_roll_mean_28"]

# Ensure no missing values in required features
if df_test[FEATURES].isnull().any().any():
    raise ValueError("Test data contains NaNs in required features. Please handle them before prediction.")

# Load model
model_path = os.path.join(os.getcwd(), "xgb_model_forecast_full.json")
loaded_model = xgb.Booster()
loaded_model.load_model(model_path)

# Prepare DMatrix and predict
dtest = xgb.DMatrix(df_test[FEATURES])
predicted_volume = loaded_model.predict(dtest)

# Add predictions to DataFrame
df_test["actual_volume"] = predicted_volume

# Optionally save to CSV
output_path = "data/test/challenge3_full_submission.csv"
df_test.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
