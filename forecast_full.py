import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

# Load CSV file
#file_path = r"C:\Users\oscar\hackathonv2\hackathon\data\challenge_3\train\forecast_full.csv"
df = pd.read_csv("data/train/forecast_full.csv", parse_dates=["date"])


# Select only the required columns
cols = ["date", "category", "state", "age_group", "gender", "expected_volume", "is_weekend", "is_holiday", "expected_volume_lag_1",
        "expected_volume_lag_7", "expected_volume_lag_14", "expected_volume_lag_28", "expected_volume_roll_mean_7",
        "expected_volume_roll_mean_14", "expected_volume_roll_mean_28"]
df = df[cols + ["actual_volume"]]  # assuming 'actual_volume' is our prediction target

# Feature engineering: date components
df["day"] = df.date.dt.day
df["month"] = df.date.dt.month
df["year"] = df.date.dt.year
df["weekday"] = df.date.dt.weekday

# Encode categorical variables using LabelEncoder
cat_features = ["category", "state", "age_group", "gender", "is_weekend", "is_holiday"]
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

for lag in [1, 7, 14, 28]:
    df[f'expected_volume_lag_{lag}'] = df['expected_volume'].shift(lag)

df['expected_volume_roll_mean_7'] = df['expected_volume'].rolling(7).mean()
df['expected_volume_roll_mean_14'] = df['expected_volume'].rolling(14).mean()
df['expected_volume_roll_mean_28'] = df['expected_volume'].rolling(28).mean()


# Define features and target
FEATURES = ["category", "state", "age_group", "gender", "expected_volume", "day", "month", "year", "weekday", "is_weekend", "is_holiday",
            "expected_volume_lag_1", "expected_volume_lag_7", "expected_volume_lag_14", "expected_volume_lag_28", "expected_volume_roll_mean_7",
"expected_volume_roll_mean_14", "expected_volume_roll_mean_28"]
TARGET = "actual_volume"
X = df[FEATURES]
y = df[TARGET]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# XGBoost params
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# Train model
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=20,
    verbose_eval=10
)

# Evaluate
y_pred = bst.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mas = mean_absolute_error(y_val, y_pred)
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation MAE: {mas:.2f}")


# Save the model
model_path = os.path.join(os.getcwd(), "xgb_model_forecast_full.json")
bst.save_model(model_path)
print(f"Model saved to {model_path}")

# Load the model later when needed
# This would typically be placed in a prediction script or another pipeline
loaded_model = xgb.Booster()
loaded_model.load_model(model_path)
print("Model loaded successfully for future predictions.")

# Example of using the loaded model for prediction:
# predictions = loaded_model.predict(xgb.DMatrix(X_val))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_val)), y_val.values, label='Actual', alpha=0.7)
plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Volume')
plt.xlabel('Sample Index')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

