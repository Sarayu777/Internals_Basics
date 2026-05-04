import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")
combined_df = pd.concat([train_df, new_df], ignore_index=True)

original_rows = len(train_df)
new_rows = len(new_df)
combined_rows = len(combined_df)

features = ["area_sqft", "bedrooms", "floor_level", "locality_score"]

# Use training data for test set (same test set for fair comparison)
X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    train_df[features], train_df["rental_price"],
    test_size=0.2, random_state=42
)

# Champion model evaluation
with open("models/best_model.pkl", "rb") as f:
    champion = pickle.load(f)

champ_preds = champion.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, champ_preds))
print(f"Champion RMSE: {champion_rmse:.2f}")

# Retrain on combined data
X_combined = combined_df[features]
y_combined = combined_df["rental_price"]

retrained = RandomForestRegressor(
    n_estimators=250,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)
retrained.fit(X_combined, y_combined)

retrain_preds = retrained.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, retrain_preds))
print(f"Retrained RMSE: {retrained_rmse:.2f}")

improvement = champion_rmse - retrained_rmse
print(f"Improvement: {improvement:.2f}")

# Promote decision
if improvement >= 0.5:
    action = "promoted"
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(retrained, f)
    print("Model promoted!")
else:
    action = "kept_champion"
    print("Champion kept.")

# Save result JSON
os.makedirs("results", exist_ok=True)
output = {
    "original_data_rows": original_rows,
    "new_data_rows": new_rows,
    "combined_data_rows": combined_rows,
    "champion_rmse": champion_rmse,
    "retrained_rmse": retrained_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse"
}
with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved results/step4_s8.json")